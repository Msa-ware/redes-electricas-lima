"""Carga del dataset de OpenStreetMap y construccion del grafo candidato.

Nodos  : subestaciones (power=substation) y centrales (power=plant) de Lima.
Aristas: posibles tendidos de cable entre cada subestacion y sus k vecinas mas
         cercanas (peso = distancia geodesica por Haversine, en metros).
"""
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .algoritmos.ufds import UFDS

R_TIERRA = 6371008.8          # metros
LAT0 = -12.05                 # latitud de referencia para proyectar Lima a metros
BBOX_LIMA = (-12.55, -11.70, -77.25, -76.60)


def haversine_np(lat1, lon1, lat2, lon2):
    """Distancia geodesica (m) entre arreglos de coordenadas."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_TIERRA * np.arcsin(np.sqrt(a))


def proyectar(lat, lon):
    """Proyeccion equirectangular local (m). Suficiente para buscar vecinos en una ciudad."""
    x = np.radians(np.asarray(lon)) * R_TIERRA * math.cos(math.radians(LAT0))
    y = np.radians(np.asarray(lat)) * R_TIERRA
    return x, y


def _limpiar(s):
    return (s or "").replace(",", " ").replace("\n", " ").strip()


def _vmax(tags):
    try:
        return max(int(x) for x in tags.get("voltage", "").replace(" ", "").split(";") if x.isdigit())
    except ValueError:
        return 0


def _clasificar(tags):
    if tags.get("power") == "plant":
        return "generacion"
    if tags.get("substation") == "transmission" or _vmax(tags) >= 60000:
        return "transmision"
    return "distribucion"


def _numero(texto):
    m = re.search(r"[\d.]+", texto or "")
    return float(m.group()) if m else None


def cargar_osm(ruta):
    """Lee el JSON del grupo (claves 'elementos'/'lineas') o un export de Overpass ('elements')."""
    with open(ruta, encoding="utf-8") as f:
        data = json.load(f)
    if "elementos" in data:
        return data
    elementos, lineas = [], []
    for e in data["elements"]:
        t = e.get("tags", {})
        p = t.get("power")
        if e["type"] == "way" and p in ("line", "minor_line", "cable"):
            if e.get("geometry"):
                lineas.append(e)
        elif p in ("substation", "plant", "tower", "pole", "transformer"):
            if "lat" not in e and "center" not in e and "bounds" in e:
                b = e["bounds"]
                e["center"] = {"lat": (b["minlat"] + b["maxlat"]) / 2, "lon": (b["minlon"] + b["maxlon"]) / 2}
            elementos.append(e)
    return {"meta": {"fuente": "Overpass Turbo (export)"}, "elementos": elementos, "lineas": lineas}


def cargar_nodos(osm):
    """DataFrame de nodos. Para las fuentes (SET y centrales) se guarda la
    potencia nominal declarada en OSM (rating / plant:output:electricity) si existe."""
    filas = []
    for e in osm["elementos"]:
        t = e.get("tags", {})
        if t.get("power") not in ("substation", "plant"):
            continue
        lat = e.get("lat", e.get("center", {}).get("lat"))
        lon = e.get("lon", e.get("center", {}).get("lon"))
        if lat is None:
            continue
        mva_osm = None
        if t.get("power") == "plant":
            mva_osm = _numero(t.get("plant:output:electricity"))
        elif "MVA" in (t.get("rating") or ""):
            mva_osm = _numero(t.get("rating"))
        filas.append({
            "osm_id": f"{e['type'][0]}{e['id']}",
            "lat": round(lat, 7), "lon": round(lon, 7),
            "tipo": _clasificar(t),
            "voltaje": _limpiar(t.get("voltage")),
            "vmax_kv": _vmax(t) / 1000,
            "operador": _limpiar(t.get("operator")),
            "nombre": _limpiar(t.get("name") or t.get("name:es") or t.get("ref")),
            "mva_osm": mva_osm,
        })
    df = pd.DataFrame(filas).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df["x"], df["y"] = proyectar(df.lat.to_numpy(), df.lon.to_numpy())
    df["es_fuente"] = df.tipo != "distribucion"
    return df


def aristas_candidatas(nodos, k=5, k_fuente=40):
    """Grafo candidato:
      - cada nodo con sus k vecinos mas cercanos (KD-tree, O(n log n));
      - cada fuente (SET/central) con sus k_fuente subestaciones mas cercanas
        (salidas posibles de alimentadores);
      - si quedan componentes aisladas, se unen con la arista mas corta (UFDS).
    Devuelve DataFrame (u, v, dist_m)."""
    xy = nodos[["x", "y"]].to_numpy()
    lat, lon = nodos.lat.to_numpy(), nodos.lon.to_numpy()
    arbol = cKDTree(xy)
    pares = set()
    _, idx = arbol.query(xy, k=min(k + 1, len(nodos)))
    for i in range(len(nodos)):
        for j in idx[i][1:]:
            pares.add((min(i, int(j)), max(i, int(j))))
    fuentes = np.flatnonzero(nodos.es_fuente.to_numpy())
    if len(fuentes):
        _, idx_f = arbol.query(xy[fuentes], k=min(k_fuente + 1, len(nodos)))
        for f, fila in zip(fuentes, idx_f):
            for j in fila[1:]:
                pares.add((min(int(f), int(j)), max(int(f), int(j))))

    # conectar componentes aisladas (UFDS)
    uf = UFDS(len(nodos))
    for u, v in pares:
        uf.union(u, v)
    raices = {}
    for i in range(len(nodos)):
        raices.setdefault(uf.find(i), []).append(i)
    if len(raices) > 1:
        grupos = sorted(raices.values(), key=len, reverse=True)
        principal = list(grupos[0])
        for g in grupos[1:]:
            sub = cKDTree(xy[principal])
            d, j = sub.query(xy[g])
            m = int(np.argmin(d))
            a, b = g[m], principal[int(j[m])]
            pares.add((min(a, b), max(a, b)))
            principal.extend(g)

    u = np.array([p[0] for p in pares])
    v = np.array([p[1] for p in pares])
    d = haversine_np(lat[u], lon[u], lat[v], lon[v])
    return pd.DataFrame({"u": u, "v": v, "dist_m": np.round(d, 2)}).sort_values(["u", "v"]).reset_index(drop=True)


def lineas_alta_tension(osm, bbox=BBOX_LIMA):
    """Tramos de las lineas reales de AT (60/220/500 kV) para dibujar como referencia."""
    tramos = []
    for w in osm.get("lineas", []):
        volt = w.get("tags", {}).get("voltage", "s/d")
        prev = None
        for p in w.get("geometry", []):
            dentro = bbox[0] <= p["lat"] <= bbox[1] and bbox[2] <= p["lon"] <= bbox[3]
            if dentro and prev is not None:
                tramos.append((prev[0], prev[1], p["lat"], p["lon"], volt))
            prev = (p["lat"], p["lon"]) if dentro else None
    return pd.DataFrame(tramos, columns=["lat1", "lon1", "lat2", "lon2", "voltaje"])
