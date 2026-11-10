"""Modelo electrico de la red de media tension (MT).

- Catalogo de conductores (cables subterraneos tipo N2XSY 8.7/15 kV): resistencia R,
  reactancia X (ohm/km), corriente admisible I (A) y costo instalado (USD/km).
  Valores referenciales de catalogo.
- Capacidad termica de una linea trifasica:  P_max = sqrt(3) * V * I * fp.
- Flujo de potencia DC (aproximacion lineal):  B * theta = P,
      flujo_uv = (theta_u - theta_v) / x_uv.
  En una red radial coincide exactamente con la suma de demandas aguas abajo.
- Caida de tension por tramo:  dV% = 100 * (P*R + Q*X) * L / V^2,  Q = P*tan(acos fp),
  acumulada desde la fuente siguiendo el sentido del flujo (orden topologico).
  Limite admisible: 5 % (tolerancia de tension en MT de la NTCSE, D.S. 020-97-EM).
"""
import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .algoritmos.recorridos import orden_topologico
from .algoritmos.ufds import UFDS


@dataclass(frozen=True)
class Conductor:
    nombre: str
    r: float          # ohm/km
    x: float          # ohm/km
    i_max: float      # A
    costo_km: float   # USD/km instalado


CONDUCTORES = [
    Conductor("N2XSY 3x1x70 mm²", 0.342, 0.19, 245, 95_000),
    Conductor("N2XSY 3x1x120 mm²", 0.196, 0.17, 335, 120_000),
    Conductor("N2XSY 3x1x240 mm²", 0.098, 0.15, 490, 165_000),
    Conductor("2 ternas 3x1x240 mm²", 0.049, 0.075, 980, 300_000),
]


@dataclass(frozen=True)
class ParamElectricos:
    v_kv: float = 10.0            # tension de distribucion primaria de Lima
    fp: float = 0.9               # factor de potencia
    dv_max: float = 5.0           # % caida de tension admisible
    cap_alim_mw: float = 6.0      # carga maxima por alimentador (MW)
    mva_set_60: float = 50.0      # capacidad SET 60/10 kV (MVA)
    mva_set_220: float = 120.0    # capacidad SET 220/60/10 kV (MVA)
    mva_central: float = 100.0    # central sin dato en OSM (MVA)


def capacidad_mw(cond, p: ParamElectricos):
    return math.sqrt(3) * p.v_kv * cond.i_max * p.fp / 1000.0


def caida_pct(flujo_mw, long_km, cond, p: ParamElectricos):
    q = np.abs(flujo_mw) * math.tan(math.acos(p.fp))
    return 100.0 * (np.abs(flujo_mw) * cond.r + q * cond.x) * long_km / (p.v_kv ** 2)


def capacidad_fuentes(nodos, p: ParamElectricos):
    """Potencia disponible (MW) de cada fuente. Usa el dato de OSM si existe;
    si no, un valor tipico segun el nivel de tension de la SET."""
    cap = np.zeros(len(nodos))
    for r in nodos.itertuples():
        if not r.es_fuente:
            continue
        if r.mva_osm and not math.isnan(r.mva_osm):
            mva = r.mva_osm
        elif r.tipo == "generacion":
            mva = p.mva_central
        elif r.vmax_kv >= 220:
            mva = p.mva_set_220
        else:
            mva = p.mva_set_60
        cap[r.id] = mva * p.fp
    return cap


def flujo_dc(n, u, v, x, inyeccion):
    """Resuelve el flujo de potencia DC.
    u, v, x: arreglos de las lineas cerradas (x = reactancia en ohm).
    inyeccion: P neta por nodo (MW, generacion - demanda); debe sumar 0 por isla.
    Se fija theta = 0 en un nodo 'slack' por cada isla. Devuelve (theta, flujo)."""
    uf = UFDS(n)
    for a, b in zip(u, v):
        uf.union(int(a), int(b))
    raiz = np.array([uf.find(i) for i in range(n)])
    slack = {}
    for i in np.argsort(-inyeccion):              # slack = nodo de mayor generacion
        slack.setdefault(raiz[i], int(i))
    es_slack = np.zeros(n, bool)
    es_slack[list(slack.values())] = True

    b = 1.0 / np.maximum(x, 1e-6)
    filas = np.concatenate([u, v, u, v])
    cols = np.concatenate([u, v, v, u])
    vals = np.concatenate([b, b, -b, -b])
    B = sp.csr_matrix((vals, (filas, cols)), shape=(n, n))
    libres = np.flatnonzero(~es_slack)
    theta = np.zeros(n)
    if len(libres):
        Bred = B[libres][:, libres].tocsc()
        theta[libres] = spsolve(Bred, inyeccion[libres])
    flujo = (theta[u] - theta[v]) * b
    return theta, flujo


def evaluar_red(n, lineas, demanda, es_fuente, cap_fuente, p: ParamElectricos):
    """Evalua una configuracion: flujo DC, carga de lineas y caida de tension.

    lineas: DataFrame con columnas u, v, long_km, cond (indice en CONDUCTORES),
            estado ('cerrado' / 'abierto').
    Devuelve (lineas_resultado, nodos_resultado dict, metricas dict).
    """
    L = lineas.copy()
    cerr = (L.estado == "cerrado").to_numpy()
    u = L.u.to_numpy()[cerr]
    v = L.v.to_numpy()[cerr]
    lk = L.long_km.to_numpy()[cerr]
    cond = L.cond.to_numpy()[cerr]
    xs = np.array([CONDUCTORES[c].x for c in cond]) * np.maximum(lk, 1e-3)

    # islas y suministro
    uf = UFDS(n)
    for a, b in zip(u, v):
        uf.union(int(a), int(b))
    raiz = np.array([uf.find(i) for i in range(n)])
    cap_isla, dem_isla = {}, {}
    for i in range(n):
        if es_fuente[i]:
            cap_isla[raiz[i]] = cap_isla.get(raiz[i], 0.0) + cap_fuente[i]
        dem_isla[raiz[i]] = dem_isla.get(raiz[i], 0.0) + demanda[i]
    suministrado = np.array([cap_isla.get(raiz[i], 0.0) > 0 for i in range(n)])
    iny = np.where(suministrado, -demanda, 0.0)
    for i in np.flatnonzero(es_fuente):           # despacho proporcional a la capacidad
        r = raiz[i]
        iny[i] += dem_isla[r] * cap_fuente[i] / cap_isla[r]

    _, flujo = flujo_dc(n, u, v, xs, iny)

    cap_lin = np.array([capacidad_mw(CONDUCTORES[c], p) for c in cond])
    carga = 100 * np.abs(flujo) / cap_lin
    dv_lin = np.array([caida_pct(f, l, CONDUCTORES[c], p) for f, l, c in zip(flujo, lk, cond)])

    # caida de tension acumulada: arcos en el sentido del flujo (DAG) + orden topologico
    arcos = []
    for k in range(len(u)):
        if flujo[k] > 1e-9:
            arcos.append((int(u[k]), int(v[k]), dv_lin[k]))
        elif flujo[k] < -1e-9:
            arcos.append((int(v[k]), int(u[k]), dv_lin[k]))
    orden, entrantes = orden_topologico(n, arcos)
    dv_nodo = np.zeros(n)
    for w in orden:
        for a, d in entrantes[w]:
            dv_nodo[w] = max(dv_nodo[w], dv_nodo[a] + d)
    dv_nodo[~suministrado] = np.nan

    for col in ("flujo_mw", "cap_mw", "carga_pct", "dv_pct"):
        L[col] = np.nan
    L.loc[cerr, "flujo_mw"] = np.abs(flujo)
    L.loc[cerr, "cap_mw"] = cap_lin
    L.loc[cerr, "carga_pct"] = carga
    L.loc[cerr, "dv_pct"] = dv_lin
    L["costo_usd"] = [CONDUCTORES[c].costo_km * l for c, l in zip(L.cond, L.long_km)]

    dem_total = float(demanda.sum())
    dem_sin = float(demanda[~suministrado].sum())
    gen_fuente = np.where(es_fuente, np.maximum(iny + demanda, 0), 0)
    sobre_fuente = int(np.sum(gen_fuente > cap_fuente + 1e-6))
    validos = ~np.isnan(dv_nodo)
    metricas = {
        "km_total": float(L.long_km.sum()),
        "costo_musd": float(L.costo_usd.sum()) / 1e6,
        "lineas": int(len(L)),
        "lineas_cerradas": int(cerr.sum()),
        "demanda_mw": dem_total,
        "demanda_sin_suministro_mw": dem_sin,
        "lineas_sobrecargadas": int(np.sum(carga > 100 + 1e-6)),
        "carga_max_pct": float(carga.max()) if len(carga) else 0.0,
        "dv_max_pct": float(np.nanmax(dv_nodo)) if validos.any() else 0.0,
        "nodos_dv_fuera": int(np.sum(dv_nodo[validos] > p.dv_max + 1e-9)),
        "fuentes_sobrecargadas": sobre_fuente,
    }
    nodos_res = {"dv_pct": dv_nodo, "suministrado": suministrado}
    return L, nodos_res, metricas
