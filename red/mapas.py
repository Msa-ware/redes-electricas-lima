"""Figuras (plotly) para la interfaz: mapas de la red y graficos."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETA = {"teal": "#0E7C7B", "azul": "#2F6690", "celeste": "#3A86C8", "verde": "#2A9D8F",
          "amarillo": "#E9C46A", "naranja": "#F4A261", "coral": "#E4572E", "morado": "#8E6CC9",
          "oscuro": "#13303F"}
COLOR_TIPO = {"distribucion": PALETA["celeste"], "transmision": PALETA["coral"], "generacion": PALETA["verde"]}
COLOR_ZONA = {"comercial": PALETA["amarillo"], "residencial_alto": PALETA["morado"],
              "residencial_medio": PALETA["celeste"], "popular": PALETA["verde"], "industrial": PALETA["coral"]}
COLOR_AT = {"500": "#6D4C9F", "220": PALETA["naranja"], "60": "#C9A227"}
COLOR_CFG = {"A": PALETA["coral"], "B": PALETA["celeste"], "C": PALETA["verde"]}
CENTRO_LIMA = {"lat": -12.06, "lon": -77.0}


def _segmentos(u, v, lat, lon):
    la = np.empty(len(u) * 3)
    lo = np.empty(len(u) * 3)
    la[0::3], la[1::3], la[2::3] = lat[u], lat[v], np.nan
    lo[0::3], lo[1::3], lo[2::3] = lon[u], lon[v], np.nan
    return la, lo


def traza_lineas(u, v, lat, lon, color, nombre, ancho=1.5, opacidad=0.9):
    la, lo = _segmentos(np.asarray(u, int), np.asarray(v, int), lat, lon)
    return go.Scattermap(lat=la, lon=lo, mode="lines", line=dict(color=color, width=ancho),
                         opacity=opacidad, name=nombre, hoverinfo="skip")


def figura_base(alto=620, centro=None, zoom=9.6):
    fig = go.Figure()
    fig.update_layout(map=dict(style="carto-positron", center=centro or CENTRO_LIMA, zoom=zoom),
                      height=alto, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=11)))
    return fig


def centro_zoom(lat, lon):
    lat, lon = np.asarray(lat), np.asarray(lon)
    if len(lat) == 0:
        return CENTRO_LIMA, 9.6
    extension = max(lat.max() - lat.min(), (lon.max() - lon.min()) * 0.98, 1e-3)
    zoom = float(np.clip(np.log2(360 / extension) - 1.3, 8.5, 16))
    return {"lat": float(lat.mean()), "lon": float(lon.mean())}, zoom


def mapa_dataset(nodos, at, colorear="tipo", demanda=None):
    fig = figura_base()
    def _nivel(volt):
        kv = str(volt).split(";")[0].strip()
        return str(int(kv) // 1000) if kv.isdigit() else "s/d"
    etiqueta = at.voltaje.map(_nivel)
    for nivel, g in at.assign(nivel=etiqueta).groupby("nivel"):
        la = np.ravel(np.column_stack([g.lat1, g.lat2, np.full(len(g), np.nan)]))
        lo = np.ravel(np.column_stack([g.lon1, g.lon2, np.full(len(g), np.nan)]))
        fig.add_trace(go.Scattermap(lat=la, lon=lo, mode="lines", hoverinfo="skip",
                                    line=dict(color=COLOR_AT.get(nivel, "#a0aec0"), width=2),
                                    name=f"Línea AT {nivel} kV", legendgroup="at"))
    if colorear == "demanda" and demanda is not None:
        d = nodos[~nodos.es_fuente]
        fig.add_trace(go.Scattermap(
            lat=d.lat, lon=d.lon, mode="markers", name="SED (demanda)",
            marker=dict(size=5, color=demanda[d.id] * 1000, colorscale=[[0, "#FFF3C4"], [0.5, PALETA["naranja"]], [1, PALETA["coral"]]], cmin=50, cmax=450,
                        colorbar=dict(title="kW", x=1.0, len=0.6)),
            text=[f"{r.nombre or r.osm_id}<br>{r.distrito} ({r.zona})<br>{demanda[r.id]*1000:.0f} kW"
                  for r in d.itertuples()], hoverinfo="text"))
    else:
        grupos = (nodos.groupby("zona") if colorear == "zona" else nodos.groupby("tipo"))
        for clave, g in grupos:
            if colorear == "zona":
                g = g[~g.es_fuente]
                color, nombre, tam = COLOR_ZONA[clave], clave.replace("_", " ").capitalize(), 5
            else:
                color, nombre = COLOR_TIPO[clave], {"distribucion": "Subestación de distribución",
                                                   "transmision": "SET (transmisión)",
                                                   "generacion": "Central de generación"}[clave]
                tam = 5 if clave == "distribucion" else 13
            fig.add_trace(go.Scattermap(
                lat=g.lat, lon=g.lon, mode="markers", name=f"{nombre} ({len(g)})",
                marker=dict(size=tam, color=color),
                text=[f"{r.nombre or r.osm_id}<br>{r.distrito}" for r in g.itertuples()], hoverinfo="text"))
    if colorear == "zona":
        f = nodos[nodos.es_fuente]
        fig.add_trace(go.Scattermap(lat=f.lat, lon=f.lon, mode="markers", name="Fuentes (SET/centrales)",
                                    marker=dict(size=12, color=PALETA["oscuro"]), text=f.nombre, hoverinfo="text"))
    return fig


BINS_CARGA = [(0, 50, PALETA["verde"], "Carga < 50 %"), (50, 80, PALETA["amarillo"], "Carga 50–80 %"),
              (80, 100.0001, PALETA["naranja"], "Carga 80–100 %"), (100.0001, np.inf, PALETA["coral"], "Sobrecarga > 100 %")]


def mapa_red(ctx, Lr, nodos_res, dv_max, resaltar=None, centro=None, zoom=9.6, alto=620):
    """Lineas coloreadas por % de carga, respaldos en morado, nodos por caida de tension."""
    lat, lon = ctx["lat"], ctx["lon"]
    fig = figura_base(alto, centro, zoom)
    cerr = Lr[Lr.estado == "cerrado"]
    for a, b, color, nombre in BINS_CARGA:
        g = cerr[(cerr.carga_pct >= a) & (cerr.carga_pct < b)]
        if len(g):
            fig.add_trace(traza_lineas(g.u, g.v, lat, lon, color, f"{nombre} ({len(g)})", 2.2))
    resp = Lr[(Lr.tipo == "respaldo") & (Lr.estado == "abierto")]
    if len(resp):
        fig.add_trace(traza_lineas(resp.u, resp.v, lat, lon, PALETA["morado"], f"Respaldo abierto ({len(resp)})", 1.2, 0.7))
    dv = nodos_res["dv_pct"]
    d = np.flatnonzero(~ctx["es_fuente"])
    ok = d[~np.isnan(dv[d])]
    fig.add_trace(go.Scattermap(
        lat=lat[ok], lon=lon[ok], mode="markers", name="SED (ΔV %)",
        marker=dict(size=5, color=np.clip(dv[ok], 0, 2 * dv_max), colorscale=[[0, PALETA["azul"]], [0.45, "#8CC7E8"],
                    [0.5, PALETA["amarillo"]], [0.75, PALETA["naranja"]], [1, "#B23A1F"]], cmin=0, cmax=2 * dv_max,
                    colorbar=dict(title="ΔV %", x=1.0, len=0.6)),
        text=[f"Nodo {i} · {ctx['nodos'].distrito.iat[i]}<br>Demanda {ctx['demanda'][i]*1000:.0f} kW<br>ΔV {dv[i]:.2f} %"
              for i in ok], hoverinfo="text"))
    sin = d[np.isnan(dv[d])]
    if len(sin):
        fig.add_trace(go.Scattermap(lat=lat[sin], lon=lon[sin], mode="markers", name=f"Sin suministro ({len(sin)})",
                                    marker=dict(size=9, color=PALETA["coral"]), hoverinfo="skip"))
    f = np.flatnonzero(ctx["es_fuente"])
    fig.add_trace(go.Scattermap(lat=lat[f], lon=lon[f], mode="markers", name="SET / central",
                                marker=dict(size=13, color=PALETA["oscuro"]),
                                text=[ctx["nodos"].nombre.iat[i] or f"Fuente {i}" for i in f], hoverinfo="text"))
    if resaltar:
        for item in resaltar:
            fig.add_trace(item)
    return fig


def curva_carga(cs, hora):
    fig = go.Figure()
    for col, color in zip(["Comercial", "Residencial alto", "Residencial medio", "Residencial popular", "Industrial"],
                          [COLOR_ZONA[z] for z in ("comercial", "residencial_alto", "residencial_medio", "popular", "industrial")]):
        fig.add_trace(go.Scatter(x=cs.hora, y=cs[col], stackgroup="z", name=col, line=dict(width=0.5, color=color)))
    fig.add_vline(x=hora, line_dash="dash", line_color=PALETA["oscuro"],
                  annotation_text=f"Hora analizada {hora}:00", annotation_position="top left")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Hora del día",
                      yaxis_title="Demanda (MW)", legend=dict(orientation="h", y=-0.25))
    return fig
