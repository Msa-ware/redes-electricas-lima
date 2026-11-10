"""
Aplicativo - Diseno de redes electricas de Lima Metropolitana
1ACC0184 Complejidad Algoritmica - UPC 2026-20 (Caso 8: Diseno de redes electricas)

Ejecutar:  streamlit run app.py
"""
import math
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import cKDTree

from red.algoritmos.fuerza_bruta import mst_backtracking, mst_fuerza_bruta
from red.algoritmos.mst import kruskal, prim
from red.contexto import cargar_base, construir_contexto
from red.datos import haversine_np, lineas_alta_tension
from red.demanda import DISTRITOS, KW_POR_SED, NOMBRE_ZONA, ZONAS, curva_sistema, demanda_mw
from red.diseno import disenar_A, disenar_B, disenar_C
from red.electrico import CONDUCTORES, ParamElectricos, capacidad_mw, evaluar_red
from red.fallos import analisis_n1, contar_puentes, flujo_maximo_red, simular_falla
from red import mapas

RUTA_JSON = Path(__file__).parent / "data" / "lima_red_electrica_osm.json"

st.set_page_config(page_title="Red Eléctrica Lima", layout="wide")

# Estilos: tarjetas para las métricas, cabecera con degradado y pestañas más visibles.
st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
.cabecera {background: linear-gradient(120deg, #0E7C7B 0%, #2F6690 100%); color: #fff;
           padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1.1rem;}
.cabecera h1 {color: #fff; font-size: 1.75rem; margin: 0; padding: 0;}
.cabecera p {color: #DDF3F1; margin: .35rem 0 0 0; font-size: .95rem;}
div[data-testid="stMetric"] {background: #F4F8F8; border: 1px solid #D9E7E6; border-left: 4px solid #0E7C7B;
           border-radius: 10px; padding: .6rem .75rem; height: 100%;}
div[data-testid="stMetricLabel"] p {color: #4A5B6A; font-size: .82rem;}
div[data-testid="stMetricLabel"] *, div[data-testid="stMetricDelta"] * {
           white-space: normal !important; overflow: visible !important; text-overflow: clip !important;}
div[data-testid="stMetricValue"] {color: #13303F; font-size: 1.45rem;}
button[data-baseweb="tab"] p {font-size: .98rem; font-weight: 600;}
section[data-testid="stSidebar"] {background: #F1F6F6;}
.guia {color: #4A5B6A; font-size: .92rem; margin: -.2rem 0 .8rem 0;}
</style>
""", unsafe_allow_html=True)


def guia(texto):
    """Línea corta que explica qué muestra cada pestaña."""
    st.markdown(f'<p class="guia">{texto}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------- caches
@st.cache_resource(show_spinner="Cargando dataset de OpenStreetMap…")
def base():
    osm, nodos = cargar_base(RUTA_JSON)
    return nodos, lineas_alta_tension(osm)


@st.cache_resource(show_spinner="Construyendo grafo candidato y demanda…")
def contexto(hora, kw, crec, p, k, distritos):
    nodos, _ = base()
    return construir_contexto(nodos, hora, dict(kw), crec, p, k, list(distritos) or None)


@st.cache_resource(show_spinner="Ejecutando algoritmos de diseño…")
def disenos(clave_ctx, alg_A, cond_A, factor_C, umbral_C):
    ctx = contexto(*clave_ctx)
    p = ctx["p"]
    out = {}
    for nombre, fn in [("A", lambda: disenar_A(ctx, alg_A, cond_A)),
                       ("B", lambda: disenar_B(ctx)),
                       ("C", lambda: disenar_C(ctx, factor_C, umbral_C))]:
        L, info = fn()
        t0 = time.perf_counter()
        Lr, nr, met = evaluar_red(ctx["n"], L, ctx["demanda"], ctx["es_fuente"], ctx["cap_fuente"], p)
        info["t_flujo_s"] = time.perf_counter() - t0
        out[nombre] = {"L": L, "Lr": Lr, "nodos": nr, "met": met, "info": info}
    return out


@st.cache_resource(show_spinner="Analizando fallas N-1, puentes y flujo máximo…")
def robustez(clave_ctx, alg_A, cond_A, factor_C, umbral_C):
    ctx = contexto(*clave_ctx)
    D = disenos(clave_ctx, alg_A, cond_A, factor_C, umbral_C)
    out = {}
    for nombre in ("A", "B", "C"):
        L = D[nombre]["L"]
        t0 = time.perf_counter()
        fm = flujo_maximo_red(ctx, L)
        t_fm = time.perf_counter() - t0
        r = {"puentes": contar_puentes(ctx, L), "flujo_max": fm, "t_flujo_max_s": t_fm}
        if nombre != "A":
            t0 = time.perf_counter()
            df, res = analisis_n1(ctx, L)
            r.update({"n1": df, "n1_res": res, "t_n1_s": time.perf_counter() - t0})
        out[nombre] = r
    return out


# ---------------------------------------------------------------- barra lateral
nodos_all, at = base()
st.sidebar.title("Parámetros")
st.sidebar.caption("Cada cambio se aplica al instante en todas las pestañas.")

with st.sidebar.expander("Ámbito de análisis", expanded=True):
    ambito = st.radio("Zona", ["Lima Metropolitana completa", "Elegir distritos"], label_visibility="collapsed")
    distritos = ()
    if ambito == "Elegir distritos":
        sel = st.multiselect("Distritos", sorted(d[0] for d in DISTRITOS),
                             default=["Miraflores", "San Isidro", "Surquillo", "San Borja", "Lince"])
        distritos = tuple(sorted(sel))
    k = st.slider("k vecinos del grafo candidato", 3, 10, 5,
                  help="Cada subestación se une con sus k vecinas más cercanas (posibles tendidos de cable).")

with st.sidebar.expander("Modelo de demanda por zonas", expanded=True):
    cs_def = curva_sistema(nodos_all)
    hora_punta = int(cs_def.loc[cs_def.total.idxmax(), "hora"])
    hora = st.slider("Hora del día", 0, 23, hora_punta,
                     help=f"Por defecto la hora de máxima demanda del sistema ({hora_punta}:00).")
    crec = st.slider("Crecimiento de la demanda (%)", 0, 60, 0, 5) / 100
    st.caption("Demanda máxima típica por subestación (kW):")
    kw = {}
    for z in ZONAS:
        kw[z] = float(st.number_input(NOMBRE_ZONA[z], 30, 1000, int(KW_POR_SED[z]), 10, key=f"kw_{z}"))

with st.sidebar.expander("Parámetros eléctricos", expanded=False):
    v_kv = st.selectbox("Tensión de distribución (kV)", [10.0, 22.9], 0)
    fp = st.slider("Factor de potencia", 0.80, 1.00, 0.90, 0.01)
    dv_max = st.slider("Caída de tensión admisible (%)", 2.0, 10.0, 5.0, 0.5,
                       help="NTCSE (D.S. 020-97-EM): tolerancia de ±5 % en media tensión.")
    cap_alim = st.slider("Capacidad por alimentador (MW)", 2.0, 15.0, 6.0, 0.5)
    mva60 = st.number_input("Capacidad SET 60/10 kV (MVA)", 10, 200, 50, 5)
    mva220 = st.number_input("Capacidad SET 220/60/10 kV (MVA)", 20, 500, 120, 10)

with st.sidebar.expander("Algoritmos de diseño", expanded=False):
    alg_A = st.radio("MST de la configuración A", ["Kruskal", "Prim"], horizontal=True)
    cond_A = st.selectbox("Conductor uniforme en A", range(len(CONDUCTORES)), 1,
                          format_func=lambda c: CONDUCTORES[c].nombre)
    factor_C = st.slider("Carga de planificación en C (% de la capacidad)", 40, 100, 50, 5,
                         help="Reserva para recibir la carga de un alimentador vecino en falla.") / 100
    umbral_C = st.slider("Proteger tramos que aíslan ≥ (MW)", 0.0, 2.0, 0.3, 0.1)

p = ParamElectricos(v_kv=v_kv, fp=fp, dv_max=dv_max, cap_alim_mw=cap_alim, mva_set_60=mva60, mva_set_220=mva220)
clave_ctx = (hora, tuple(sorted(kw.items())), crec, p, k, distritos)
ctx = contexto(*clave_ctx)

st.markdown("""
<div class="cabecera">
  <h1>Diseño de la red eléctrica de Lima Metropolitana</h1>
  <p>Complejidad Algorítmica (1ACC0184) · Caso 8: Diseño de redes eléctricas · Datos: OpenStreetMap (ODbL)</p>
</div>
""", unsafe_allow_html=True)

if ctx["n"] < 3 or not ctx["es_fuente"].any():
    st.error("La zona elegida no tiene suficientes nodos o no contiene ninguna SET/central. Agrega más distritos.")
    st.stop()

D = disenos(clave_ctx, alg_A, cond_A, factor_C, umbral_C)
NOMBRE_CFG = {"A": f"A · MST clásico ({alg_A})", "B": "B · Kruskal capacitado + conductores",
              "C": "C · Red robusta (B con reserva + respaldos)"}

tabs = st.tabs(["Dataset y demanda", "Diseño de la red", "Fallos (N-1)",
                "Comparación", "Validación", "Metodología"])

# ================================================================ 1. dataset
with tabs[0]:
    guia("Subestaciones de OpenStreetMap y demanda estimada por zona. Pasa el cursor sobre un punto para ver su detalle.")
    nodos = ctx["nodos"]
    c = st.columns(6)
    c[0].metric("Nodos", f"{ctx['n']:,}")
    c[1].metric("SED", f"{(nodos.tipo == 'distribucion').sum():,}")
    c[2].metric("SET", int((nodos.tipo == "transmision").sum()))
    c[3].metric("Centrales", int((nodos.tipo == "generacion").sum()))
    c[4].metric("Aristas cand.", f"{len(ctx['cand']):,}")
    c[5].metric(f"Demanda ({hora}:00)", f"{ctx['demanda'].sum():,.0f} MW")

    col1, col2 = st.columns([3, 2])
    with col1:
        colorear = st.radio("Colorear nodos por", ["tipo", "zona", "demanda"], horizontal=True,
                            format_func={"tipo": "Tipo de nodo", "zona": "Zona de demanda",
                                         "demanda": "Demanda (kW)"}.get)
        st.plotly_chart(mapas.mapa_dataset(nodos, at, colorear, ctx["demanda"]), use_container_width=True)
    with col2:
        st.subheader("Curva de carga del sistema")
        cs = curva_sistema(nodos, kw, crec)
        st.plotly_chart(mapas.curva_carga(cs, hora), use_container_width=True)
        st.subheader("Demanda por distrito")
        tabla = (nodos.assign(demanda_mw=ctx["demanda"]).groupby(["distrito", "zona"])
                 .agg(SED=("id", "size"), demanda_MW=("demanda_mw", "sum")).reset_index()
                 .sort_values("demanda_MW", ascending=False))
        tabla["zona"] = tabla.zona.map(NOMBRE_ZONA)
        st.dataframe(tabla.round(2), hide_index=True, use_container_width=True, height=260)

    with st.expander("Ver tabla de nodos y aristas candidatas"):
        a, b = st.columns(2)
        a.dataframe(nodos[["id", "osm_id", "tipo", "nombre", "distrito", "zona", "lat", "lon"]]
                    .assign(demanda_kW=np.round(ctx["demanda"] * 1000, 1)), hide_index=True, height=300)
        b.dataframe(ctx["cand"], hide_index=True, height=300)

# ================================================================ 2. diseno
with tabs[1]:
    guia("Elige una configuración. Las líneas se colorean según su carga y las subestaciones según su caída de tensión.")
    cfg = st.radio("Configuración", ["A", "B", "C"], format_func=NOMBRE_CFG.get, horizontal=True, index=1)
    R = D[cfg]
    m = R["met"]
    c = st.columns(6)
    c[0].metric("Cable", f"{m['km_total']:,.0f} km")
    c[1].metric("Costo (MUS$)", f"{m['costo_musd']:,.1f}")
    c[2].metric("Sobrecargas", m["lineas_sobrecargadas"],
                delta=f"carga máx. {m['carga_max_pct']:,.0f} %", delta_color="off")
    dvtxt = f"{m['dv_max_pct']:,.2f} %" if m["dv_max_pct"] < 100 else "> 100 %"
    c[3].metric("ΔV máxima", dvtxt)
    c[4].metric(f"SED con ΔV > {dv_max:g} %", f"{m['nodos_dv_fuera']:,}")
    c[5].metric("Cómputo", f"{(R['info']['t_algoritmo_s'] + R['info'].get('t_conductores_s', 0)) * 1000:,.0f} ms")

    ok = m["lineas_sobrecargadas"] == 0 and m["nodos_dv_fuera"] == 0 and m["demanda_sin_suministro_mw"] == 0
    if ok:
        st.success(f"La configuración {cfg} abastece {m['demanda_mw']:,.1f} MW en hora de máxima demanda "
                   f"sin sobrecargar líneas y con caída de tensión ≤ {dv_max:g} %.")
    else:
        st.error(f"La configuración {cfg} **no es operable**: {m['lineas_sobrecargadas']:,} líneas superan su "
                 f"capacidad térmica y {m['nodos_dv_fuera']:,} subestaciones tienen caída de tensión mayor a {dv_max:g} %. "
                 "Minimizar solo la longitud (MST) no garantiza soportar la demanda.")

    st.plotly_chart(mapas.mapa_red(ctx, R["Lr"], R["nodos"], dv_max), use_container_width=True)

    i = R["info"]
    if cfg == "A":
        st.info(f"Árbol de expansión mínima con {alg_A}: {len(R['L']):,} tramos, todos con "
                f"{CONDUCTORES[cond_A].nombre} (capacidad {capacidad_mw(CONDUCTORES[cond_A], p):.1f} MW). "
                "El flujo de potencia DC reparte la demanda por un único árbol que une todas las SET.")
    else:
        extra = f" · {i['enlaces_respaldo']:,} enlaces de respaldo (cobertura {i['cobertura_pct']:.0f} %)" if cfg == "C" else ""
        st.info(f"{i['alimentadores']:,} alimentadores radiales ({i['expresos']} expresos) · "
                f"{i['refuerzos']:,} tramos reforzados por caída de tensión{extra}.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uso de conductores")
        uso = R["Lr"].groupby(["tipo", "cond"]).agg(tramos=("u", "size"), km=("long_km", "sum"),
                                                costo_MUSD=("costo_usd", "sum")).reset_index()
        uso["cond"] = uso.cond.map(lambda c: CONDUCTORES[c].nombre)
        uso["costo_MUSD"] /= 1e6
        st.dataframe(uso.round(2), hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Distribución de la carga de las líneas")
        cc = R["Lr"][R["Lr"].estado == "cerrado"].carga_pct.clip(upper=300)
        fig = px.histogram(cc, nbins=60, labels={"value": "Carga (% de la capacidad, máx. 300)"},
                           color_discrete_sequence=[mapas.PALETA["teal"]])
        fig.add_vline(x=100, line_color=mapas.PALETA["coral"], line_dash="dash")
        fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tramos más cargados")
    top = R["Lr"][R["Lr"].estado == "cerrado"].sort_values("carga_pct", ascending=False).head(15).copy()
    top["conductor"] = top.cond.map(lambda c: CONDUCTORES[c].nombre)
    st.dataframe(top[["u", "v", "tipo", "conductor", "long_km", "flujo_mw", "cap_mw", "carga_pct", "dv_pct"]].round(3),
                 hide_index=True, use_container_width=True)
    st.download_button("Descargar líneas de la configuración (CSV)", R["Lr"].to_csv(index=False).encode(),
                       f"lineas_config_{cfg}.csv", "text/csv")

# ================================================================ 3. fallos
with tabs[2]:
    guia("Criterio N-1: se hace fallar cada tramo, uno a la vez, y se revisa si un enlace de respaldo recupera la zona aislada.")
    RB = robustez(clave_ctx, alg_A, cond_A, factor_C, umbral_C)
    cfg_f = st.radio("Configuración a evaluar", ["B", "C"], format_func=NOMBRE_CFG.get, horizontal=True, index=1)
    rr, R = RB[cfg_f], D[cfg_f]
    res = rr["n1_res"]
    c = st.columns(5)
    c[0].metric("Tramos (N-1)", f"{res['tramos']:,}")
    c[1].metric("Recuperables", f"{res['pct_restaurables']:.1f} %")
    c[2].metric("No servida (kW/falla)", f"{res['ens_promedio_mw'] * 1000:,.0f}")
    c[3].metric("Peor falla", f"{res['peor_falla_mw']:.2f} MW")
    c[4].metric("Puentes (Tarjan)", f"{rr['puentes']:,}")
    st.caption("Para cada tramo se calcula la demanda que queda aislada (subárbol del DFS) y si algún enlace de "
               "respaldo puede recuperarla sin superar la capacidad de emergencia (120 %) del alimentador y la SET receptores.")

    n1 = rr["n1"]
    st.subheader("Simular la falla de un tramo")
    modo = st.radio("Elegir tramo", ["El que aísla más demanda", "Aleatorio", "Por índice"], horizontal=True)
    if modo == "El que aísla más demanda":
        idx_list = n1.sort_values("demanda_aislada_mw", ascending=False).linea.tolist()
        pos = st.slider("Ranking", 1, min(50, len(idx_list)), 1)
        linea = int(idx_list[pos - 1])
    elif modo == "Aleatorio":
        if st.button("Elegir otro tramo") or "linea_rand" not in st.session_state:
            st.session_state.linea_rand = int(np.random.choice(n1.linea))
        linea = st.session_state.linea_rand
        if linea not in set(n1.linea):
            linea = int(n1.linea.iloc[0])
    else:
        linea = int(st.number_input("Índice de la línea", 0, int(n1.linea.max()), int(n1.linea.iloc[0])))
        if linea not in set(n1.linea):
            st.warning("Ese índice no es un tramo cerrado de esta configuración; se usa el primero.")
            linea = int(n1.linea.iloc[0])

    sim = simular_falla(ctx, R["L"], linea, n1)
    fila = R["L"].loc[linea]
    u0, v0 = int(fila.u), int(fila.v)
    lat, lon = ctx["lat"], ctx["lon"]
    extra = [go.Scattermap(lat=[lat[u0], lat[v0]], lon=[lon[u0], lon[v0]], mode="lines+markers",
                           line=dict(color="#13303F", width=6), marker=dict(size=10, color="#13303F"),
                           name="Tramo en falla")]
    if len(sim["aislados"]):
        noc = np.setdiff1d(sim["aislados"], sim["restaurados"])
        if len(sim["restaurados"]):
            extra.append(go.Scattermap(lat=lat[sim["restaurados"]], lon=lon[sim["restaurados"]], mode="markers",
                                       marker=dict(size=11, color=mapas.PALETA["verde"]), name="Recuperadas por respaldo"))
        if len(noc):
            extra.append(go.Scattermap(lat=lat[noc], lon=lon[noc], mode="markers",
                                       marker=dict(size=11, color=mapas.PALETA["coral"]), name="Sin suministro"))
    if sim["enlace"] is not None:
        e = R["L"].loc[sim["enlace"]]
        extra.append(go.Scattermap(lat=[lat[int(e.u)], lat[int(e.v)]], lon=[lon[int(e.u)], lon[int(e.v)]],
                                   mode="lines", line=dict(color=mapas.PALETA["morado"], width=7), name="Respaldo cerrado"))
    zona_nodos = np.concatenate([[u0, v0], sim["aislados"]]).astype(int)
    centro, zoom = mapas.centro_zoom(lat[zona_nodos], lon[zona_nodos])
    zoom = min(zoom, 15.0)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(mapas.mapa_red(ctx, sim["L"], sim["nodos"], dv_max, extra, centro, zoom, 560),
                        use_container_width=True)
    with c2:
        dem_ais = ctx["demanda"][sim["aislados"]].sum()
        dem_rec = ctx["demanda"][sim["restaurados"]].sum()
        st.markdown(f"**Tramo {linea}** (nodos {u0} – {v0}, {fila.long_km * 1000:.0f} m, {fila.tipo})")
        st.metric("Subestaciones aisladas por la falla", f"{len(sim['aislados'])}", f"{dem_ais:.2f} MW", delta_color="off")
        if sim["enlace"] is not None:
            st.success(f"Se cierra el enlace de respaldo {sim['enlace']} → se recuperan "
                       f"{len(sim['restaurados'])} SED ({dem_rec:.2f} MW).")
        elif len(sim["aislados"]):
            st.error("No hay enlace de respaldo con capacidad suficiente: la demanda queda sin servicio "
                     "hasta reparar el tramo.")
        mm = sim["met"]
        st.markdown("**Estado de la red después de la maniobra**")
        st.write(pd.DataFrame({
            "Indicador": ["Demanda sin suministro (MW)", "Líneas sobrecargadas", "Carga máxima (%)",
                          "Caída de tensión máx. (%)", f"SED con ΔV > {dv_max:g} %"],
            "Valor": [round(mm["demanda_sin_suministro_mw"], 2), mm["lineas_sobrecargadas"],
                      round(mm["carga_max_pct"], 1), round(mm["dv_max_pct"], 2), mm["nodos_dv_fuera"]]}))
        st.caption("En emergencia se tolera temporalmente hasta 120 % de carga; el mapa muestra los "
                   "nuevos flujos (flujo DC) y tensiones tras la reconfiguración.")

    st.subheader("Tramos críticos (fallas que aíslan más demanda)")
    crit = n1.sort_values("demanda_aislada_mw", ascending=False).head(20)
    st.dataframe(crit.round(3), hide_index=True, use_container_width=True)

# ================================================================ 4. comparacion
with tabs[3]:
    guia("Las tres configuraciones lado a lado: costo, cumplimiento de las restricciones físicas y robustez ante fallas.")
    RB = robustez(clave_ctx, alg_A, cond_A, factor_C, umbral_C)
    filas = []
    for cfg in ("A", "B", "C"):
        m, i, r = D[cfg]["met"], D[cfg]["info"], RB[cfg]
        filas.append({
            "Configuración": NOMBRE_CFG[cfg],
            "Longitud (km)": m["km_total"], "Costo (MUS$)": m["costo_musd"],
            "Líneas sobrecargadas": m["lineas_sobrecargadas"], "Carga máx. (%)": m["carga_max_pct"],
            "ΔV máx. (%)": m["dv_max_pct"], f"SED con ΔV>{dv_max:g}%": m["nodos_dv_fuera"],
            "Demanda abastecible – flujo máx. (%)": 100 * r["flujo_max"] / max(m["demanda_mw"], 1e-9),
            "Puentes": r["puentes"],
            "Fallas N-1 recuperables (%)": r["n1_res"]["pct_restaurables"] if cfg != "A" else 0.0,
            "Tiempo algoritmo (ms)": 1000 * (i["t_algoritmo_s"] + i.get("t_conductores_s", 0)),
        })
    comp = pd.DataFrame(filas)
    st.dataframe(comp.round(2), hide_index=True, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    corto = comp.assign(cfg=["A", "B", "C"])
    for col, y, titulo in [(c1, "Costo (MUS$)", "Costo de inversión"),
                           (c2, "Líneas sobrecargadas", "Líneas sobrecargadas"),
                           (c3, "Fallas N-1 recuperables (%)", "Robustez (N-1)")]:
        fig = px.bar(corto, x="cfg", y=y, color="cfg", text_auto=".3s",
                     color_discrete_map=mapas.COLOR_CFG)
        fig.update_layout(title=titulo, height=300, showlegend=False, xaxis_title="", margin=dict(l=10, r=10, t=40, b=10))
        col.plotly_chart(fig, use_container_width=True)

    a, b = D["A"]["met"], D["B"]["met"]
    st.markdown(f"""
**Interpretación**
- **A (MST, enfoque del TB1)** es la red más corta ({a['km_total']:,.0f} km), pero al concentrar la demanda de toda la
  ciudad en un solo árbol, el flujo DC sobrecarga **{a['lineas_sobrecargadas']:,}** tramos y la caída de tensión
  supera el {dv_max:g} % en **{a['nodos_dv_fuera']:,}** subestaciones: **no es operable**.
- **B** agrega restricciones de potencia: cuesta un {100 * (b['costo_musd'] / a['costo_musd'] - 1):.0f} % más, pero
  cumple capacidad térmica y caída de tensión en hora punta. Sin embargo, es radial: cada tramo es un puente y cualquier falla deja clientes sin servicio.
- **C** planifica los alimentadores con reserva y agrega enlaces de respaldo normalmente abiertos: aumenta el costo,
  pero recupera la mayoría de las fallas simples (criterio N-1).
""")

# ================================================================ 5. validacion
with tabs[4]:
    guia("Pruebas que comprueban que los algoritmos dan resultados correctos.")
    st.subheader("Fuerza bruta y Backtracking vs Kruskal / Prim en un subgrafo pequeño")
    st.caption("Se toman las n subestaciones más cercanas a una SET y se forma el grafo completo entre ellas "
               "(todas las conexiones posibles). Se verifica que Kruskal/Prim encuentran el árbol óptimo.")
    nodos = ctx["nodos"]
    fuentes = nodos[nodos.es_fuente]
    c1, c2 = st.columns(2)
    centro_id = c1.selectbox("SET de referencia", fuentes.id.tolist(),
                             format_func=lambda i: f"{i} · {nodos.nombre.iat[i] or 'sin nombre'} ({nodos.distrito.iat[i]})")
    nv = c2.slider("Número de nodos n", 4, 10, 7)
    _, vecinos = cKDTree(nodos[["x", "y"]].to_numpy()).query(nodos.loc[centro_id, ["x", "y"]].to_numpy(dtype=float), k=nv)
    vecinos = list(map(int, vecinos))
    la, lo = ctx["lat"][vecinos], ctx["lon"][vecinos]
    ar = []
    for a_ in range(nv):
        for b_ in range(a_ + 1, nv):
            ar.append((float(haversine_np(la[a_], lo[a_], la[b_], lo[b_])), a_, b_))
    E = len(ar)
    filas = []
    t0 = time.perf_counter(); ik, ck = kruskal(nv, ar); filas.append(("Kruskal + UFDS", ck, len(ar), time.perf_counter() - t0, "O(E log E)"))
    t0 = time.perf_counter(); ip, cp = prim(nv, ar); filas.append(("Prim (heap)", cp, len(ar), time.perf_counter() - t0, "O(E log V)"))
    t0 = time.perf_counter(); cb, ib, nb = mst_backtracking(nv, ar); filas.append(("Backtracking", cb, nb, time.perf_counter() - t0, "Exponencial (con poda)"))
    combinaciones = comb(E, nv - 1)
    if combinaciones <= 3_000_000:
        t0 = time.perf_counter(); cf, if_, nf = mst_fuerza_bruta(nv, ar); filas.append(("Fuerza bruta", cf, nf, time.perf_counter() - t0, f"O(C(E,n-1)·n)"))
    else:
        st.warning(f"Fuerza bruta omitida: C({E},{nv - 1}) = {combinaciones:,} combinaciones.")
    val = pd.DataFrame(filas, columns=["Algoritmo", "Costo del árbol (m)", "Combinaciones / nodos explorados",
                                       "Tiempo (s)", "Complejidad"])
    st.dataframe(val.round({"Costo del árbol (m)": 2, "Tiempo (s)": 5}), hide_index=True, use_container_width=True)
    if np.allclose(val["Costo del árbol (m)"], ck):
        st.success("Todos los métodos obtienen el mismo costo: Kruskal y Prim encuentran el óptimo.")
    else:
        st.error("Los costos difieren.")
    fig = mapas.figura_base(420, *mapas.centro_zoom(la, lo))
    fig.add_trace(mapas.traza_lineas([a_ for _, a_, _ in ar], [b_ for _, _, b_ in ar], la, lo, "#C5D3D6", "Conexiones posibles", 1))
    fig.add_trace(mapas.traza_lineas([ar[i][1] for i in ik], [ar[i][2] for i in ik], la, lo, mapas.PALETA["teal"], "MST", 4))
    fig.add_trace(go.Scattermap(lat=la, lon=lo, mode="markers+text", text=[str(i) for i in range(nv)],
                                marker=dict(size=12, color="#13303F"), name="Nodos", textposition="top right"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Verificación del flujo de potencia DC")
    Lr = D["B"]["Lr"]
    st.caption("En una red radial, el flujo DC de cada tramo debe ser igual a la suma de demandas aguas abajo "
               "(conservación de la potencia). Además, el flujo máximo (Dinic) de la red B debe igualar la demanda total.")
    from red.algoritmos.recorridos import arbol_radial
    arb = arbol_radial(ctx["n"], list(zip(D["B"]["L"].u, D["B"]["L"].v)), list(np.flatnonzero(ctx["es_fuente"])))
    sub = ctx["demanda"].copy()
    for w in reversed(arb["preorden"]):
        if arb["padre"][w] != -1:
            sub[arb["padre"][w]] += sub[w]
    hijos = [w for w in range(ctx["n"]) if arb["arista_padre"][w] != -1]
    err = max(abs(Lr.flujo_mw.iat[arb["arista_padre"][w]] - sub[w]) for w in hijos)
    RBv = robustez(clave_ctx, alg_A, cond_A, factor_C, umbral_C)
    c = st.columns(3)
    c[0].metric("Error máx. flujo DC vs suma de demandas", f"{err:.2e} MW")
    c[1].metric("Flujo máximo (Dinic) en B", f"{RBv['B']['flujo_max']:,.2f} MW")
    c[2].metric("Demanda total", f"{ctx['demanda'].sum():,.2f} MW")

# ================================================================ 6. metodologia
with tabs[5]:
    st.markdown(f"""
### Problema
Diseñar la red de media tensión que conecta las **{(ctx['nodos'].tipo == 'distribucion').sum():,} subestaciones de
distribución (SED)** con las **SET y centrales** de Lima, minimizando el costo del cableado, **soportando la demanda
en hora punta** y siendo **robusta ante fallos**.

### Modelo
| Elemento | Representación |
|---|---|
| Nodo | SED (consumo), SET o central (fuente) – OpenStreetMap |
| Arista candidata | k vecinos más cercanos (KD-tree) + salidas de cada SET; peso = distancia Haversine |
| Demanda | kW típicos por SED según el **tipo de zona del distrito** × curva de carga horaria × diversidad |
| Capacidad de línea | P = √3 · V · I<sub>máx</sub> · fp del conductor (catálogo N2XSY) |
| Flujo de potencia | **Flujo DC**: B·θ = P, f<sub>uv</sub> = (θ<sub>u</sub> − θ<sub>v</sub>) / x<sub>uv</sub> |
| Caída de tensión | ΔV% = 100·(P·R + Q·X)·L / V², acumulada en orden topológico desde la fuente; límite {dv_max:g} % (NTCSE) |

### Técnicas del curso utilizadas
| Técnica | Dónde se usa | Complejidad |
|---|---|---|
| UFDS | Kruskal, Kruskal capacitado, islas del flujo DC, conectar componentes | O(α(n)) por operación |
| MST – Kruskal y Prim | Configuración A (red de mínima longitud) | O(E log E) / O(E log V) |
| Algoritmo voraz | Kruskal capacitado, selección de conductores, refuerzo por ΔV, enlaces de respaldo | O(E log E), O(E·h) |
| BFS / DFS | Árbol radial, alimentadores, recorrido de Euler (subárboles), islas | O(V + E) |
| Ordenamiento topológico (Kahn) | Propagar la caída de tensión en el sentido del flujo | O(V + E) |
| Puentes (DFS de Tarjan) | Tramos cuya falla desconecta la red | O(V + E) |
| Flujo máximo (Ford-Fulkerson: Edmonds-Karp / Dinic) | Demanda máxima abastecible con las capacidades | O(V²E) |
| Fuerza bruta y Backtracking | Validar el óptimo del MST en subgrafos | Exponencial |

### Configuraciones comparadas
- **A – MST clásico:** sólo minimiza longitud (enfoque del TB1).
- **B – Kruskal capacitado:** alimentadores radiales que no superan la capacidad (MW) ni la de su SET, conductor mínimo que soporta el flujo y refuerzo voraz hasta cumplir ΔV.
- **C – Robusta:** B planificada al {factor_C * 100:.0f} % de capacidad + enlaces de respaldo normalmente abiertos (voraz de cobertura) para el criterio N-1.

*Los parámetros de demanda, conductores y costos son referenciales y se pueden editar en la barra lateral.*
""", unsafe_allow_html=True)
