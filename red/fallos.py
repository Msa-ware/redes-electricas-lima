"""Robustez ante fallos.

- analisis_n1: para CADA tramo cerrado de la red radial calcula la demanda que
  queda aislada si falla (subarbol aguas abajo, recorrido de Euler del DFS) y si
  existe un enlace de respaldo que la pueda recuperar sin sobrecargar el
  alimentador/SET que la recibe.
- simular_falla: abre un tramo, BFS para hallar los nodos aislados, cierra el
  mejor enlace de respaldo y recalcula el flujo DC y las tensiones.
- puentes: aristas puente (Tarjan) de la topologia con todos los respaldos.
- flujo_maximo_red: demanda maxima abastecible respetando capacidades (Dinic).
"""
import numpy as np

from .algoritmos.flujo_maximo import RedFlujo, dinic
from .algoritmos.recorridos import arbol_radial, puentes_tarjan
from .electrico import CONDUCTORES, capacidad_mw, evaluar_red

# sobrecarga admisible de corta duracion en emergencia (120 % de la capacidad nominal)
SOBRECARGA_EMERGENCIA = 1.2


def _subarbol_carga(arb, demanda, n):
    """Suma de demanda de cada subarbol (preorden invertido)."""
    carga = np.array(demanda, dtype=float).copy()
    for w in reversed(arb["preorden"]):
        p = arb["padre"][w]
        if p != -1:
            carga[p] += carga[w]
    return carga


def analisis_n1(ctx, L):
    n, p, dem = ctx["n"], ctx["p"], ctx["demanda"]
    fuentes = [int(s) for s in np.flatnonzero(ctx["es_fuente"])]
    cerr = L[L.estado == "cerrado"]
    idx_cerr = cerr.index.to_numpy()
    arb = arbol_radial(n, list(zip(cerr.u, cerr.v)), fuentes)
    padre, alim, raiz = arb["padre"], arb["alimentador"], arb["raiz"]
    tin, tout = arb["tin"], arb["tout"]
    carga_sub = _subarbol_carga(arb, dem, n)
    carga_alim = {a: carga_sub[a] for a in set(alim) if a != -1}
    carga_fuente = {}
    for a, c in carga_alim.items():
        carga_fuente[raiz[a]] = carga_fuente.get(raiz[a], 0.0) + c

    ties = L[(L.tipo == "respaldo")]
    # para cada tramo (identificado por su nodo hijo) -> enlaces candidatos
    candidatos = {}
    for t in ties.itertuples():
        a, b = int(t.u), int(t.v)
        if raiz[a] == -1 or raiz[b] == -1:
            continue
        for x, y in ((a, b), (b, a)):
            w = x                                  # tramos en el camino x -> fuente
            while padre[w] != -1:
                if not (tin[w] <= tin[y] <= tout[w]):   # y queda fuera del subarbol
                    candidatos.setdefault(w, []).append((t.Index, y))
                w = padre[w]

    # capacidad (MW) de cada tramo del arbol, indexada por su nodo hijo
    cond_cerr = cerr.cond.to_numpy()
    cap_tramo = np.full(n, np.inf)
    for z in range(n):
        if arb["arista_padre"][z] != -1:
            cap_tramo[z] = capacidad_mw(CONDUCTORES[int(cond_cerr[arb["arista_padre"][z]])], p)
    cap_tie = {t.Index: capacidad_mw(CONDUCTORES[int(t.cond)], p) for t in ties.itertuples()}

    def ruta_soporta(y, w, S):
        """Los tramos entre y y su ancestro comun con w reciben +S MW."""
        z = y
        while padre[z] != -1 and not (tin[z] <= tin[w] <= tout[z]):
            if carga_sub[z] + S > SOBRECARGA_EMERGENCIA * cap_tramo[z]:
                return False
            z = padre[z]
        return True

    filas = []
    for w in range(n):
        if padre[w] == -1:
            continue
        S = carga_sub[w]
        ok, enlace = False, None
        for ti, y in candidatos.get(w, []):
            g = alim[y]
            if g == alim[w]:
                libre_al, libre_f = np.inf, np.inf     # mismo alimentador: la carga no cambia
            else:
                libre_al = SOBRECARGA_EMERGENCIA * p.cap_alim_mw - carga_alim[g]
                libre_f = ctx["cap_fuente"][raiz[y]] - carga_fuente[raiz[y]]
            if (S <= libre_al + 1e-9 and S <= libre_f + 1e-9 and S <= cap_tie[ti]
                    and ruta_soporta(y, w, S)):
                ok, enlace = True, ti
                break
        filas.append((int(idx_cerr[arb["arista_padre"][w]]), w, S, ok, enlace,
                      arb["tout"][w] - arb["tin"][w] + 1))
    import pandas as pd
    df = pd.DataFrame(filas, columns=["linea", "nodo_hijo", "demanda_aislada_mw", "restaurable",
                                      "enlace", "nodos_aislados"])
    dem_total = float(np.sum(dem))
    no_rest = df[~df.restaurable]
    resumen = {
        "tramos": len(df),
        "pct_restaurables": 100.0 * df.restaurable.mean() if len(df) else 0.0,
        "ens_promedio_mw": float(no_rest.demanda_aislada_mw.sum() / max(len(df), 1)),
        "peor_falla_mw": float(no_rest.demanda_aislada_mw.max()) if len(no_rest) else 0.0,
        "dem_total_mw": dem_total,
    }
    return df, resumen


def simular_falla(ctx, L, linea, n1=None):
    """Abre la linea `linea`, cierra el enlace de respaldo indicado por el analisis
    N-1 (si existe) y evalua la red resultante."""
    n, p = ctx["n"], ctx["p"]
    L2 = L.copy()
    L2.loc[linea, "estado"] = "fallada"
    sin_resp, nodos_sin, met_sin = evaluar_red(n, L2, ctx["demanda"], ctx["es_fuente"], ctx["cap_fuente"], p)
    aislados = np.flatnonzero(~nodos_sin["suministrado"] & (ctx["demanda"] > 0))
    enlace = None
    if n1 is not None:
        fila = n1[n1.linea == linea]
        if len(fila) and fila.iloc[0].restaurable:
            enlace = int(fila.iloc[0].enlace)
    if enlace is None:
        return {"aislados": aislados, "restaurados": np.array([], int), "enlace": None,
                "L": sin_resp, "nodos": nodos_sin, "met": met_sin}
    L3 = L2.copy()
    L3.loc[enlace, "estado"] = "cerrado"
    con_resp, nodos_con, met_con = evaluar_red(n, L3, ctx["demanda"], ctx["es_fuente"], ctx["cap_fuente"], p)
    restaurados = aislados[nodos_con["suministrado"][aislados]]
    return {"aislados": aislados, "restaurados": restaurados, "enlace": enlace,
            "L": con_resp, "nodos": nodos_con, "met": met_con}


def contar_puentes(ctx, L):
    """Puentes de la topologia considerando todos los enlaces (cerrados y de respaldo)."""
    usar = L[L.estado != "fallada"]
    pts = puentes_tarjan(ctx["n"], list(zip(usar.u, usar.v)))
    return len(pts)


def flujo_maximo_red(ctx, L, incluir_respaldo=True):
    """Red de flujo: super-fuente -> SET (cap. de la SET), lineas en ambos sentidos
    (cap. termica del conductor), subestacion -> super-sumidero (su demanda).
    Devuelve el flujo maximo (MW) = demanda que la red puede abastecer."""
    n, p, dem = ctx["n"], ctx["p"], ctx["demanda"]
    S, T = n, n + 1
    red = RedFlujo(n + 2)
    for s in np.flatnonzero(ctx["es_fuente"]):
        red.agregar_arco(S, int(s), ctx["cap_fuente"][s])
    usar = L if incluir_respaldo else L[L.estado == "cerrado"]
    for r in usar.itertuples():
        c = capacidad_mw(CONDUCTORES[int(r.cond)], p)
        red.agregar_arco(int(r.u), int(r.v), c, c)
    for i in np.flatnonzero(dem > 0):
        red.agregar_arco(int(i), T, dem[i])
    return dinic(red, S, T)
