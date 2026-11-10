"""Diseno de configuraciones de red.

A - MST clasico (Kruskal o Prim): minimiza solo la longitud de cable (enfoque del TB1).
B - Red con restricciones electricas:
      1) Kruskal capacitado (voraz + UFDS): alimentadores radiales que no superan
         la capacidad del alimentador ni la de su SET.
      2) Racimos que quedaron sin fuente -> alimentador expreso a la SET con
         capacidad libre mas cercana.
      3) Flujo de potencia DC -> seleccion voraz del conductor minimo que soporta
         el flujo de cada tramo.
      4) Refuerzo voraz de conductores en el camino del nodo con peor caida de
         tension hasta cumplir el limite (mejor reduccion de caida por dolar).
C - Red robusta: B + enlaces de respaldo normalmente abiertos entre alimentadores
    (voraz: el enlace mas corto por cada par de alimentadores vecinos), para poder
    reconfigurar ante la falla de cualquier tramo (criterio N-1).
"""
import time

import numpy as np
import pandas as pd

from .algoritmos.mst import kruskal, kruskal_capacitado, prim
from .algoritmos.recorridos import arbol_radial
from .datos import haversine_np
from .electrico import CONDUCTORES, caida_pct, capacidad_mw, evaluar_red


def _lineas_df(cand, idx, cond, tipo):
    L = cand.iloc[idx][["u", "v", "dist_m"]].copy()
    L["long_km"] = L.dist_m / 1000.0
    L["cond"] = cond
    L["tipo"] = tipo
    L["estado"] = "cerrado"
    return L.drop(columns="dist_m").reset_index(drop=True)


def disenar_A(ctx, algoritmo="Kruskal", cond=1):
    """MST sobre el grafo candidato completo con un conductor uniforme."""
    t0 = time.perf_counter()
    aristas = list(zip(ctx["cand"].dist_m, ctx["cand"].u, ctx["cand"].v))
    if algoritmo == "Kruskal":
        idx, _ = kruskal(ctx["n"], aristas)
    else:
        inicio = int(np.flatnonzero(ctx["es_fuente"])[0])
        idx, _ = prim(ctx["n"], aristas, inicio)
    t_alg = time.perf_counter() - t0
    L = _lineas_df(ctx["cand"], idx, cond, "MST")
    return L, {"t_algoritmo_s": t_alg}


def _conductor_minimo(flujo_mw, p):
    for c, cd in enumerate(CONDUCTORES):
        if capacidad_mw(cd, p) >= flujo_mw:
            return c
    return len(CONDUCTORES) - 1


def _alimentador_expreso(ctx, nodos_racimo, carga, carga_fuente):
    """Une un racimo sin fuente con la SET con capacidad libre mas cercana."""
    lat, lon = ctx["lat"], ctx["lon"]
    fuentes = np.flatnonzero(ctx["es_fuente"])
    libre = np.array([ctx["cap_fuente"][s] - carga_fuente[s] for s in fuentes])
    ok = fuentes[libre >= carga]
    if len(ok) == 0:
        ok = fuentes
    best = None
    for x in nodos_racimo:
        d = haversine_np(lat[x], lon[x], lat[ok], lon[ok])
        j = int(np.argmin(d))
        if best is None or d[j] < best[2]:
            best = (int(ok[j]), int(x), float(d[j]))
    return best


def disenar_B(ctx, factor_plan=1.0):
    """factor_plan < 1 deja reserva en cada alimentador (carga de planificacion =
    factor_plan * capacidad) para poder recibir carga de un vecino que falla."""
    n, p = ctx["n"], ctx["p"]
    cand, dem = ctx["cand"], ctx["demanda"]
    t0 = time.perf_counter()
    aristas = list(zip(cand.dist_m, cand.u, cand.v))
    res = kruskal_capacitado(n, aristas, ctx["es_fuente"], dem, p.cap_alim_mw * factor_plan,
                             ctx["cap_fuente"] * factor_plan)
    L = _lineas_df(cand, res["elegidas"], 0, "alimentador")

    # 2) racimos sin fuente -> alimentador expreso
    extra = []
    carga_fuente = dict(res["carga_fuente"])
    for raiz, miembros in res["racimos"].items():
        c = res["carga"][raiz]
        s, x, d = _alimentador_expreso(ctx, miembros, c, carga_fuente)
        carga_fuente[s] += c
        extra.append({"u": min(s, x), "v": max(s, x), "long_km": d / 1000.0,
                      "cond": 0, "tipo": "expreso", "estado": "cerrado"})
    if extra:
        L = pd.concat([L, pd.DataFrame(extra)], ignore_index=True)
    t_kc = time.perf_counter() - t0

    # 3) flujo de potencia (red radial) -> conductor minimo por tramo
    t1 = time.perf_counter()
    Lr, _, _ = evaluar_red(n, L, dem, ctx["es_fuente"], ctx["cap_fuente"], p)
    L["cond"] = [_conductor_minimo(f, p) for f in Lr.flujo_mw]
    flujo = Lr.flujo_mw.to_numpy()

    # 4) refuerzo voraz por caida de tension (sobre el arbol radial)
    fuentes = [int(s) for s in np.flatnonzero(ctx["es_fuente"])]
    arb = arbol_radial(n, list(zip(L.u, L.v)), fuentes)
    pre, tin, tout = arb["preorden"], arb["tin"], arb["tout"]
    ep = arb["arista_padre"]
    lk = L.long_km.to_numpy()
    cond = L.cond.to_numpy().copy()
    dv_arista = np.array([caida_pct(f, l, CONDUCTORES[c], p) for f, l, c in zip(flujo, lk, cond)])
    dv = np.zeros(n)
    for w in pre:
        if ep[w] != -1:
            dv[w] = dv[arb["padre"][w]] + dv_arista[ep[w]]
    pre_arr = np.array(pre)
    bloqueado = np.zeros(n, bool)           # nodos que ya no se pueden mejorar
    mejoras = 0
    for _ in range(50000):
        valores = np.where(bloqueado[pre_arr], -np.inf, dv[pre_arr])
        peor = int(pre_arr[np.argmax(valores)])
        if bloqueado[peor] or dv[peor] <= p.dv_max:
            break
        mejor = None
        w = peor
        while ep[w] != -1:                      # recorrer el camino hacia la fuente
            e = ep[w]
            c = cond[e]
            if c + 1 < len(CONDUCTORES):
                gan = dv_arista[e] - caida_pct(flujo[e], lk[e], CONDUCTORES[c + 1], p)
                costo = (CONDUCTORES[c + 1].costo_km - CONDUCTORES[c].costo_km) * lk[e]
                ratio = gan / max(costo, 1e-9)
                if mejor is None or ratio > mejor[0]:
                    mejor = (ratio, e, w, gan)
            w = arb["padre"][w]
        if mejor is None:                       # todo el camino ya tiene el mejor conductor
            bloqueado[peor] = True
            continue
        _, e, hijo, gan = mejor
        cond[e] += 1
        dv_arista[e] -= gan
        dv[pre_arr[tin[hijo]:tout[hijo] + 1]] -= gan   # baja la caida de todo el subarbol
        mejoras += 1
    L["cond"] = cond
    t_cond = time.perf_counter() - t1
    info = {"t_algoritmo_s": t_kc, "t_conductores_s": t_cond, "refuerzos": mejoras,
            "alimentadores": len(res["fuente_de_alim"]) + len(extra), "expresos": len(extra)}
    return L, info


def disenar_C(ctx, factor_plan=0.5, umbral_mw=0.3):
    """Red robusta: B planificada con reserva (factor_plan) + enlaces de respaldo
    normalmente abiertos (voraz de cobertura)."""
    L_B, info_B = disenar_B(ctx, factor_plan)
    L, info = agregar_respaldos(ctx, L_B, umbral_mw)
    info_B.update(info)
    info_B["t_algoritmo_s"] += info["t_respaldo_s"]
    return L, info_B


def agregar_respaldos(ctx, L_B, umbral_mw=0.3):
    """Voraz de cobertura de enlaces de respaldo (normalmente abiertos).

    Un enlace (a, b) que no esta en la red "protege" a todos los tramos del camino
    a..b en el bosque radial (si falla uno de ellos, cerrar (a, b) reconecta lo
    aislado). Se recorren las aristas candidatas de menor a mayor longitud y se
    agrega una solo si protege al menos un tramo aun no protegido cuya falla
    aislaria >= umbral_mw. Costo: O(E_cand * profundidad)."""
    n, cand = ctx["n"], ctx["cand"]
    t0 = time.perf_counter()
    fuentes = [int(s) for s in np.flatnonzero(ctx["es_fuente"])]
    arb = arbol_radial(n, list(zip(L_B.u, L_B.v)), fuentes)
    padre, tin, tout, raiz = arb["padre"], arb["tin"], arb["tout"], arb["raiz"]
    carga_sub = np.array(ctx["demanda"], dtype=float).copy()
    for w in reversed(arb["preorden"]):
        if padre[w] != -1:
            carga_sub[padre[w]] += carga_sub[w]
    proteger = np.array([padre[w] != -1 and carga_sub[w] >= umbral_mw for w in range(n)])
    cubierto = np.zeros(n, bool)
    usadas = set(zip(L_B.u, L_B.v))

    def ancestro(w, x):
        return tin[w] <= tin[x] <= tout[w]

    filas = []
    for r in cand.sort_values("dist_m").itertuples():
        a, b = int(r.u), int(r.v)
        if (a, b) in usadas or ctx["es_fuente"][a] or ctx["es_fuente"][b]:
            continue
        if raiz[a] == -1 or raiz[b] == -1:
            continue
        camino = []
        for x, y in ((a, b), (b, a)):
            w = x
            while padre[w] != -1 and not ancestro(w, y):
                camino.append(w)
                w = padre[w]
        if any(proteger[w] and not cubierto[w] for w in camino):
            cubierto[camino] = True
            filas.append({"u": a, "v": b, "long_km": r.dist_m / 1000.0, "cond": 2,
                          "tipo": "respaldo", "estado": "abierto"})
    L = pd.concat([L_B, pd.DataFrame(filas)], ignore_index=True)
    cobertura = 100.0 * cubierto[proteger].mean() if proteger.any() else 100.0
    return L, {"t_respaldo_s": time.perf_counter() - t0, "enlaces_respaldo": len(filas),
               "cobertura_pct": cobertura}
