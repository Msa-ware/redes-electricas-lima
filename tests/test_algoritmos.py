"""Pruebas de los algoritmos.  Ejecutar:  python -m pytest -q"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from red.algoritmos.flujo_maximo import RedFlujo, dinic, edmonds_karp
from red.algoritmos.fuerza_bruta import mst_backtracking, mst_fuerza_bruta
from red.algoritmos.mst import kruskal, kruskal_capacitado, prim
from red.algoritmos.recorridos import arbol_radial, orden_topologico, puentes_tarjan
from red.algoritmos.ufds import UFDS
from red.electrico import ParamElectricos, evaluar_red


def grafo_aleatorio(n, m, semilla):
    rnd = random.Random(semilla)
    aristas = [(rnd.uniform(1, 100), i, rnd.randrange(i)) for i in range(1, n)]   # conexo
    while len(aristas) < m:
        a, b = rnd.sample(range(n), 2)
        aristas.append((rnd.uniform(1, 100), a, b))
    return aristas


def test_ufds():
    uf = UFDS(5)
    assert uf.union(0, 1) != -1 and uf.union(1, 2) != -1
    assert uf.mismo_conjunto(0, 2) and not uf.mismo_conjunto(0, 3)
    assert uf.union(0, 2) == -1 and uf.num_conjuntos == 3


def test_kruskal_prim_fuerza_bruta_backtracking():
    for s in range(15):
        n = 6
        ar = grafo_aleatorio(n, 11, s)
        _, ck = kruskal(n, ar)
        _, cp = prim(n, ar)
        cf, _, _ = mst_fuerza_bruta(n, ar)
        cb, _, _ = mst_backtracking(n, ar)
        assert abs(ck - cp) < 1e-9 and abs(ck - cf) < 1e-9 and abs(ck - cb) < 1e-9


def test_kruskal_capacitado_respeta_capacidad():
    rnd = random.Random(1)
    n = 60
    ar = grafo_aleatorio(n, 200, 3)
    es_fuente = [i < 3 for i in range(n)]
    dem = [0.0 if es_fuente[i] else rnd.uniform(0.1, 0.5) for i in range(n)]
    cap_f = [100.0 if es_fuente[i] else 0.0 for i in range(n)]
    res = kruskal_capacitado(n, ar, es_fuente, dem, 2.0, cap_f)
    sel = [ar[i] for i in res["elegidas"]]
    arb = arbol_radial(n, [(u, v) for _, u, v in sel], [0, 1, 2])
    carga = {}
    for x in range(n):
        if arb["alimentador"][x] != -1:
            carga[arb["alimentador"][x]] = carga.get(arb["alimentador"][x], 0) + dem[x]
    assert all(c <= 2.0 + 1e-9 for c in carga.values())
    assert len(sel) == len(set(sel))          # sin aristas repetidas


def test_flujo_maximo_dinic_igual_edmonds_karp():
    for s in range(10):
        rnd = random.Random(s)
        n = 12
        arcos = [(rnd.randrange(n), rnd.randrange(n), rnd.randint(1, 20)) for _ in range(40)]
        r1, r2 = RedFlujo(n), RedFlujo(n)
        for u, v, c in arcos:
            if u != v:
                r1.agregar_arco(u, v, c)
                r2.agregar_arco(u, v, c)
        assert abs(edmonds_karp(r1, 0, n - 1) - dinic(r2, 0, n - 1)) < 1e-9


def test_puentes_y_topologico():
    # triangulo 0-1-2 + cola 2-3-4: los puentes son (2,3) y (3,4)
    aristas = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
    assert sorted(puentes_tarjan(5, aristas)) == [3, 4]
    orden, _ = orden_topologico(4, [(0, 1, None), (1, 2, None), (0, 3, None), (3, 2, None)])
    assert orden.index(0) < orden.index(1) < orden.index(2) and orden.index(3) < orden.index(2)


def test_flujo_dc_radial_igual_suma_de_demandas():
    # fuente 0 -> 1 -> 2 y 1 -> 3
    lineas = pd.DataFrame({"u": [0, 1, 1], "v": [1, 2, 3], "long_km": [1.0, 0.5, 0.7],
                           "cond": [2, 0, 1], "estado": "cerrado", "tipo": "x"})
    dem = np.array([0.0, 0.5, 1.0, 2.0])
    es_f = np.array([True, False, False, False])
    Lr, nodos, m = evaluar_red(4, lineas, dem, es_f, np.array([10.0, 0, 0, 0]), ParamElectricos())
    assert np.allclose(Lr.flujo_mw.to_numpy(), [3.5, 1.0, 2.0])
    assert nodos["dv_pct"][2] > nodos["dv_pct"][1] > 0
