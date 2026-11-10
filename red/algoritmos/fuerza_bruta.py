"""Validacion del optimo en subgrafos pequenos: Fuerza Bruta y Backtracking.

Ambos buscan el arbol de expansion de menor costo para comprobar que el
resultado de Kruskal/Prim es optimo.
"""
from itertools import combinations

from .ufds import UFDS


def mst_fuerza_bruta(n, aristas):
    """Prueba TODAS las combinaciones de n-1 aristas: O(C(E, n-1) * n).
    Devuelve (mejor_costo, mejores_indices, combinaciones_evaluadas)."""
    mejor, mejor_sel, evaluadas = float("inf"), None, 0
    for sel in combinations(range(len(aristas)), n - 1):
        evaluadas += 1
        uf = UFDS(n)
        costo, valido = 0.0, True
        for i in sel:
            w, u, v = aristas[i]
            if uf.union(u, v) == -1:      # ciclo -> no es arbol
                valido = False
                break
            costo += w
        if valido and costo < mejor:
            mejor, mejor_sel = costo, list(sel)
    return mejor, mejor_sel, evaluadas


def mst_backtracking(n, aristas):
    """Backtracking: decide incluir/excluir cada arista (ordenadas por peso) y poda
    cuando (a) se forma un ciclo, (b) el costo parcial + cota inferior >= mejor,
    (c) no quedan aristas suficientes. Devuelve (mejor_costo, indices, nodos_explorados)."""
    orden = sorted(range(len(aristas)), key=lambda i: aristas[i][0])
    pesos = [aristas[i][0] for i in orden]
    mejor = [float("inf"), None]
    explorados = [0]

    def rec(k, elegidas, costo, padres):
        explorados[0] += 1
        faltan = n - 1 - len(elegidas)
        if faltan == 0:
            if costo < mejor[0]:
                mejor[0], mejor[1] = costo, [orden[i] for i in elegidas]
            return
        if len(orden) - k < faltan:
            return                                         # poda (c)
        cota = costo + sum(pesos[k:k + faltan])            # las mas baratas restantes
        if cota >= mejor[0]:
            return                                         # poda (b)
        w, u, v = aristas[orden[k]]
        uf = UFDS(n)
        uf.padre = list(padres)
        if uf.find(u) != uf.find(v):                       # incluir (si no hay ciclo)
            uf.union(u, v)
            rec(k + 1, elegidas + [k], costo + w, uf.padre)
        rec(k + 1, elegidas, costo, padres)                # excluir

    rec(0, [], 0.0, list(range(n)))
    return mejor[0], mejor[1], explorados[0]
