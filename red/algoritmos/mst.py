"""Arbol de Expansion Minima (MST): Kruskal + UFDS, Prim con heap y
Kruskal capacitado (voraz) que respeta la capacidad de cada alimentador.

aristas: lista de tuplas (peso, u, v).
"""
import heapq

from .ufds import UFDS


def kruskal(n, aristas):
    """Kruskal clasico. O(E log E). Devuelve (indices_elegidos, peso_total)."""
    orden = sorted(range(len(aristas)), key=lambda i: aristas[i][0])
    uf = UFDS(n)
    elegidas, total = [], 0.0
    for i in orden:
        w, u, v = aristas[i]
        if uf.union(u, v) != -1:          # no forma ciclo
            elegidas.append(i)
            total += w
            if len(elegidas) == n - 1:
                break
    return elegidas, total


def prim(n, aristas, inicio=0):
    """Prim con cola de prioridad (heapq). O(E log V).
    Si el grafo no es conexo, reinicia en cada componente (bosque)."""
    ady = [[] for _ in range(n)]
    for i, (w, u, v) in enumerate(aristas):
        ady[u].append((w, v, i))
        ady[v].append((w, u, i))
    en_arbol = [False] * n
    elegidas, total = [], 0.0
    for s in [inicio] + list(range(n)):
        if en_arbol[s]:
            continue
        en_arbol[s] = True
        heap = list(ady[s])
        heapq.heapify(heap)
        while heap:
            w, v, i = heapq.heappop(heap)
            if en_arbol[v]:
                continue
            en_arbol[v] = True
            elegidas.append(i)
            total += w
            for item in ady[v]:
                if not en_arbol[item[1]]:
                    heapq.heappush(heap, item)
    return elegidas, total


def kruskal_capacitado(n, aristas, es_fuente, demanda, cap_alimentador, cap_fuente):
    """Kruskal capacitado (heuristica voraz) para disenar alimentadores radiales.

    Recorre las aristas de menor a mayor longitud (como Kruskal) pero solo acepta
    una arista si, despues de unir, ningun alimentador supera `cap_alimentador` (MW)
    y ninguna fuente (SET/central) supera su capacidad `cap_fuente[s]` (MW).

    Conjuntos del UFDS:
      - "racimo": grupo de subestaciones aun sin fuente.
      - "alimentador": grupo ya conectado a una fuente (por una sola arista).
    Dos alimentadores nunca se unen (la red debe quedar radial).

    Devuelve dict con: elegidas (indices), alim_de_raiz, fuente_de_alim,
    carga_fuente, uf, carga (por raiz) y los racimos que quedaron sin fuente.
    """
    orden = sorted(range(len(aristas)), key=lambda i: aristas[i][0])
    uf = UFDS(n)
    carga = list(demanda)                 # carga del conjunto (valida en la raiz)
    alim = [-1] * n                       # id de alimentador del conjunto (en la raiz)
    fuente_de_alim = []
    carga_fuente = {s: 0.0 for s in range(n) if es_fuente[s]}
    elegidas = []

    for i in orden:
        w, u, v = aristas[i]
        if es_fuente[u] and es_fuente[v]:
            continue
        if es_fuente[v]:
            u, v = v, u
        if es_fuente[u]:                                  # u fuente, v subestacion
            s, r = u, uf.find(v)
            if alim[r] != -1:
                continue                                  # ya esta alimentado
            if carga[r] <= cap_alimentador and carga_fuente[s] + carga[r] <= cap_fuente[s]:
                alim[r] = len(fuente_de_alim)
                fuente_de_alim.append(s)
                carga_fuente[s] += carga[r]
                elegidas.append(i)
            continue
        ru, rv = uf.find(u), uf.find(v)
        if ru == rv:
            continue                                      # formaria ciclo
        fu, fv = alim[ru], alim[rv]
        if fu != -1 and fv != -1:
            continue                                      # uniria dos alimentadores
        nueva = carga[ru] + carga[rv]
        if nueva > cap_alimentador:
            continue                                      # alimentador sobrecargado
        f = fu if fu != -1 else fv
        if f != -1:
            extra = carga[rv] if fu != -1 else carga[ru]
            s = fuente_de_alim[f]
            if carga_fuente[s] + extra > cap_fuente[s]:
                continue                                  # SET sobrecargada
            carga_fuente[s] += extra
        r = uf.union(ru, rv)
        carga[r] = nueva
        alim[r] = f
        elegidas.append(i)

    racimos = {}
    for x in range(n):
        if es_fuente[x]:
            continue
        r = uf.find(x)
        if alim[r] == -1:
            racimos.setdefault(r, []).append(x)
    return {"elegidas": elegidas, "alim": alim, "fuente_de_alim": fuente_de_alim,
            "carga_fuente": carga_fuente, "uf": uf, "carga": carga, "racimos": racimos}
