"""Recorridos en grafos: BFS, DFS (arbol radial / recorrido de Euler),
ordenamiento topologico (Kahn) y puentes (Tarjan).

Todas las funciones son O(V + E).
"""
from collections import deque


def lista_adyacencia(n, aristas):
    """aristas: lista de (u, v). Devuelve ady[u] = [(v, indice_arista), ...]."""
    ady = [[] for _ in range(n)]
    for i, (u, v) in enumerate(aristas):
        ady[u].append((v, i))
        ady[v].append((u, i))
    return ady


def bfs_multifuente(n, ady, fuentes):
    """BFS desde varias fuentes a la vez. Devuelve (visitado, padre, orden)."""
    visitado = [False] * n
    padre = [-1] * n
    orden = []
    cola = deque()
    for s in fuentes:
        if not visitado[s]:
            visitado[s] = True
            cola.append(s)
    while cola:
        u = cola.popleft()
        orden.append(u)
        for v, _ in ady[u]:
            if not visitado[v]:
                visitado[v] = True
                padre[v] = u
                cola.append(v)
    return visitado, padre, orden


def componentes_conexas(n, ady):
    """Etiqueta cada nodo con el id de su componente conexa (BFS)."""
    comp = [-1] * n
    c = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        comp[s] = c
        cola = deque([s])
        while cola:
            u = cola.popleft()
            for v, _ in ady[u]:
                if comp[v] == -1:
                    comp[v] = c
                    cola.append(v)
        c += 1
    return comp, c


def arbol_radial(n, aristas, fuentes):
    """DFS iterativo desde cada fuente sobre una red RADIAL (bosque).

    Devuelve un dict con:
      padre[v], arista_padre[v] (indice de la arista que une v con su padre),
      raiz[v] (fuente que alimenta a v), alimentador[v] (hijo directo de la fuente
      del que cuelga v -> identifica el alimentador/feeder), preorden (lista),
      tin[v] y tout[v] (recorrido de Euler: u es descendiente de v
      si tin[v] <= tin[u] <= tout[v]).
    Nodos no alcanzados quedan con raiz = -1.
    """
    ady = lista_adyacencia(n, aristas)
    padre = [-1] * n
    arista_padre = [-1] * n
    raiz = [-1] * n
    alimentador = [-1] * n
    tin = [-1] * n
    tout = [-1] * n
    preorden = []
    for s in fuentes:
        if raiz[s] != -1:
            continue
        raiz[s] = s
        tin[s] = len(preorden)
        preorden.append(s)
        pila = [(s, iter(ady[s]))]
        while pila:
            u, it = pila[-1]
            avanzo = False
            for v, ei in it:
                if raiz[v] == -1:
                    raiz[v] = s
                    padre[v] = u
                    arista_padre[v] = ei
                    alimentador[v] = v if u == s else alimentador[u]
                    tin[v] = len(preorden)
                    preorden.append(v)
                    pila.append((v, iter(ady[v])))
                    avanzo = True
                    break
            if not avanzo:
                tout[u] = len(preorden) - 1
                pila.pop()
    return {"padre": padre, "arista_padre": arista_padre, "raiz": raiz,
            "alimentador": alimentador, "preorden": preorden, "tin": tin, "tout": tout}


def orden_topologico(n, arcos):
    """Algoritmo de Kahn. arcos: lista de (u, v, dato) dirigidos u -> v.
    Devuelve (orden, entrantes) donde entrantes[v] = [(u, dato), ...].
    Si el grafo dirigido tiene ciclo, el orden devuelto tiene menos de n nodos."""
    grado_in = [0] * n
    salientes = [[] for _ in range(n)]
    entrantes = [[] for _ in range(n)]
    for u, v, dato in arcos:
        salientes[u].append(v)
        entrantes[v].append((u, dato))
        grado_in[v] += 1
    cola = deque(i for i in range(n) if grado_in[i] == 0)
    orden = []
    while cola:
        u = cola.popleft()
        orden.append(u)
        for v in salientes[u]:
            grado_in[v] -= 1
            if grado_in[v] == 0:
                cola.append(v)
    return orden, entrantes


def puentes_tarjan(n, aristas):
    """Aristas puente (cuya falla desconecta el grafo). DFS iterativo de Tarjan
    con tiempos de descubrimiento y low-link. Devuelve lista de indices de arista."""
    ady = lista_adyacencia(n, aristas)
    disc = [-1] * n
    low = [0] * n
    puentes = []
    t = 0
    for s in range(n):
        if disc[s] != -1:
            continue
        disc[s] = low[s] = t
        t += 1
        pila = [(s, -1, iter(ady[s]))]
        while pila:
            u, e_padre, it = pila[-1]
            avanzo = False
            for v, ei in it:
                if ei == e_padre:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = t
                    t += 1
                    pila.append((v, ei, iter(ady[v])))
                    avanzo = True
                    break
                low[u] = min(low[u], disc[v])
            if not avanzo:
                pila.pop()
                if pila:
                    p = pila[-1][0]
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        puentes.append(e_padre)
    return puentes
