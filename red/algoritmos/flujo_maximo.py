"""Flujo maximo (metodo de Ford-Fulkerson).

- edmonds_karp: Ford-Fulkerson con caminos aumentantes por BFS. O(V * E^2).
- dinic: Ford-Fulkerson con grafo de niveles (BFS) y flujos bloqueantes (DFS
  iterativo). O(V^2 * E); mucho mas rapido en la practica, se usa en la red completa.
"""
from collections import deque


class RedFlujo:
    def __init__(self, n):
        self.n = n
        self.cabeza = [[] for _ in range(n)]   # indices de arcos que salen de cada nodo
        self.destino = []
        self.cap = []

    def agregar_arco(self, u, v, c, c_inverso=0.0):
        """Arco u->v con capacidad c (y v->u con c_inverso, para lineas no dirigidas)."""
        self.cabeza[u].append(len(self.destino)); self.destino.append(v); self.cap.append(float(c))
        self.cabeza[v].append(len(self.destino)); self.destino.append(u); self.cap.append(float(c_inverso))
        return len(self.destino) - 2

    def flujo_en(self, arco, cap_original):
        return cap_original - self.cap[arco]


def edmonds_karp(red, s, t):
    flujo = 0.0
    while True:
        previo = [-1] * red.n
        previo[s] = -2
        cola = deque([s])
        while cola and previo[t] == -1:
            u = cola.popleft()
            for a in red.cabeza[u]:
                v = red.destino[a]
                if previo[v] == -1 and red.cap[a] > 1e-9:
                    previo[v] = a
                    cola.append(v)
        if previo[t] == -1:
            return flujo
        # cuello de botella
        b, v = float("inf"), t
        while v != s:
            a = previo[v]
            b = min(b, red.cap[a])
            v = red.destino[a ^ 1]
        v = t
        while v != s:
            a = previo[v]
            red.cap[a] -= b
            red.cap[a ^ 1] += b
            v = red.destino[a ^ 1]
        flujo += b


def dinic(red, s, t):
    n, flujo = red.n, 0.0
    cabeza, destino, cap = red.cabeza, red.destino, red.cap
    while True:
        nivel = [-1] * n
        nivel[s] = 0
        cola = deque([s])
        while cola:
            u = cola.popleft()
            for a in cabeza[u]:
                if cap[a] > 1e-9 and nivel[destino[a]] < 0:
                    nivel[destino[a]] = nivel[u] + 1
                    cola.append(destino[a])
        if nivel[t] < 0:
            return flujo
        ptr = [0] * n
        while True:                     # buscar caminos en el grafo de niveles
            camino = []                 # arcos del camino actual
            u = s
            while u != t:
                avanzo = False
                while ptr[u] < len(cabeza[u]):
                    a = cabeza[u][ptr[u]]
                    v = destino[a]
                    if cap[a] > 1e-9 and nivel[v] == nivel[u] + 1:
                        camino.append(a)
                        u = v
                        avanzo = True
                        break
                    ptr[u] += 1
                if not avanzo:
                    if u == s:
                        break
                    nivel[u] = -1               # nodo sin salida: se poda
                    a = camino.pop()
                    u = destino[a ^ 1]
                    ptr[u] += 1
            if u != t:
                break
            b = min(cap[a] for a in camino)
            for a in camino:
                cap[a] -= b
                cap[a ^ 1] += b
            flujo += b
