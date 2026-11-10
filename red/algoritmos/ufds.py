"""Union-Find Disjoint Sets (UFDS) con compresion de caminos y union por rango.

find y union cuestan O(alpha(n)) amortizado (practicamente constante).
"""


class UFDS:
    def __init__(self, n):
        self.padre = list(range(n))
        self.rango = [0] * n
        self.tamano = [1] * n
        self.num_conjuntos = n

    def find(self, x):
        raiz = x
        while self.padre[raiz] != raiz:
            raiz = self.padre[raiz]
        while self.padre[x] != raiz:          # compresion de caminos
            self.padre[x], x = raiz, self.padre[x]
        return raiz

    def mismo_conjunto(self, a, b):
        return self.find(a) == self.find(b)

    def union(self, a, b):
        """Une los conjuntos de a y b. Devuelve la nueva raiz (o -1 si ya estaban unidos)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return -1
        if self.rango[ra] < self.rango[rb]:
            ra, rb = rb, ra
        self.padre[rb] = ra
        self.tamano[ra] += self.tamano[rb]
        if self.rango[ra] == self.rango[rb]:
            self.rango[ra] += 1
        self.num_conjuntos -= 1
        return ra
