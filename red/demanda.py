"""Modelo de demanda energetica por zonas.

1. Cada subestacion de distribucion (SED) se asigna al distrito cuyo centroide
   esta mas cerca (aproximacion de Voronoi).
2. Cada distrito tiene un tipo de zona (comercial, residencial alto/medio,
   popular, industrial) con una demanda maxima tipica por SED (kW) y una curva
   de carga diaria (factor 0..1 por hora).
3. Demanda de la SED i a la hora h:
      P_i(h) = kW_zona * curva_zona[h] * factor_diversidad_i * (1 + crecimiento)
   El factor de diversidad (0.8 - 1.2) es pseudoaleatorio pero reproducible
   (semilla fija) y representa que no todas las SED tienen el mismo transformador.

Los valores son referenciales (editables en la app): SED urbanas de 250-630 kVA
con factor de utilizacion 0.5-0.7 en hora punta.
"""
import numpy as np
import pandas as pd

# (distrito, lat, lon, tipo de zona) - centroides aproximados
DISTRITOS = [
    ("Cercado de Lima", -12.0464, -77.0428, "comercial"),
    ("San Isidro", -12.0977, -77.0365, "comercial"),
    ("Miraflores", -12.1211, -77.0297, "comercial"),
    ("La Victoria", -12.0730, -77.0160, "comercial"),
    ("Lince", -12.0835, -77.0350, "comercial"),
    ("Jesús María", -12.0760, -77.0480, "residencial_alto"),
    ("San Borja", -12.1000, -76.9990, "residencial_alto"),
    ("Santiago de Surco", -12.1450, -76.9910, "residencial_alto"),
    ("La Molina", -12.0850, -76.9330, "residencial_alto"),
    ("Barranco", -12.1490, -77.0210, "residencial_alto"),
    ("Magdalena del Mar", -12.0910, -77.0700, "residencial_alto"),
    ("Pueblo Libre", -12.0740, -77.0630, "residencial_alto"),
    ("San Miguel", -12.0770, -77.0910, "residencial_alto"),
    ("Surquillo", -12.1130, -77.0110, "residencial_medio"),
    ("Breña", -12.0590, -77.0510, "residencial_medio"),
    ("Rímac", -12.0290, -77.0300, "residencial_medio"),
    ("San Luis", -12.0750, -76.9950, "residencial_medio"),
    ("San Martín de Porres", -12.0000, -77.0700, "residencial_medio"),
    ("Los Olivos", -11.9700, -77.0700, "residencial_medio"),
    ("Independencia", -11.9900, -77.0500, "residencial_medio"),
    ("Chorrillos", -12.1750, -77.0150, "residencial_medio"),
    ("Comas", -11.9350, -77.0500, "popular"),
    ("Carabayllo", -11.8700, -77.0300, "popular"),
    ("Puente Piedra", -11.8650, -77.0750, "popular"),
    ("Ancón", -11.7700, -77.1700, "popular"),
    ("Santa Rosa", -11.8000, -77.1650, "popular"),
    ("San Juan de Lurigancho", -11.9800, -77.0000, "popular"),
    ("El Agustino", -12.0450, -76.9950, "popular"),
    ("Lurigancho-Chosica", -11.9400, -76.7000, "popular"),
    ("Chaclacayo", -11.9750, -76.7700, "popular"),
    ("Cieneguilla", -12.1100, -76.8100, "popular"),
    ("Villa María del Triunfo", -12.1600, -76.9300, "popular"),
    ("San Juan de Miraflores", -12.1600, -76.9700, "popular"),
    ("Villa El Salvador", -12.2100, -76.9400, "popular"),
    ("Pachacámac", -12.2300, -76.8600, "popular"),
    ("Punta Hermosa", -12.3350, -76.8250, "popular"),
    ("San Bartolo", -12.3900, -76.7800, "popular"),
    ("Pucusana", -12.4800, -76.7900, "popular"),
    ("Santa Anita", -12.0450, -76.9700, "industrial"),
    ("Ate", -12.0300, -76.9200, "industrial"),
    ("Lurín", -12.2750, -76.8700, "industrial"),
    ("Callao", -12.0550, -77.1200, "industrial"),
    ("Ventanilla", -11.8800, -77.1300, "industrial"),
]

ZONAS = ["comercial", "residencial_alto", "residencial_medio", "popular", "industrial"]
NOMBRE_ZONA = {"comercial": "Comercial", "residencial_alto": "Residencial alto",
               "residencial_medio": "Residencial medio", "popular": "Residencial popular",
               "industrial": "Industrial"}

# Demanda maxima tipica por SED (kW) en la hora punta de su zona
KW_POR_SED = {"comercial": 300.0, "residencial_alto": 220.0, "residencial_medio": 170.0,
              "popular": 120.0, "industrial": 350.0}

# Curvas de carga diaria tipicas (factor respecto a la punta), horas 0..23
CURVAS = {
    "comercial":         [.35, .32, .30, .30, .30, .33, .40, .55, .75, .90, .96, 1.0,
                          1.0, .98, .97, .96, .93, .88, .80, .75, .68, .58, .48, .40],
    "residencial_alto":  [.45, .40, .37, .35, .35, .38, .48, .58, .60, .58, .57, .58,
                          .62, .62, .60, .58, .60, .70, .88, 1.0, .98, .90, .75, .58],
    "residencial_medio": [.42, .38, .35, .34, .34, .38, .50, .58, .56, .54, .54, .56,
                          .60, .58, .56, .55, .58, .70, .90, 1.0, .97, .88, .72, .55],
    "popular":           [.40, .36, .34, .33, .34, .40, .52, .55, .52, .50, .50, .52,
                          .56, .54, .52, .52, .56, .70, .92, 1.0, .96, .85, .68, .50],
    "industrial":        [.60, .58, .57, .57, .58, .62, .75, .90, .98, 1.0, 1.0, .98,
                          .92, .97, 1.0, .99, .95, .88, .80, .76, .72, .68, .65, .62],
}


def asignar_distritos(nodos):
    """Asigna distrito y tipo de zona a cada nodo (centroide mas cercano)."""
    from .datos import proyectar
    lat_d = np.array([d[1] for d in DISTRITOS])
    lon_d = np.array([d[2] for d in DISTRITOS])
    xd, yd = proyectar(lat_d, lon_d)
    dx = nodos.x.to_numpy()[:, None] - xd[None, :]
    dy = nodos.y.to_numpy()[:, None] - yd[None, :]
    j = np.argmin(dx * dx + dy * dy, axis=1)
    out = nodos.copy()
    out["distrito"] = [DISTRITOS[k][0] for k in j]
    out["zona"] = [DISTRITOS[k][3] for k in j]
    rng = np.random.default_rng(2026)
    out["diversidad"] = np.round(rng.uniform(0.8, 1.2, len(out)), 3)
    return out


def demanda_mw(nodos, hora, kw_por_sed=None, crecimiento=0.0):
    """Demanda (MW) de cada nodo a la hora dada. Las fuentes no tienen demanda."""
    kw = kw_por_sed or KW_POR_SED
    base = nodos.zona.map(kw).to_numpy(dtype=float)
    curva = nodos.zona.map(lambda z: CURVAS[z][hora]).to_numpy(dtype=float)
    p = base * curva * nodos.diversidad.to_numpy() * (1 + crecimiento) / 1000.0
    p[nodos.es_fuente.to_numpy()] = 0.0
    return p


def curva_sistema(nodos, kw_por_sed=None, crecimiento=0.0):
    """Demanda total (MW) por hora y por tipo de zona -> para hallar la hora punta."""
    filas = []
    for h in range(24):
        p = demanda_mw(nodos, h, kw_por_sed, crecimiento)
        fila = {"hora": h, "total": p.sum()}
        for z in ZONAS:
            fila[NOMBRE_ZONA[z]] = p[nodos.zona.to_numpy() == z].sum()
        filas.append(fila)
    return pd.DataFrame(filas)
