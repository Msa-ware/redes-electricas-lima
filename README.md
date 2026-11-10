# Diseño de la red eléctrica de Lima Metropolitana

Aplicativo del Trabajo Final de **Complejidad Algorítmica (1ACC0184) – UPC 2026-20**, Caso 8: *Diseño de redes eléctricas*.

Diseña la red de media tensión que une las **6 187 subestaciones** de Lima (OpenStreetMap) para que:

1. **minimice el costo del cableado**;
2. **soporte la demanda energética en hora punta**, con restricciones físicas de potencia: capacidad térmica de los conductores, **flujo de potencia DC** y **caída de tensión ≤ 5 %** (NTCSE);
3. **sea robusta ante fallos**, según el criterio N-1 con enlaces de respaldo.

## Cómo ejecutarlo

Requisito: Python 3.10 o superior.

```bash
git clone https://github.com/Msa-ware/redes-electricas-lima.git
cd redes-electricas-lima
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Sin Git, también se puede descargar con **Code → Download ZIP** y descomprimir. En Windows basta con hacer doble clic en `ejecutar_app.bat`.

La aplicación se abre en `http://localhost:8501`. Para las pruebas unitarias: `python -m pytest -q`.

### Publicarlo con un link (opcional)
1. Entrar a <https://share.streamlit.io> e iniciar sesión con GitHub.
2. **Create app** → elegir el repositorio, la rama `main` y el archivo `app.py`.
3. Streamlit Community Cloud instala `requirements.txt` y entrega un link público.

## Qué hace el aplicativo

| Pestaña | Contenido |
|---|---|
| Dataset y demanda | Mapa de subestaciones y líneas de alta tensión, demanda por distrito y curva de carga horaria por tipo de zona |
| Diseño de la red | Configuraciones A / B / C en el mapa: carga de cada línea (%), caída de tensión de cada SED, costo y conductores |
| Fallos (N-1) | Análisis de todas las fallas simples, simulación de la falla de un tramo y cierre automático del respaldo |
| Comparación | Costo, sobrecargas, ΔV, flujo máximo, puentes y robustez de las tres configuraciones |
| Validación | Fuerza bruta y backtracking contra Kruskal/Prim; verificación del flujo DC y del flujo máximo |
| Metodología | Modelo y técnicas utilizadas |

En la barra lateral se pueden cambiar la zona (Lima completa o distritos), la hora, el crecimiento de la demanda, los kW por tipo de zona, la tensión (10 / 22,9 kV), el factor de potencia, el límite de ΔV, la capacidad de los alimentadores y de las SET, y los parámetros de los algoritmos.

## Configuraciones

- **A – MST clásico (Kruskal o Prim):** minimiza solo la longitud, como en el TB1. Con el flujo DC se comprueba que **no es operable**: miles de tramos sobrecargados y caídas de tensión inadmisibles.
- **B – Kruskal capacitado:** algoritmo voraz con UFDS que forma alimentadores radiales sin superar la capacidad del alimentador ni la de su SET. Luego elige el conductor mínimo que soporta el flujo de cada tramo y refuerza conductores (voraz por ΔV/costo) hasta cumplir la caída de tensión.
- **C – Red robusta:** B planificada con reserva, por ejemplo al 50 % de la capacidad, más **enlaces de respaldo normalmente abiertos**, elegidos con un voraz de cobertura sobre el bosque radial.

## Modelo eléctrico

- **Demanda por zonas:** cada SED se asigna al distrito más cercano. El tipo de zona (comercial, residencial alto, medio o popular, industrial) fija los kW típicos y la curva de carga horaria; se multiplica por un factor de diversidad.
- **Capacidad de línea:** `P = √3 · V · I_max · fp`, con un catálogo de cables N2XSY de 70, 120 y 240 mm² y doble terna.
- **Flujo de potencia DC:** `B·θ = P`, `f_uv = (θ_u − θ_v)/x_uv`, resuelto con matrices dispersas (scipy). Cada isla tiene su barra *slack*.
- **Caída de tensión:** `ΔV% = 100·(P·R + Q·X)·L / V²`, acumulada en el sentido del flujo con **ordenamiento topológico** (Kahn).

## Técnicas del curso (implementadas desde cero en `red/algoritmos/`)

| Técnica | Archivo | Uso |
|---|---|---|
| UFDS | `ufds.py` | Kruskal, Kruskal capacitado, islas eléctricas |
| MST: Kruskal y Prim | `mst.py` | Configuración A |
| Voraz (Kruskal capacitado) | `mst.py`, `diseno.py` | Configuraciones B y C, conductores y respaldos |
| BFS / DFS / recorrido de Euler | `recorridos.py` | Árbol radial, alimentadores, subárboles aislados |
| Ordenamiento topológico | `recorridos.py` | Propagación de la caída de tensión |
| Puentes (Tarjan) | `recorridos.py` | Tramos críticos |
| Flujo máximo (Edmonds-Karp y Dinic) | `flujo_maximo.py` | Demanda máxima abastecible |
| Fuerza bruta y Backtracking | `fuerza_bruta.py` | Validación del óptimo |

## Estructura

```
app.py                  interfaz (Streamlit)
red/
  datos.py              carga OSM, nodos, grafo candidato (k-NN + Haversine)
  demanda.py            modelo de demanda por zonas y curvas de carga
  electrico.py          conductores, flujo DC, caída de tensión, evaluación
  diseno.py             configuraciones A, B, C
  fallos.py             análisis N-1, simulación de fallas, puentes, flujo máximo
  mapas.py              mapas y gráficos (plotly)
  contexto.py           arma los datos comunes de cada análisis
  algoritmos/           UFDS, MST, recorridos, flujo máximo, fuerza bruta
data/lima_red_electrica_osm.json   dataset (OpenStreetMap, ODbL)
tests/                  pruebas unitarias (pytest)
```

## Datos
© Colaboradores de OpenStreetMap, licencia ODbL. Descarga con la Overpass API del 04/10/2026; la consulta está en el informe del TB1.
Los parámetros de demanda, conductores y costos son **referenciales** y se pueden editar en la aplicación.
