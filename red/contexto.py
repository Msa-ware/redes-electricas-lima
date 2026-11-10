"""Arma el 'contexto' comun que usan todos los disenos: nodos, demanda,
fuentes, capacidades y grafo candidato."""
import numpy as np

from .datos import aristas_candidatas, cargar_nodos, cargar_osm
from .demanda import asignar_distritos, demanda_mw
from .electrico import ParamElectricos, capacidad_fuentes


def cargar_base(ruta_json):
    osm = cargar_osm(ruta_json)
    nodos = asignar_distritos(cargar_nodos(osm))
    return osm, nodos


def construir_contexto(nodos, hora, kw_por_sed, crecimiento, p: ParamElectricos, k=5, distritos=None):
    """Si se pasan `distritos`, el analisis se restringe a esa zona (se re-indexan los nodos)."""
    if distritos:
        nodos = nodos[nodos.distrito.isin(distritos)].copy()
    nodos = nodos.reset_index(drop=True)
    nodos["id"] = range(len(nodos))
    cand = aristas_candidatas(nodos, k=k)
    dem = demanda_mw(nodos, hora, kw_por_sed, crecimiento)
    return {
        "n": len(nodos), "nodos": nodos, "cand": cand, "demanda": dem,
        "es_fuente": nodos.es_fuente.to_numpy(), "cap_fuente": capacidad_fuentes(nodos, p),
        "lat": nodos.lat.to_numpy(), "lon": nodos.lon.to_numpy(), "p": p, "hora": hora,
    }
