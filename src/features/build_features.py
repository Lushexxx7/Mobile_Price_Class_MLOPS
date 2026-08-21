"""Funciones para construir la matriz de características."""

import pandas as pd

from src.config import TARGET


def construir_caracteristicas(datos: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Devuelve las variables predictoras sin modificar el DataFrame original."""
    if target not in datos.columns:
        return datos.copy()
    return datos.drop(columns=[target]).copy()
