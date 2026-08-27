"""Funciones para construir la matriz de características."""

import pandas as pd

from src.config import TARGET


def construir_caracteristicas(datos: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Devuelve las variables predictoras sin modificar el DataFrame original.

    :param datos: dataset, con o sin la columna objetivo
    :param target: nombre de la columna objetivo a descartar
    :return: una copia sin esa columna; si no estaba, una copia tal cual

    >>> import pandas as pd
    >>> construir_caracteristicas(pd.DataFrame({"ram": [1], "price_range": [0]}))
       ram
    0    1
    """
    if target not in datos.columns:
        return datos.copy()
    return datos.drop(columns=[target]).copy()
