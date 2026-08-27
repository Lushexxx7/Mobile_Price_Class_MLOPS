"""Punto de entrada reutilizable para inferencia."""

import pandas as pd

from src.config import MODEL_PATH
from src.models.pipeline import PipelineTelefonos


def predecir(datos: pd.DataFrame, ruta_modelo=MODEL_PATH):
    """Genera predicciones usando un artefacto previamente entrenado.

    :param datos: filas a predecir
    :param ruta_modelo: artefacto .pkl a usar
    :return: la clase predicha para cada fila
    """
    return PipelineTelefonos.predecir(datos, ruta_modelo)
