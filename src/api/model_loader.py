"""Carga del modelo campeón para servirlo por HTTP.

Dos niveles:
  1. MLflow Model Registry (`models:/<nombre>@champion`) -> fuente de verdad.
  2. Fallback al artefacto local de DVC (`models/modelo_final.pkl`).

Nunca lanza excepción al arrancar: si falla, la app queda viva y /health
responde 503. Así el contenedor no entra en CrashLoopBackOff.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_ALIAS_PRODUCCION,
    MLFLOW_REGISTRO_MODELO,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
)
from src.models.pipeline import PipelineTelefonos

MODEL_NAME = MLFLOW_REGISTRO_MODELO
ALIAS = MLFLOW_ALIAS_PRODUCCION

_SIN_DATOS = {"version": "desconocida", "run_id": "desconocido", "origen": "ninguno"}

_metadatos: dict[str, str] = dict(_SIN_DATOS)
_columnas: list[str] = []


def _desde_registry() -> tuple[Any, dict[str, str], list[str]]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    uri = f"models:/{MODEL_NAME}@{ALIAS}"
    version = cliente.get_model_version_by_alias(name=MODEL_NAME, alias=ALIAS)
    modelo = mlflow.sklearn.load_model(uri)

    firma = mlflow.models.get_model_info(uri).signature
    if firma is None:
        raise ValueError("El modelo registrado no tiene firma; no sé el orden de columnas.")

    metadatos = {
        "version": str(version.version),
        "run_id": version.run_id,
        "origen": f"registry@{ALIAS}",
    }
    return modelo, metadatos, list(firma.inputs.input_names())


def _desde_pkl() -> tuple[Any, dict[str, str], list[str]]:
    artefacto = PipelineTelefonos.cargar(MODEL_PATH)
    metadatos = {
        "version": "local",
        "run_id": str(artefacto.get("nombre", "desconocido")),
        "origen": "artefacto_local",
    }
    return artefacto["modelo"], metadatos, list(artefacto["columnas"])


def cargar_modelo() -> Any | None:
    """Devuelve el estimador listo para predecir, o None si no hay ninguno."""
    global _metadatos, _columnas

    for cargar, etiqueta in ((_desde_registry, "Registry"), (_desde_pkl, "artefacto local")):
        try:
            modelo, metadatos, columnas = cargar()
        except Exception as error:
            print(f"[API] No pude cargar desde {etiqueta}: {error}")
            continue
        _metadatos, _columnas = metadatos, columnas
        return modelo

    _metadatos, _columnas = dict(_SIN_DATOS), []
    return None


def obtener_metadatos() -> dict[str, str]:
    return dict(_metadatos)


def obtener_columnas() -> list[str]:
    return list(_columnas)