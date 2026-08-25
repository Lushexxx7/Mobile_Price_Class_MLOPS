"""Carga centralizada del modelo para la API.

Prioriza el Model Registry de MLflow, pidiendo siempre el alias
``champion`` (asignado en `main.py` vía `RastreadorMLflow.registrar_y_asignar_alias`)
en vez de un número de versión fijo. Así la API siempre sirve la última
versión que el equipo haya entrenado y promovido, sin tocar código.

Si el Registry no está disponible (por ejemplo, corriendo la API sin
`mlflow.db`, o sin conexión al tracking server), cae de vuelta al `.pkl`
clásico generado por `PipelineTelefonos.guardar`, para que la API nunca
quede completamente inutilizable.
"""

from sklearn.base import ClassifierMixin

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.config import MODEL_PATH, PARAMS
from src.models.pipeline import PipelineTelefonos

MLFLOW_CONFIG = PARAMS["mlflow"]
MODEL_NAME = str(MLFLOW_CONFIG["registered_model_name"])
MODEL_ALIAS = "champion"

# Columnas esperadas por el modelo, en el orden con el que se entrenó
# (ver data/raw/train.csv; price_range es la variable objetivo, no entra aquí).
FEATURE_COLUMNS = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]


def load_model() -> ClassifierMixin | None:
    """Carga el modelo con alias @champion; si falla, usa el .pkl local."""
    try:
        mlflow.set_tracking_uri(str(MLFLOW_CONFIG["tracking_uri"]))
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except Exception as error:  # noqa: BLE001 - degradamos sin tumbar la API
        print(f"[model_loader] No se pudo cargar desde MLflow Registry: {error}")

    try:
        artefacto = PipelineTelefonos.cargar(MODEL_PATH)
        return artefacto["modelo"]
    except Exception as error:  # noqa: BLE001
        print(f"[model_loader] No se pudo cargar el .pkl local: {error}")
        return None


def get_model_metadata() -> dict[str, str]:
    """Devuelve versión y run_id de la versión con alias @champion."""
    try:
        mlflow.set_tracking_uri(str(MLFLOW_CONFIG["tracking_uri"]))
        client = MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        return {"version": version.version, "run_id": version.run_id}
    except Exception as error:  # noqa: BLE001
        print(f"[model_loader] No se pudo leer metadata de MLflow: {error}")

    return {"version": "local-pkl", "run_id": "N/A"}
