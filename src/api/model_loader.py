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

from src.config import (
    MLFLOW_ALIAS_PRODUCCION,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
)
from src.models.pipeline import PipelineTelefonos

MODEL_NAME = MLFLOW_REGISTERED_MODEL_NAME
MODEL_ALIAS = MLFLOW_ALIAS_PRODUCCION

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


# Origen del modelo cargado en el ultimo `load_model()`. Lo rellena la carga y
# lo lee `get_model_metadata()`, para no volver a preguntarle a MLflow en cada
# peticion: antes cada `GET /` abria una conexion al Registry que, si el
# servidor no estaba, tardaba en fallar y ensuciaba los logs.
_ORIGEN: dict[str, str] = {"origen": "sin-modelo", "version": "N/A", "run_id": "N/A"}


def load_model() -> ClassifierMixin | None:
    """Carga el modelo con el alias de produccion; si falla, usa el .pkl local."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        modelo = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except Exception as error:  # noqa: BLE001 - degradamos sin tumbar la API
        print(f"[model_loader] No se pudo cargar desde MLflow Registry: {error}")
    else:
        _ORIGEN.update({"origen": "mlflow-registry", **_metadata_registry()})
        return modelo

    try:
        artefacto = PipelineTelefonos.cargar(MODEL_PATH)
    except Exception as error:  # noqa: BLE001
        print(f"[model_loader] No se pudo cargar el .pkl local: {error}")
        _ORIGEN.update({"origen": "sin-modelo", "version": "N/A", "run_id": "N/A"})
        return None

    _ORIGEN.update({"origen": "pkl-local", "version": "local-pkl", "run_id": "N/A"})
    return artefacto["modelo"]


def _metadata_registry() -> dict[str, str]:
    """Version y run_id de la version con el alias de produccion."""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        return {"version": str(version.version), "run_id": str(version.run_id)}
    except Exception as error:  # noqa: BLE001
        print(f"[model_loader] No se pudo leer metadata de MLflow: {error}")
        return {"version": "desconocida", "run_id": "N/A"}


def get_model_metadata() -> dict[str, str]:
    """Metadatos del modelo que hay cargado ahora mismo."""
    return dict(_ORIGEN)
