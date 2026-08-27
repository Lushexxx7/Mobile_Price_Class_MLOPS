"""Rutas del proyecto y parametros leidos de params.yaml."""

import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def cargar_parametros(ruta: Path = PARAMS_PATH) -> dict:
    """Lee params.yaml y lo devuelve como dict.

    :param ruta: fichero YAML a leer
    :return: los parametros, o un dict vacio si el fichero esta vacio
    """
    with ruta.open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo) or {}


PARAMS = cargar_parametros()

RANDOM_STATE = int(PARAMS["data"]["random_state"])
TEST_SIZE = float(PARAMS["data"]["test_size"])
TARGET = str(PARAMS["data"]["target"])

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_final.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
VALIDATION_PATH = PROJECT_ROOT / "reports" / "data_validation.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predicciones.csv"


# --------------------------------------------------------------------- MLflow
_MLFLOW = PARAMS.get("mlflow", {})

MLFLOW_EXPERIMENT_NAME = str(_MLFLOW.get("experiment_name", "telefonos_price_classification"))
MLFLOW_REGISTERED_MODEL_NAME = str(_MLFLOW.get("registered_model_name", "TelefonosPriceClassifier"))

# El alias que sirve la API y el que produce la busqueda de hiperparametros.
# Estaban escritos a mano en main.py, tune_hyperparameters.py y model_loader.py;
# centralizarlos evita que se desincronicen y que la API pida un alias que el
# entrenamiento nunca asigno.
MLFLOW_ALIAS_PRODUCCION = str(_MLFLOW.get("alias_produccion", "champion"))
MLFLOW_ALIAS_CHALLENGER = str(_MLFLOW.get("alias_challenger", "challenger"))


def _resolver_tracking_uri() -> str:
    """URI de MLflow, con el entorno por encima de params.yaml.

    :return: la URI ya resuelta, con las rutas sqlite ancladas a la raiz

    Dos motivos para no leer `params.yaml` a secas:

    1. `MLFLOW_TRACKING_URI` del entorno debe ganar. Es lo que permite apuntar
       a un servidor de tracking (por ejemplo el del contenedor) sin editar el
       repo ni ensuciar el diff de params.yaml.
    2. `sqlite:///mlflow.db` es una ruta relativa y SQLite la resuelve contra
       el directorio de trabajo. Ejecutar `python main.py` desde la raiz y
       `pytest` desde otra carpeta creaba dos bases distintas, con la mitad de
       los experimentos en cada una. Se ancla a la raiz del proyecto.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI") or str(
        _MLFLOW.get("tracking_uri", "sqlite:///mlflow.db")
    )

    prefijo = "sqlite:///"
    if uri.startswith(prefijo):
        ruta = Path(uri[len(prefijo) :])
        if not ruta.is_absolute():
            ruta = PROJECT_ROOT / ruta
        return f"{prefijo}{ruta.as_posix()}"

    return uri


MLFLOW_TRACKING_URI = _resolver_tracking_uri()
