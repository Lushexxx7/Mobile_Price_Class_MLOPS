from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def cargar_parametros(ruta: Path = PARAMS_PATH) -> dict:
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
