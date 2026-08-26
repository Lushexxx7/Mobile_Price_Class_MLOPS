import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARAMS_PATH = PROJECT_ROOT / "params.yaml"
with open(PARAMS_PATH, encoding="utf-8") as archivo:
    PARAMS = yaml.safe_load(archivo)

RANDOM_STATE = PARAMS["split"]["random_state"]
TEST_SIZE = PARAMS["split"]["test_size"]
TARGET = PARAMS["split"]["target"]
METRICA_SELECCION = PARAMS["modelos"]["metrica_seleccion"]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_final.pkl"
METRICS_PATH = PROJECT_ROOT / "metrics" / "eval.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predicciones.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"
COMPARACION_PLOT_PATH = PLOTS_DIR / "comparacion_modelos.csv"
CONFUSION_PLOT_PATH = PLOTS_DIR / "matriz_confusion.csv"
IMPORTANCIAS_PLOT_PATH = PLOTS_DIR / "importancias.csv"

# --- MLflow ---
MLFLOW = PARAMS.get("mlflow", {})
MLFLOW_ACTIVO = bool(MLFLOW.get("activo", False))
MLFLOW_EXPERIMENTO = MLFLOW.get("experimento", "Mobile_Price_Classification")
MLFLOW_REGISTRO_MODELO = MLFLOW.get("registro_modelo", "Mobile_Price_Classifier")
MLFLOW_ALIAS_PRODUCCION = MLFLOW.get("alias_produccion", "champion")

# La URI relativa se ancla a la raíz del proyecto para que funcione igual
# ejecutando `python main.py`, `dvc repro` o pytest desde cualquier carpeta.
# MLFLOW_TRACKING_URI (entorno) tiene prioridad sobre params.yaml: permite
# apuntar a un servidor de tracking remoto sin editar el repo.
_URI = os.getenv("MLFLOW_TRACKING_URI") or MLFLOW.get("tracking_uri", "sqlite:///mlflow.db")
if _URI.startswith("sqlite:///") and not Path(_URI.replace("sqlite:///", "")).is_absolute():
    MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
else:
    MLFLOW_TRACKING_URI = _URI