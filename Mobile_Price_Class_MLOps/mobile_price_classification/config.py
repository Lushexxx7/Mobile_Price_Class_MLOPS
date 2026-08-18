"""
Configuración central del proyecto: rutas de carpetas y parámetros globales.

Toda ruta de datos, modelos o reportes debe importarse desde este módulo
en lugar de escribirse "a mano" en otros archivos, para que el proyecto
sea portable entre máquinas (Windows/Linux/Mac) y entornos (local/CI).
"""
from pathlib import Path
import logging

try:
    # Permite definir variables de entorno en un .env (opcional)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Raíz del proyecto: dos niveles arriba de este archivo
# (<root>/mobile_price_classification/config.py -> <root>)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"      # datos de terceros, sin modificar
INTERIM_DATA_DIR = DATA_DIR / "interim"        # datos transformados a medio camino
PROCESSED_DATA_DIR = DATA_DIR / "processed"    # datos finales listos para modelar
RAW_DATA_DIR = DATA_DIR / "raw"                # datos originales, inmutables

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Modelos (todo lo entrenado se guarda aquí)
# ---------------------------------------------------------------------------
MODELS_DIR = PROJ_ROOT / "models"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODELS_COMPARISON_PATH = MODELS_DIR / "models_comparison.json"
BEST_PARAMS_PATH = MODELS_DIR / "best_model_params.json"

# ---------------------------------------------------------------------------
# Reportes / figuras / predicciones
# ---------------------------------------------------------------------------
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUBMISSION_PATH = REPORTS_DIR / "predicciones_test.csv"

# ---------------------------------------------------------------------------
# Parámetros del problema
# ---------------------------------------------------------------------------
TARGET_COLUMN = "price_range"
ID_COLUMN = "id"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CLASS_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto", 3: "Muy alto"}

# Crear carpetas si no existen (idempotente, seguro en cualquier entorno)
for _dir in (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mobile_price_classification")
