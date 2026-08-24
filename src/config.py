from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "price_range"

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_final.pkl"
