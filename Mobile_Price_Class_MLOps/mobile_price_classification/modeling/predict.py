"""
Carga de modelo entrenado y generación de predicciones sobre datos nuevos.
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from mobile_price_classification.config import (
    BEST_MODEL_PATH,
    ID_COLUMN,
    SCALER_PATH,
    SUBMISSION_PATH,
    TARGET_COLUMN,
    logger,
)


@dataclass
class Predictor:
    """Carga el modelo y el scaler ya entrenados, y predice sobre datos nuevos.

    Ejemplo
    -------
    >>> predictor = Predictor()
    >>> predictor.load()
    >>> submission = predictor.predict_test(test_df)
    >>> predictor.save_submission(submission)
    """

    model_path: Path = BEST_MODEL_PATH
    scaler_path: Path = SCALER_PATH
    model=None
    scaler=None

    def load(self) -> "Predictor":
        logger.info(f"Cargando modelo desde {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"Cargando scaler desde {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        return self

    def predict_test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Genera predicciones para un DataFrame tipo test.csv (con columna 'id')."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Debes llamar a .load() antes de predecir.")

        X_test = test_df.drop(columns=[ID_COLUMN])
        X_test_scaled = self.scaler.transform(X_test)
        preds = self.model.predict(X_test_scaled)

        submission = test_df[[ID_COLUMN]].copy()
        submission[TARGET_COLUMN] = preds
        return submission

    def predict_single(self, features: dict) -> int:
        """Predice el price_range de un único móvil a partir de un diccionario de features."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Debes llamar a .load() antes de predecir.")

        df = pd.DataFrame([features])
        X_scaled = self.scaler.transform(df)
        return int(self.model.predict(X_scaled)[0])

    def save_submission(self, submission: pd.DataFrame, path: Path = SUBMISSION_PATH) -> None:
        submission.to_csv(path, index=False)
        logger.info(f"Predicciones guardadas en {path}")
