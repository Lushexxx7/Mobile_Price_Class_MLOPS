"""
Ingeniería de features: split train/validación y escalado de variables.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mobile_price_classification.config import (
    RANDOM_STATE,
    TEST_SIZE,
    SCALER_PATH,
    logger,
)


@dataclass
class FeatureEngineer:
    """Encapsula el split train/validación y el escalado de variables numéricas.

    El mismo scaler ajustado sobre train debe reutilizarse para transformar
    validación y test; esta clase se encarga de eso y de persistirlo en disco.

    Ejemplo
    -------
    >>> fe = FeatureEngineer()
    >>> X_train, X_val, y_train, y_val = fe.split(X, y)
    >>> X_train_scaled = fe.fit_transform(X_train)
    >>> X_val_scaled = fe.transform(X_val)
    >>> fe.save_scaler()
    """

    scaler: StandardScaler = field(default_factory=StandardScaler)
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split estratificado train/validación."""
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X)

    def save_scaler(self, path: Path = SCALER_PATH) -> None:
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler guardado en {path}")

    def load_scaler(self, path: Path = SCALER_PATH) -> "FeatureEngineer":
        self.scaler = joblib.load(path)
        logger.info(f"Scaler cargado desde {path}")
        return self
