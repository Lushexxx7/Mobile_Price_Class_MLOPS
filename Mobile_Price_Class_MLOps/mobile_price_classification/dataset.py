"""
Carga y validación de los datos crudos del proyecto.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from mobile_price_classification.config import (
    TRAIN_FILE,
    TEST_FILE,
    TARGET_COLUMN,
    ID_COLUMN,
    logger,
)


@dataclass
class DataLoader:
    """Encapsula la carga y validación de train.csv y test.csv.

    Ejemplo
    -------
    >>> loader = DataLoader()
    >>> train_df = loader.load_train()
    >>> X, y = loader.get_features_target(train_df)
    """

    train_path: Path = TRAIN_FILE
    test_path: Path = TEST_FILE

    def load_train(self) -> pd.DataFrame:
        """Carga train.csv (incluye la variable objetivo)."""
        logger.info(f"Cargando datos de entrenamiento desde {self.train_path}")
        df = pd.read_csv(self.train_path)
        self._validate_train(df)
        return df

    def load_test(self) -> pd.DataFrame:
        """Carga test.csv (sin variable objetivo, incluye 'id')."""
        logger.info(f"Cargando datos de prueba desde {self.test_path}")
        df = pd.read_csv(self.test_path)
        self._validate_test(df)
        return df

    def get_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separa un DataFrame de entrenamiento en X (features) e y (target)."""
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        return X, y

    # -- validaciones internas -------------------------------------------------
    def _validate_train(self, df: pd.DataFrame) -> None:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"La columna objetivo '{TARGET_COLUMN}' no se encontró en train.csv"
            )
        n_nulls = df.isnull().sum().sum()
        if n_nulls > 0:
            logger.warning(f"Se detectaron {n_nulls} valores nulos en train.csv")
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            logger.warning(f"Se detectaron {n_dupes} filas duplicadas en train.csv")

    def _validate_test(self, df: pd.DataFrame) -> None:
        if ID_COLUMN not in df.columns:
            raise ValueError(f"La columna '{ID_COLUMN}' no se encontró en test.csv")
