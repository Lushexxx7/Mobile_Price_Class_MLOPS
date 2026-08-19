import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET, TEST_SIZE


class PreprocesadorTelefonos:
    def __init__(
        self,
        target: str = TARGET,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def separar_variables(
        self, datos: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        if self.target not in datos.columns:
            raise ValueError(f"No existe la variable objetivo: {self.target}")
        return datos.drop(columns=[self.target]), datos[self.target]

    def dividir_datos(self, x: pd.DataFrame, y: pd.Series) -> tuple:
        if len(x) != len(y):
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        return train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    @staticmethod
    def preparar_inferencia(
        datos: pd.DataFrame, columnas: list[str]
    ) -> pd.DataFrame:
        faltantes = sorted(set(columnas) - set(datos.columns))
        if faltantes:
            raise ValueError(f"Faltan columnas para predecir: {faltantes}")
        return datos.loc[:, columnas].copy()
