"""Separacion de variables y division train/validacion del dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET, TEST_SIZE


class PreprocesadorTelefonos:
    """Prepara el dataset para entrenar y para predecir.

    Los valores por defecto salen de `params.yaml`, asi que cambiar el split o
    la semilla no obliga a tocar codigo.

    :param target: nombre de la variable objetivo
    :param test_size: proporcion que se reserva para validacion, entre 0 y 1
    :param random_state: semilla, para que la division sea reproducible
    :ivar target: la variable objetivo configurada
    :ivar test_size: la proporcion de validacion
    :ivar random_state: la semilla en uso
    """

    def __init__(
        self,
        target: str = TARGET,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        """:param target: variable objetivo
        :param test_size: proporcion de validacion, tiene que estar entre 0 y 1
        :param random_state: semilla de la division
        :raises ValueError: si test_size cae fuera de (0, 1)
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def separar_variables(self, datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Parte el DataFrame en predictoras y objetivo.

        :param datos: dataset completo, con la columna objetivo incluida
        :return: la tupla (X, y)
        :raises ValueError: si el dataset no trae la columna objetivo
        """
        if self.target not in datos.columns:
            raise ValueError(f"No existe la variable objetivo: {self.target}")
        return datos.drop(columns=[self.target]), datos[self.target]

    def dividir_datos(self, x: pd.DataFrame, y: pd.Series) -> tuple:
        """Divide en train y validacion, estratificando por la clase.

        Estratifica a proposito: las cuatro clases estan balanceadas en el
        dataset y una division al azar podria desbalancearlas.

        :param x: matriz de predictoras
        :param y: vector objetivo
        :return: la tupla (x_train, x_val, y_train, y_val)
        :raises ValueError: si x e y no tienen el mismo numero de filas
        """
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
    def preparar_inferencia(datos: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
        """Deja solo las columnas del modelo, en el orden con que se entreno.

        Descarta de paso las columnas que sobran, como el `id` de test.csv.

        :param datos: filas a predecir, pueden traer columnas de mas
        :param columnas: columnas que el modelo espera, en orden
        :return: un DataFrame nuevo con exactamente esas columnas
        :raises ValueError: si falta alguna de las columnas pedidas

        >>> import pandas as pd
        >>> datos = pd.DataFrame({"id": [1], "ram": [2048], "blue": [1]})
        >>> PreprocesadorTelefonos.preparar_inferencia(datos, ["blue", "ram"])
           blue   ram
        0     1  2048
        """
        faltantes = sorted(set(columnas) - set(datos.columns))
        if faltantes:
            raise ValueError(f"Faltan columnas para predecir: {faltantes}")
        return datos.loc[:, columnas].copy()
