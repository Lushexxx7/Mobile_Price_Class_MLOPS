"""Los tres clasificadores del proyecto, detras de una interfaz comun."""

from abc import ABC

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PARAMS, RANDOM_STATE


class ModeloClasificacion(ABC):
    """Clase base que encapsula un estimador de clasificación.

    Las hijas solo tienen que construir su estimador y pasarlo aqui. Gracias a
    eso el pipeline los trata a todos igual y anadir un modelo nuevo no obliga
    a tocar nada mas.

    :param nombre: nombre legible del modelo, el que sale en los reportes
    :param modelo: estimador de scikit-learn ya configurado
    :ivar nombre: el nombre anterior
    :ivar _modelo: el estimador que hace el trabajo
    """

    def __init__(self, nombre: str, modelo: ClassifierMixin):
        """:param nombre: nombre legible del modelo
        :param modelo: estimador de scikit-learn ya configurado
        """
        self.nombre = nombre
        self._modelo = modelo

    def entrenar(self, x: pd.DataFrame, y: pd.Series) -> "ModeloClasificacion":
        """Ajusta el estimador.

        :param x: matriz de predictoras
        :param y: vector objetivo
        :return: la propia instancia, para poder encadenar entrenar().predecir()
        """
        self._modelo.fit(x, y)
        return self

    def predecir(self, x: pd.DataFrame) -> np.ndarray:
        """:param x: filas a predecir
        :return: la clase predicha para cada fila
        """
        return self._modelo.predict(x)

    @property
    def modelo(self) -> ClassifierMixin:
        """:return: el estimador de scikit-learn que hay debajo"""
        return self._modelo


class ModeloRegresionLogistica(ModeloClasificacion):
    """Linea base: escalado + regresion logistica.

    El StandardScaler va dentro del Pipeline y no fuera para que se ajuste solo
    con train y no filtre informacion de validacion.

    :param random_state: semilla del estimador
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        """:param random_state: semilla del estimador"""
        parametros = PARAMS["logistic_regression"]
        modelo = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=int(parametros["max_iter"]), random_state=random_state
                    ),
                ),
            ]
        )
        super().__init__("Regresión Logística", modelo)


class ModeloRandomForest(ModeloClasificacion):
    """Ensamble de arboles. No necesita escalado y da importancias.

    :param random_state: semilla del estimador
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        """:param random_state: semilla del estimador"""
        parametros = PARAMS["random_forest"]
        modelo = RandomForestClassifier(
            n_estimators=int(parametros["n_estimators"]),
            max_depth=parametros["max_depth"],
            random_state=random_state,
            n_jobs=int(parametros["n_jobs"]),
        )
        super().__init__("Random Forest", modelo)

    def obtener_importancias(self) -> np.ndarray:
        """Devuelve la importancia calculada para cada característica.

        :return: un peso por columna, en el orden de entrenamiento
        :raises RuntimeError: si el modelo todavia no se ha entrenado
        """
        if not hasattr(self._modelo, "feature_importances_"):
            raise RuntimeError("El modelo debe entrenarse primero.")
        return self._modelo.feature_importances_


class ModeloSVM(ModeloClasificacion):
    """Alternativa no lineal: escalado + SVC con kernel RBF.

    El escalado aqui no es opcional, un SVC sin escalar da resultados malos.

    :param random_state: semilla del estimador
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        """:param random_state: semilla del estimador"""
        parametros = PARAMS["svm"]
        modelo = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel=str(parametros["kernel"]),
                        C=float(parametros["C"]),
                        gamma=str(parametros["gamma"]),
                        random_state=random_state,
                    ),
                ),
            ]
        )
        super().__init__("SVM", modelo)


def crear_modelos(random_state: int = RANDOM_STATE) -> list[ModeloClasificacion]:
    """Crea los tres modelos que compara el pipeline.

    :param random_state: semilla que comparten los tres
    :return: la lista de modelos sin entrenar
    """
    return [
        ModeloRegresionLogistica(random_state),
        ModeloRandomForest(random_state),
        ModeloSVM(random_state),
    ]
