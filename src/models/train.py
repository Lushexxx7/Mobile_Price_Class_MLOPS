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

_LOGISTICA = PARAMS["modelos"]["logistica"]
_RANDOM_FOREST = PARAMS["modelos"]["random_forest"]
_SVM = PARAMS["modelos"]["svm"]

class ModeloClasificacion(ABC):
    """Clase base que encapsula un estimador de clasificación."""

    def __init__(self, nombre: str, modelo: ClassifierMixin):
        self.nombre = nombre
        self._modelo = modelo

    def entrenar(self, x: pd.DataFrame, y: pd.Series) -> "ModeloClasificacion":
        self._modelo.fit(x, y)
        return self

    def predecir(self, x: pd.DataFrame) -> np.ndarray:
        return self._modelo.predict(x)

    @property
    def modelo(self) -> ClassifierMixin:
        return self._modelo


class ModeloRegresionLogistica(ModeloClasificacion):

    def __init__(self, random_state: int = RANDOM_STATE):
        modelo = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=_LOGISTICA["max_iter"], random_state=random_state),
                ),
            ]
        )
        super().__init__("Regresión Logística", modelo)


class ModeloRandomForest(ModeloClasificacion):

    def __init__(self, random_state: int = RANDOM_STATE):
        modelo = RandomForestClassifier(
            n_estimators=_RANDOM_FOREST["n_estimators"],
            random_state=random_state,
            n_jobs=-1,
        )
        super().__init__("Random Forest", modelo)

    def obtener_importancias(self) -> np.ndarray:
        """Devuelve la importancia calculada para cada característica."""
        if not hasattr(self._modelo, "feature_importances_"):
            raise RuntimeError("El modelo debe entrenarse primero.")
        return self._modelo.feature_importances_


class ModeloSVM(ModeloClasificacion):

    def __init__(self, random_state: int = RANDOM_STATE):
        modelo = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel=_SVM["kernel"],
                        C=_SVM["C"],
                        gamma=_SVM["gamma"],
                        random_state=random_state,
                    ),
                ),
            ]
        )
        super().__init__("SVM", modelo)


def crear_modelos(random_state: int = RANDOM_STATE) -> list[ModeloClasificacion]:
    return [
        ModeloRegresionLogistica(random_state),
        ModeloRandomForest(random_state),
        ModeloSVM(random_state),
    ]
