"""Pruebas de la API de inferencia.

No dependen de MLflow ni del .pkl: el modelo se inyecta en `_estado` con un
doble. Asi la suite corre igual en una maquina recien clonada, sin `dvc pull`
y sin haber entrenado nada.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app as modulo_app
from src.api.model_loader import FEATURE_COLUMNS


class ModeloFalso:
    """Clasificador minimo con la interfaz que consume la API."""

    classes_ = np.array([0, 1, 2, 3])

    def predict(self, x):
        return np.array([2] * len(x))

    def predict_proba(self, x):
        return np.array([[0.0, 0.1, 0.7, 0.2]] * len(x))


class ModeloSinProba:
    """Algunos estimadores (SVC sin `probability=True`) no traen predict_proba.

    No hereda de ModeloFalso a proposito: la API decide con `hasattr`, asi que
    el atributo tiene que faltar de verdad, no valer None.
    """

    classes_ = np.array([0, 1, 2, 3])

    def predict(self, x):
        return np.array([2] * len(x))


def payload(n: int = 1) -> dict:
    fila = {columna: 1.0 for columna in FEATURE_COLUMNS}
    return {"data": [fila for _ in range(n)]}


@pytest.fixture
def cliente_con_modelo():
    """Cliente con un modelo ya cargado, saltandose el lifespan real."""
    modulo_app._estado["modelo"] = ModeloFalso()
    with TestClient(modulo_app.app) as cliente:
        # TestClient ejecuta el lifespan, que llama a load_model() y deja el
        # estado a None si no hay modelo real. Se reinyecta despues.
        modulo_app._estado["modelo"] = ModeloFalso()
        yield cliente
    modulo_app._estado["modelo"] = None


@pytest.fixture
def cliente_sin_modelo():
    with TestClient(modulo_app.app) as cliente:
        modulo_app._estado["modelo"] = None
        yield cliente


def test_raiz_responde_aunque_no_haya_modelo(cliente_sin_modelo):
    respuesta = cliente_sin_modelo.get("/")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "online"


def test_health_devuelve_503_sin_modelo(cliente_sin_modelo):
    """Es la condicion que usa el HEALTHCHECK del contenedor."""
    respuesta = cliente_sin_modelo.get("/health")

    assert respuesta.status_code == 503


def test_health_devuelve_200_con_modelo(cliente_con_modelo):
    respuesta = cliente_con_modelo.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ready"


def test_predict_devuelve_503_sin_modelo(cliente_sin_modelo):
    """No 500: el servicio esta sano, lo que falta es el modelo."""
    respuesta = cliente_sin_modelo.post("/predict", json=payload())

    assert respuesta.status_code == 503


def test_predict_lote_completo(cliente_con_modelo):
    respuesta = cliente_con_modelo.post("/predict", json=payload(3))

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["total"] == 3
    assert len(cuerpo["results"]) == 3

    primero = cuerpo["results"][0]
    assert primero["price_range"] == 2
    assert primero["etiqueta"] == "Precio Alto"
    assert primero["confianza"] == pytest.approx(70.0)


def test_predict_conserva_una_probabilidad_de_cero(cliente_con_modelo):
    """La clase 0 tiene probabilidad 0.0 y debe aparecer, no desaparecer."""
    respuesta = cliente_con_modelo.post("/predict", json=payload())

    probabilidades = respuesta.json()["results"][0]["probabilidades"]
    assert probabilidades["Precio Bajo"] == 0.0


def test_predict_rechaza_lote_vacio(cliente_con_modelo):
    respuesta = cliente_con_modelo.post("/predict", json={"data": []})

    assert respuesta.status_code == 422


def test_predict_rechaza_columna_faltante(cliente_con_modelo):
    incompleto = payload()
    del incompleto["data"][0]["ram"]

    respuesta = cliente_con_modelo.post("/predict", json=incompleto)

    assert respuesta.status_code == 422


def test_predict_sin_predict_proba_no_reporta_confianza(cliente_con_modelo):
    modulo_app._estado["modelo"] = ModeloSinProba()

    respuesta = cliente_con_modelo.post("/predict", json=payload())

    assert respuesta.status_code == 200
    resultado = respuesta.json()["results"][0]
    assert resultado["confianza"] is None
    assert resultado["probabilidades"] is None
