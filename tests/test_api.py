"""Pruebas de la capa de servicio. Requieren un modelo entrenado."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

TELEFONO = {
    "battery_power": 1043, "blue": 1, "clock_speed": 1.8, "dual_sim": 1,
    "fc": 14, "four_g": 0, "int_memory": 5, "m_dep": 0.1, "mobile_wt": 193,
    "n_cores": 3, "pc": 16, "px_height": 226, "px_width": 1412, "ram": 3476,
    "sc_h": 12, "sc_w": 7, "talk_time": 2, "three_g": 0, "touch_screen": 1,
    "wifi": 0,
}


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as clientep:   # el `with` dispara el lifespan
        yield clientep


def test_raiz_responde(cliente):
    assert cliente.get("/").status_code == 200


def test_prediccion_devuelve_clase_valida(cliente):
    respuesta = cliente.post("/predict", json={"data": [TELEFONO]})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["total"] == 1
    assert cuerpo["results"][0]["price_range"] in {0, 1, 2, 3}


def test_lote_conserva_el_orden(cliente):
    respuesta = cliente.post("/predict", json={"data": [TELEFONO, TELEFONO]})
    assert [r["index"] for r in respuesta.json()["results"]] == [0, 1]


def test_campo_desconocido_es_rechazado(cliente):
    respuesta = cliente.post("/predict", json={"data": [{**TELEFONO, "price_range": 3}]})
    assert respuesta.status_code == 422


def test_flag_fuera_de_rango_es_rechazado(cliente):
    respuesta = cliente.post("/predict", json={"data": [{**TELEFONO, "blue": 9}]})
    assert respuesta.status_code == 422