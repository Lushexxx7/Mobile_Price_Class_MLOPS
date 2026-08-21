import pandas as pd
import pytest

from src.models.train import (
    ModeloRandomForest,
    ModeloRegresionLogistica,
    ModeloSVM,
)


@pytest.fixture
def datos_modelo():
    x = pd.DataFrame(
        {
            "ram": [512, 600, 1000, 1200, 2000, 2200, 3000, 3200] * 3,
            "battery_power": [500, 600, 800, 900, 1100, 1200, 1500, 1600] * 3,
        }
    )
    y = pd.Series([0, 0, 1, 1, 2, 2, 3, 3] * 3)
    return x, y


@pytest.mark.parametrize(
    "modelo",
    [
        ModeloRegresionLogistica(),
        ModeloRandomForest(),
        ModeloSVM(),
    ],
)
def test_modelos_entrenan_y_generan_clases_validas(modelo, datos_modelo):
    x, y = datos_modelo

    predicciones = modelo.entrenar(x, y).predecir(x)

    assert len(predicciones) == len(y)
    assert set(predicciones).issubset({0, 1, 2, 3})


def test_random_forest_expone_importancias(datos_modelo):
    x, y = datos_modelo
    modelo = ModeloRandomForest().entrenar(x, y)

    importancias = modelo.obtener_importancias()

    assert len(importancias) == x.shape[1]
    assert importancias.sum() == pytest.approx(1.0)

