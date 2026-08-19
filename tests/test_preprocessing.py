import pandas as pd
import pytest

from src.preprocessing import PreprocesadorTelefonos


@pytest.fixture
def datos_clasificacion():
    return pd.DataFrame(
        {
            "ram": range(40),
            "battery_power": range(100, 140),
            "price_range": [0, 1, 2, 3] * 10,
        }
    )


def test_separar_variables(datos_clasificacion):
    x, y = PreprocesadorTelefonos().separar_variables(datos_clasificacion)

    assert "price_range" not in x.columns
    assert y.name == "price_range"


def test_division_es_estratificada_y_reproducible(datos_clasificacion):
    preprocesador = PreprocesadorTelefonos(test_size=0.20, random_state=42)
    x, y = preprocesador.separar_variables(datos_clasificacion)
    primera = preprocesador.dividir_datos(x, y)
    segunda = preprocesador.dividir_datos(x, y)

    assert len(primera[1]) == 8
    assert set(primera[3]) == {0, 1, 2, 3}
    pd.testing.assert_frame_equal(primera[0], segunda[0])


def test_rechaza_target_inexistente(datos_clasificacion):
    with pytest.raises(ValueError, match="objetivo"):
        PreprocesadorTelefonos(target="inexistente").separar_variables(
            datos_clasificacion
        )


def test_preparar_inferencia_descarta_id_y_ordena_columnas():
    datos = pd.DataFrame({"id": [1], "ram": [2048], "blue": [1]})

    resultado = PreprocesadorTelefonos.preparar_inferencia(
        datos, ["blue", "ram"]
    )

    assert resultado.columns.tolist() == ["blue", "ram"]
