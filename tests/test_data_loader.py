import pandas as pd
import pytest

from src.data.load_data import CargadorDatos
from src.config import TARGET, TRAIN_PATH


def test_cargar_csv(tmp_path):
    ruta = tmp_path / "datos.csv"
    pd.DataFrame({"ram": [1024, 2048], "price_range": [0, 1]}).to_csv(
        ruta, index=False
    )

    cargador = CargadorDatos(ruta)
    datos = cargador.cargar()

    assert cargador.obtener_dimensiones() == (2, 2)
    assert "price_range" in datos.columns
    assert cargador.obtener_nulos().sum() == 0
    assert cargador.obtener_duplicados() == 0


def test_requiere_cargar_datos_antes_de_consultarlos(tmp_path):
    cargador = CargadorDatos(tmp_path / "datos.csv")

    with pytest.raises(RuntimeError, match="cargar"):
        cargador.obtener_dimensiones()


def test_archivo_inexistente(tmp_path):
    cargador = CargadorDatos(tmp_path / "inexistente.csv")

    with pytest.raises(FileNotFoundError):
        cargador.cargar()


def test_dataset_de_entrenamiento_del_proyecto():
    datos = CargadorDatos(TRAIN_PATH).cargar()

    assert TARGET in datos.columns
    assert datos.drop(columns=[TARGET]).shape[1] == 20
    assert set(datos[TARGET].unique()) == {0, 1, 2, 3}

