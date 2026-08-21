import pandas as pd
import pytest

from src.config import TARGET, TEST_PATH, TRAIN_PATH


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def test():
    return pd.read_csv(TEST_PATH)


def test_train_tiene_la_forma_esperada(train):
    assert train.shape == (2000, 21)
    assert TARGET in train.columns


def test_train_no_tiene_nulos_ni_duplicados(train):
    assert train.isna().sum().sum() == 0
    assert train.duplicated().sum() == 0


def test_clases_balanceadas(train):
    conteo = train[TARGET].value_counts()
    assert set(conteo.index) == {0, 1, 2, 3}
    assert conteo.nunique() == 1  # 500 observaciones por clase


def test_test_tiene_id_y_las_mismas_predictoras(test, train):
    assert "id" in test.columns
    assert TARGET not in test.columns
    predictoras_train = set(train.columns) - {TARGET}
    assert set(test.columns) - {"id"} == predictoras_train


def test_rangos_fisicos_validos(train):
    assert train["ram"].between(256, 3998).all()
    assert train["battery_power"].between(501, 1998).all()
    assert train["n_cores"].between(1, 8).all()