from pathlib import Path

from scripts.tune_hyperparameters import candidatos
from src.models.tracking import RastreadorMLflow


def test_hash_dvc(tmp_path: Path):
    puntero = tmp_path / "datos.csv.dvc"
    puntero.write_text("outs:\n  - md5: abc123\n", encoding="utf-8")

    assert RastreadorMLflow._hash_dvc(puntero) == "abc123"


def test_busqueda_genera_dos_candidatos_por_modelo():
    lista = list(candidatos())

    assert len(lista) == 6
    assert {modelo.nombre for modelo, _ in lista} == {
        "Regresión Logística",
        "Random Forest",
        "SVM",
    }
    cantidades = {}
    for modelo, _ in lista:
        cantidades[modelo.nombre] = cantidades.get(modelo.nombre, 0) + 1
    assert set(cantidades.values()) == {2}
