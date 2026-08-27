"""Fixtures compartidos que evitan conflictos de permisos en Windows."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.api import security


@pytest.fixture
def tmp_path():
    """Crea un directorio temporal aislado sin usar la caché de pytest."""
    with TemporaryDirectory(prefix="telefonos_mlops_") as directory:
        yield Path(directory)


@pytest.fixture(autouse=True)
def api_key_desactivada(monkeypatch):
    """Corre la suite siempre sin autenticacion, ignore lo que haya en el entorno.

    `security.API_KEY` se resuelve al importar el modulo, asi que una terminal
    con `$env:API_KEY` puesta (tipico despues de probar el 401 a mano) hacia
    fallar seis tests de la API con 401 en vez del codigo esperado. El fallo no
    tenia nada que ver con el codigo bajo prueba, que es lo que lo hacia caro
    de diagnosticar. Los tests que quieran probar la autenticacion la vuelven a
    activar con monkeypatch por su cuenta.
    """
    monkeypatch.setattr(security, "API_KEY", "")
