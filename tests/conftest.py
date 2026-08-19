"""Fixtures compartidos que evitan conflictos de permisos en Windows."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def tmp_path():
    """Crea un directorio temporal aislado sin usar la caché de pytest."""
    with TemporaryDirectory(prefix="telefonos_mlops_") as directory:
        yield Path(directory)
