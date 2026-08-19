from pathlib import Path

import pandas as pd


class CargadorDatos:

    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta)
        self._datos: pd.DataFrame | None = None

    def cargar(self) -> pd.DataFrame:
        if not self.ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo: {self.ruta}")

        self._datos = pd.read_csv(self.ruta)
        return self._datos

    def obtener_dimensiones(self) -> tuple[int, int]:
        return self._obtener_datos().shape

    def obtener_nulos(self) -> pd.Series:
        return self._obtener_datos().isnull().sum()

    def obtener_duplicados(self) -> int:
        return int(self._obtener_datos().duplicated().sum())

    def _obtener_datos(self) -> pd.DataFrame:
        if self._datos is None:
            raise RuntimeError("Primero debes ejecutar cargar()")
        return self._datos
