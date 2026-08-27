"""Carga de los CSV del proyecto y primeras comprobaciones de calidad."""

from pathlib import Path

import pandas as pd


class CargadorDatos:
    """Lee un CSV y responde preguntas basicas sobre su calidad.

    Guarda el DataFrame despues de `cargar()` para no releer el fichero cada
    vez que se le pregunta algo.

    :param ruta: ruta del CSV que se va a leer
    :ivar ruta: la ruta anterior, ya como Path
    :ivar _datos: el DataFrame leido, o None mientras no se llame a cargar()
    """

    def __init__(self, ruta: str | Path):
        """:param ruta: ruta del CSV, como texto o como Path"""
        self.ruta = Path(ruta)
        self._datos: pd.DataFrame | None = None

    def cargar(self) -> pd.DataFrame:
        """Lee el CSV de disco y lo deja cacheado en la instancia.

        :return: el contenido del fichero
        :raises FileNotFoundError: si la ruta no apunta a un fichero
        """
        if not self.ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo: {self.ruta}")

        self._datos = pd.read_csv(self.ruta)
        return self._datos

    def obtener_dimensiones(self) -> tuple[int, int]:
        """:return: la tupla (filas, columnas) del dataset cargado"""
        return self._obtener_datos().shape

    def obtener_nulos(self) -> pd.Series:
        """:return: cuantos nulos hay en cada columna"""
        return self._obtener_datos().isnull().sum()

    def obtener_duplicados(self) -> int:
        """:return: cuantas filas estan repetidas enteras"""
        return int(self._obtener_datos().duplicated().sum())

    def _obtener_datos(self) -> pd.DataFrame:
        """Devuelve el DataFrame cacheado y falla claro si no lo hay.

        Centralizar la comprobacion aqui evita repetirla en cada getter y hace
        que el error diga que hacer en vez de un AttributeError sobre None.

        :return: el DataFrame cargado
        :raises RuntimeError: si todavia no se ha llamado a cargar()
        """
        if self._datos is None:
            raise RuntimeError("Primero debes ejecutar cargar()")
        return self._datos
