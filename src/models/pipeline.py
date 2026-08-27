"""Orquestacion del entrenamiento: compara modelos, elige y persiste."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE
from src.data.load_data import CargadorDatos
from src.models.evaluate import EvaluadorModelo
from src.models.train import ModeloClasificacion, crear_modelos
from src.data.preprocessing import PreprocesadorTelefonos


class PipelineTelefonos:
    """Entrena varios modelos, se queda con el mejor y lo guarda.

    Es la pieza que ata las demas: preprocesador, modelos y evaluador. El
    artefacto que escribe lleva tambien las columnas de entrenamiento, para que
    la inferencia pueda reordenar la entrada sin adivinar.

    :param modelos: modelos a comparar; si es None usa los tres por defecto
    :param metrica_seleccion: metrica con la que se elige al ganador
    :param target: nombre de la variable objetivo
    :param test_size: proporcion reservada a validacion
    :param random_state: semilla de la division y de los modelos
    :ivar mejor_modelo: el ganador, o None mientras no se entrene
    :ivar columnas: columnas de entrenamiento, en orden
    :ivar resultados: tabla de metricas ordenada por `metrica_seleccion`
    """

    def __init__(
        self,
        modelos: list[ModeloClasificacion] | None = None,
        metrica_seleccion: str = "accuracy",
        target: str = TARGET,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        """:param modelos: modelos a comparar, None para los tres por defecto
        :param metrica_seleccion: accuracy, precision, recall o f1
        :param target: nombre de la variable objetivo
        :param test_size: proporcion reservada a validacion
        :param random_state: semilla
        :raises ValueError: si la metrica no es una de las cuatro validas
        """
        metricas_validas = {"accuracy", "precision", "recall", "f1"}
        if metrica_seleccion not in metricas_validas:
            raise ValueError(f"Métrica no válida: {metrica_seleccion}")
        self.modelos = modelos or crear_modelos(random_state)
        self.metrica_seleccion = metrica_seleccion
        self.preprocesador = PreprocesadorTelefonos(
            target=target,
            test_size=test_size,
            random_state=random_state,
        )
        self.evaluador = EvaluadorModelo()
        self.mejor_modelo: ModeloClasificacion | None = None
        self.columnas: list[str] = []
        self.resultados: pd.DataFrame | None = None

    def entrenar(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Entrena todos los modelos y ordena los resultados.

        :param datos: dataset completo, con la columna objetivo
        :return: tabla de metricas, el ganador en la primera fila
        """
        x, y = self.preprocesador.separar_variables(datos)
        x_train, x_val, y_train, y_val = self.preprocesador.dividir_datos(x, y)
        self.columnas = x.columns.tolist()
        filas = []

        for modelo in self.modelos:
            modelo.entrenar(x_train, y_train)
            predicciones = modelo.predecir(x_val)
            metricas = self.evaluador.evaluar(y_val, predicciones)
            filas.append({"modelo": modelo.nombre, **metricas})

        self.resultados = (
            pd.DataFrame(filas)
            .sort_values(self.metrica_seleccion, ascending=False)
            .reset_index(drop=True)
        )
        nombre_ganador = self.resultados.loc[0, "modelo"]
        self.mejor_modelo = next(
            modelo for modelo in self.modelos if modelo.nombre == nombre_ganador
        )
        return self.resultados.copy()

    def guardar(self, ruta: str | Path = MODEL_PATH) -> Path:
        """Serializa el modelo ganador junto con su metadato.

        :param ruta: destino del .pkl
        :return: la ruta escrita
        :raises RuntimeError: si todavia no se ha entrenado
        """
        if self.mejor_modelo is None or self.resultados is None:
            raise RuntimeError("Primero debes entrenar el pipeline.")
        destino = Path(ruta)
        destino.parent.mkdir(parents=True, exist_ok=True)
        artefacto = {
            "modelo": self.mejor_modelo.modelo,
            "nombre": self.mejor_modelo.nombre,
            "columnas": self.columnas,
            "target": self.preprocesador.target,
            "metrica_seleccion": self.metrica_seleccion,
            "resultados": self.resultados.to_dict(orient="records"),
        }
        joblib.dump(artefacto, destino)
        return destino

    @staticmethod
    def cargar(ruta: str | Path = MODEL_PATH) -> dict[str, Any]:
        """Lee un artefacto de disco y comprueba que tenga lo que hace falta.

        :param ruta: fichero .pkl a leer
        :return: el dict con modelo, nombre, columnas y target
        :raises FileNotFoundError: si el fichero no existe
        :raises ValueError: si el .pkl no tiene el formato esperado
        """
        origen = Path(ruta)
        if not origen.is_file():
            raise FileNotFoundError(f"No existe el modelo: {origen}")
        artefacto = joblib.load(origen)
        requeridos = {"modelo", "nombre", "columnas", "target"}
        if not requeridos.issubset(artefacto):
            raise ValueError("El artefacto no tiene el formato esperado.")
        return artefacto

    @classmethod
    def predecir(cls, datos: pd.DataFrame, ruta: str | Path = MODEL_PATH):
        """Predice con un artefacto guardado, sin necesidad de reentrenar.

        :param datos: filas a predecir, pueden traer columnas de mas
        :param ruta: artefacto .pkl a usar
        :return: la clase predicha para cada fila
        """
        artefacto = cls.cargar(ruta)
        x = PreprocesadorTelefonos.preparar_inferencia(datos, artefacto["columnas"])
        return artefacto["modelo"].predict(x)

    @classmethod
    def entrenar_desde_csv(
        cls, ruta_datos: str | Path, ruta_modelo: str | Path = MODEL_PATH
    ) -> tuple["PipelineTelefonos", pd.DataFrame]:
        """Ejecuta el flujo completo a partir de un CSV.

        :param ruta_datos: CSV de entrenamiento
        :param ruta_modelo: destino del artefacto
        :return: la tupla (pipeline entrenado, tabla de resultados)
        """
        datos = CargadorDatos(ruta_datos).cargar()
        pipeline = cls()
        resultados = pipeline.entrenar(datos)
        pipeline.guardar(ruta_modelo)
        return pipeline, resultados
