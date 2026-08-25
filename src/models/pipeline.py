from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import METRICA_SELECCION, MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE
from src.data.load_data import CargadorDatos
from src.models.evaluate import EvaluadorModelo
from src.models.train import ModeloClasificacion, crear_modelos
from src.data.preprocessing import PreprocesadorTelefonos


class PipelineTelefonos:
    def __init__(
        self,
        modelos: list[ModeloClasificacion] | None = None,
        metrica_seleccion: str = METRICA_SELECCION,
        target: str = TARGET,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
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
        self.y_val: pd.Series | None = None
        self.x_val: pd.DataFrame | None = None     
        self.predicciones: dict[str, Any] = {}

    def entrenar(self, datos: pd.DataFrame) -> pd.DataFrame:
        x, y = self.preprocesador.separar_variables(datos)
        x_train, x_val, y_train, y_val = self.preprocesador.dividir_datos(x, y)
        self.y_val = y_val
        self.x_val = x_val                          
        self.columnas = x.columns.tolist()
        filas = []

        for modelo in self.modelos:
            modelo.entrenar(x_train, y_train)
            predicciones = modelo.predecir(x_val)
            metricas = self.evaluador.evaluar(y_val, predicciones)
            filas.append({"modelo": modelo.nombre, **metricas})
            self.predicciones[modelo.nombre] = predicciones

        self.resultados = pd.DataFrame(filas).sort_values(
            self.metrica_seleccion, ascending=False
        ).reset_index(drop=True)
        nombre_ganador = self.resultados.loc[0, "modelo"]
        self.mejor_modelo = next(
            modelo for modelo in self.modelos if modelo.nombre == nombre_ganador
        )
        return self.resultados.copy()

    def guardar(self, ruta: str | Path = MODEL_PATH) -> Path:
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
        artefacto = cls.cargar(ruta)
        x = PreprocesadorTelefonos.preparar_inferencia(
            datos, artefacto["columnas"]
        )
        return artefacto["modelo"].predict(x)

    @classmethod
    def entrenar_desde_csv(
        cls, ruta_datos: str | Path, ruta_modelo: str | Path = MODEL_PATH
    ) -> tuple["PipelineTelefonos", pd.DataFrame]:
        """Ejecuta el flujo completo a partir de un CSV."""
        datos = CargadorDatos(ruta_datos).cargar()
        pipeline = cls()
        resultados = pipeline.entrenar(datos)
        pipeline.guardar(ruta_modelo)
        return pipeline, resultados

