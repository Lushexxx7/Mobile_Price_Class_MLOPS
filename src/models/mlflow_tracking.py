"""Integración con MLflow Tracking para el proyecto de clasificación de precios.

Define dos experimentos independientes:

1. ``mobile-price-model-comparison``: un run por cada modelo candidato
   (Regresión Logística, Random Forest, SVM) entrenado sobre el mismo split
   train/validation. Sirve para comparar arquitecturas antes de elegir una.
2. ``mobile-price-production-model``: un único run con el modelo ganador,
   listo para "producción". Incluye el artefacto del modelo, la matriz de
   confusión y el registro en el Model Registry de MLflow.

El módulo reutiliza las clases ya existentes en ``src`` (``CargadorDatos``,
``PreprocesadorTelefonos``, ``crear_modelos``, ``EvaluadorModelo``) para no
duplicar lógica de negocio; MLflow solo añade la capa de tracking.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import PROJECT_ROOT, RANDOM_STATE, TARGET, TEST_SIZE
from src.data.load_data import CargadorDatos
from src.data.preprocessing import PreprocesadorTelefonos
from src.models.evaluate import EvaluadorModelo
from src.models.train import ModeloClasificacion, crear_modelos

EXPERIMENTO_COMPARACION = "mobile-price-model-comparison"
EXPERIMENTO_PRODUCCION = "mobile-price-production-model"

MLRUNS_PATH = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


class RegistradorExperimentos:
    """Orquesta el entrenamiento y registro de runs en MLflow."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        metrica_seleccion: str = "accuracy",
        target: str = TARGET,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        # MLflow >= 3.x marcó el backend de archivos ("./mlruns") como legacy
        # y exige un backend de base de datos. Usamos SQLite local, que
        # `mlflow ui --backend-store-uri sqlite:///mlflow.db` puede leer
        # directamente.
        self.tracking_uri = tracking_uri or f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
        mlflow.set_tracking_uri(self.tracking_uri)

        self.metrica_seleccion = metrica_seleccion
        self.random_state = random_state
        self.preprocesador = PreprocesadorTelefonos(
            target=target, test_size=test_size, random_state=random_state
        )
        self.evaluador = EvaluadorModelo()

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _hiperparametros(self, modelo: ModeloClasificacion) -> dict[str, Any]:
        """Extrae hiperparámetros planos del estimador (o pipeline sklearn)."""
        params = modelo.modelo.get_params()
        return {
            f"param_{clave}": valor
            for clave, valor in params.items()
            if not callable(valor)
        }

    def _figura_matriz_confusion(self, y_val, predicciones, nombre_modelo: str):
        matriz = confusion_matrix(y_val, predicciones)
        disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Matriz de confusión - {nombre_modelo}")
        return fig

    # ------------------------------------------------------------------
    # Experimento 1: comparación de modelos
    # ------------------------------------------------------------------
    def ejecutar_experimento_comparacion(
        self, datos: pd.DataFrame
    ) -> pd.DataFrame:
        """Entrena cada modelo candidato en un run separado y compara métricas."""
        mlflow.set_experiment(EXPERIMENTO_COMPARACION)

        x, y = self.preprocesador.separar_variables(datos)
        x_train, x_val, y_train, y_val = self.preprocesador.dividir_datos(x, y)

        modelos = crear_modelos(self.random_state)
        filas = []

        for modelo in modelos:
            with mlflow.start_run(run_name=modelo.nombre):
                mlflow.set_tags(
                    {
                        "proyecto": "mobile-price-classification",
                        "etapa": "comparacion",
                        "modelo": modelo.nombre,
                    }
                )
                mlflow.log_params(self._hiperparametros(modelo))
                mlflow.log_param("test_size", self.preprocesador.test_size)
                mlflow.log_param("random_state", self.random_state)
                mlflow.log_param("n_features", x_train.shape[1])
                mlflow.log_param("n_train", x_train.shape[0])
                mlflow.log_param("n_val", x_val.shape[0])

                modelo.entrenar(x_train, y_train)
                predicciones = modelo.predecir(x_val)
                metricas = self.evaluador.evaluar(y_val, predicciones)
                mlflow.log_metrics(metricas)

                fig = self._figura_matriz_confusion(y_val, predicciones, modelo.nombre)
                mlflow.log_figure(fig, "matriz_confusion.png")
                plt.close(fig)

                mlflow.sklearn.log_model(
                    modelo.modelo,
                    name="model",
                    input_example=x_val.head(3),
                )

                filas.append({"modelo": modelo.nombre, **metricas})

        resultados = (
            pd.DataFrame(filas)
            .sort_values(self.metrica_seleccion, ascending=False)
            .reset_index(drop=True)
        )
        return resultados

    # ------------------------------------------------------------------
    # Experimento 2: modelo de producción
    # ------------------------------------------------------------------
    def ejecutar_experimento_produccion(
        self,
        datos: pd.DataFrame,
        resultados_comparacion: pd.DataFrame,
        ruta_modelo: str | Path,
        registrar_en_registry: bool = True,
        nombre_registrado: str = "mobile-price-classifier",
    ) -> dict[str, Any]:
        """Reentrena el modelo ganador y lo registra como run único de producción."""
        mlflow.set_experiment(EXPERIMENTO_PRODUCCION)

        nombre_ganador = resultados_comparacion.loc[0, "modelo"]
        modelos = crear_modelos(self.random_state)
        ganador = next(m for m in modelos if m.nombre == nombre_ganador)

        x, y = self.preprocesador.separar_variables(datos)
        x_train, x_val, y_train, y_val = self.preprocesador.dividir_datos(x, y)

        with mlflow.start_run(run_name=f"produccion-{nombre_ganador}"):
            mlflow.set_tags(
                {
                    "proyecto": "mobile-price-classification",
                    "etapa": "produccion",
                    "modelo": nombre_ganador,
                    "stage": "champion",
                }
            )
            mlflow.log_params(self._hiperparametros(ganador))
            mlflow.log_param("metrica_seleccion", self.metrica_seleccion)
            mlflow.log_param("test_size", self.preprocesador.test_size)
            mlflow.log_param("random_state", self.random_state)

            ganador.entrenar(x_train, y_train)
            predicciones = ganador.predecir(x_val)
            metricas = self.evaluador.evaluar(y_val, predicciones)
            mlflow.log_metrics(metricas)

            fig = self._figura_matriz_confusion(y_val, predicciones, nombre_ganador)
            mlflow.log_figure(fig, "matriz_confusion.png")
            plt.close(fig)

            mlflow.log_table(
                resultados_comparacion, artifact_file="comparacion_modelos.json"
            )

            ruta_modelo = Path(ruta_modelo)
            if ruta_modelo.is_file():
                mlflow.log_artifact(str(ruta_modelo), artifact_path="artefacto_pkl")

            model_info = mlflow.sklearn.log_model(
                ganador.modelo,
                name="model",
                input_example=x_val.head(3),
                registered_model_name=(
                    nombre_registrado if registrar_en_registry else None
                ),
            )

            run_id = mlflow.active_run().info.run_id

        return {
            "run_id": run_id,
            "nombre_modelo": nombre_ganador,
            "metricas": metricas,
            "model_uri": model_info.model_uri,
        }

    # ------------------------------------------------------------------
    # Flujo completo (comparación + producción) desde un CSV
    # ------------------------------------------------------------------
    def ejecutar_desde_csv(
        self, ruta_datos: str | Path, ruta_modelo: str | Path
    ) -> dict[str, Any]:
        datos = CargadorDatos(ruta_datos).cargar()
        resultados_comparacion = self.ejecutar_experimento_comparacion(datos)
        info_produccion = self.ejecutar_experimento_produccion(
            datos, resultados_comparacion, ruta_modelo
        )
        return {
            "resultados_comparacion": resultados_comparacion,
            "produccion": info_produccion,
        }
