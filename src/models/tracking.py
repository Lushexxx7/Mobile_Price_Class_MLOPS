import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import matplotlib
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PARAMS,
    PROJECT_ROOT,
)
from src.models.evaluate import EvaluadorModelo
from src.models.train import ModeloClasificacion


class RastreadorMLflow:
    def __init__(self, config: dict | None = None, tracking_uri: str | None = None):
        self.config = config or PARAMS["mlflow"]
        # Por defecto la URI viene de src.config, que ya aplico el override de
        # MLFLOW_TRACKING_URI y anclo las rutas sqlite relativas a la raiz del
        # proyecto. Leerla de params.yaml aqui romperia ambas cosas.
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        self.experiment_name = str(self.config.get("experiment_name", MLFLOW_EXPERIMENT_NAME))
        self.registered_model_name = str(
            self.config.get("registered_model_name", MLFLOW_REGISTERED_MODEL_NAME)
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    @staticmethod
    def _comando_git(*args: str) -> str:
        """Linaje de Git, en el mejor de los casos.

        El OSError no es defensivo de mas: la imagen de entrenamiento no lleva
        git instalado ni el directorio .git, asi que subprocess lanza
        FileNotFoundError al no encontrar el binario. Con `check=False` solo se
        cubria el caso de que git existiera y devolviera error, y entrenar
        dentro del contenedor reventaba antes de registrar nada. El commit es
        un metadato util, no un requisito para entrenar.
        """
        try:
            resultado = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return "no-disponible"

        return resultado.stdout.strip() or "no-disponible"

    @staticmethod
    def _hash_dvc(ruta: Path) -> str:
        if not ruta.is_file():
            return "no-disponible"
        contenido = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
        salidas = contenido.get("outs", [])
        return str(salidas[0].get("md5", "no-disponible")) if salidas else "no-disponible"

    def linaje(self) -> dict[str, str]:
        return {
            "git_commit": self._comando_git("rev-parse", "--short", "HEAD"),
            "git_branch": self._comando_git("branch", "--show-current"),
            "dvc_train_md5": self._hash_dvc(PROJECT_ROOT / "data" / "raw" / "train.csv.dvc"),
            "dvc_test_md5": self._hash_dvc(PROJECT_ROOT / "data" / "raw" / "test.csv.dvc"),
        }

    @staticmethod
    def parametros_modelo(modelo: ModeloClasificacion) -> dict[str, Any]:
        permitidos = (str, int, float, bool)
        return {
            clave: valor if isinstance(valor, permitidos) or valor is None else str(valor)
            for clave, valor in modelo.modelo.get_params(deep=True).items()
            if "__" not in clave or clave.startswith("classifier__")
        }

    def registrar_modelo(
        self,
        modelo: ModeloClasificacion,
        x_validacion: pd.DataFrame,
        y_validacion: pd.Series,
        evaluador: EvaluadorModelo,
        run_name: str,
        parametros: dict[str, Any] | None = None,
        nested: bool = True,
    ) -> dict[str, Any]:
        predicciones = modelo.predecir(x_validacion)
        metricas = evaluador.evaluar(y_validacion, predicciones)
        diagnostico = evaluador.diagnostico(y_validacion, predicciones)
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            mlflow.set_tags({"modelo": modelo.nombre, **self.linaje()})
            mlflow.log_params(parametros or self.parametros_modelo(modelo))
            mlflow.log_metrics(metricas)
            mlflow.log_dict(
                {
                    "matriz_confusion": diagnostico["matriz_confusion"].tolist(),
                    "reporte_clasificacion": diagnostico["reporte"],
                },
                "diagnosticos/clasificacion.json",
            )
            figura, eje = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(
                confusion_matrix=diagnostico["matriz_confusion"]
            ).plot(ax=eje, colorbar=False)
            eje.set_title(f"Matriz de confusión - {modelo.nombre}")
            figura.tight_layout()
            mlflow.log_figure(figura, "graficas/matriz_confusion.png")
            plt.close(figura)
            ejemplo_entrada = x_validacion.astype("float64")
            firma = infer_signature(ejemplo_entrada, predicciones)
            model_info = mlflow.sklearn.log_model(
                sk_model=modelo.modelo,
                name="model",
                signature=firma,
                input_example=ejemplo_entrada.head(5),
            )
            return {
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "metricas": metricas,
                "modelo": modelo.nombre,
            }

    @staticmethod
    def registrar_resumen(
        registros: list[dict[str, Any]], nombre: str
    ) -> pd.DataFrame:
        filas = []
        for registro in registros:
            fila = {"modelo": registro["modelo"], **registro["metricas"]}
            fila.update(
                {
                    f"param_{clave}": valor
                    for clave, valor in registro.get("parametros", {}).items()
                }
            )
            filas.append(fila)
        tabla = pd.DataFrame(filas)
        mlflow.log_table(tabla, artifact_file=f"resumen/{nombre}.json")

        metricas = [
            metrica
            for metrica in ("accuracy", "precision", "recall", "f1")
            if metrica in tabla.columns
        ]
        figura, eje = plt.subplots(figsize=(10, 6))
        tabla.set_index("modelo")[metricas].plot(kind="bar", ax=eje)
        eje.set_ylim(0, 1)
        eje.set_ylabel("Valor")
        eje.set_title("Comparación de métricas")
        eje.tick_params(axis="x", rotation=25)
        figura.tight_layout()
        mlflow.log_figure(figura, f"graficas/{nombre}.png")
        plt.close(figura)
        return tabla

    def registrar_y_asignar_alias(
        self, model_uri: str, alias: str, tags: dict[str, str] | None = None
    ) -> str:
        version = mlflow.register_model(model_uri, self.registered_model_name)
        self.client.set_registered_model_alias(
            self.registered_model_name, alias, version.version
        )
        for clave, valor in (tags or {}).items():
            self.client.set_model_version_tag(
                self.registered_model_name, version.version, clave, valor
            )
        return str(version.version)
