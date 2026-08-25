"""Integración de MLflow 3.x con el pipeline de clasificación de teléfonos.

Notas de compatibilidad (MLflow 3.15.1):
- `log_model()` usa `name=`; el antiguo `artifact_path=` está deprecado.
- Los *stages* (Staging/Production) fueron ELIMINADOS en MLflow 3.
  La promoción se hace con alias (`@champion`).
- El Model Registry exige backend SQL (sqlite/postgres).
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_ACTIVO,
    MLFLOW_ALIAS_PRODUCCION,
    MLFLOW_EXPERIMENTO,
    MLFLOW_REGISTRO_MODELO,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
)

_TIPOS_SIMPLES = (int, float, str, bool)


class LinajeDatos:
    """Metadatos de Git y DVC que hacen auditable cada corrida."""

    @staticmethod
    def _ejecutar(comando: list[str]) -> str:
        try:
            resultado = subprocess.run(
                comando,
                capture_output=True,
                text=True,
                timeout=20,
                cwd=PROJECT_ROOT,
            )
            return resultado.stdout.strip()
        except Exception:
            return ""

    @classmethod
    def obtener(cls) -> dict[str, str]:
        commit = cls._ejecutar(["git", "rev-parse", "--short", "HEAD"])
        rama = cls._ejecutar(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        sucio = cls._ejecutar(["git", "status", "--porcelain"])
        dvc_salida = cls._ejecutar(["dvc", "status"])

        return {
            "git_commit": commit or "sin_git",
            "git_rama": rama or "desconocida",
            "git_limpio": "no" if sucio else "si",
            "dvc_estado": (
                "sincronizado"
                if not dvc_salida or "up to date" in dvc_salida
                else "cambios_pendientes"
            ),
        }


class RastreadorMLflow:
    """Encapsula el tracking de MLflow para el pipeline de teléfonos."""

    def __init__(
        self,
        experimento: str = MLFLOW_EXPERIMENTO,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        registro_modelo: str = MLFLOW_REGISTRO_MODELO,
        alias_produccion: str = MLFLOW_ALIAS_PRODUCCION,
        activo: bool = MLFLOW_ACTIVO,
    ):
        self.activo = activo
        self.registro_modelo = registro_modelo
        self.alias_produccion = alias_produccion
        self._cliente: MlflowClient | None = None

        if self.activo:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experimento)
            self._cliente = MlflowClient(tracking_uri=tracking_uri)

    @property
    def cliente(self) -> MlflowClient:
        if self._cliente is None:
            raise RuntimeError("El rastreador está desactivado (mlflow.activo=false).")
        return self._cliente

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _limpiar_params(params: dict[str, Any]) -> dict[str, Any]:
        """Conserva solo valores primitivos: la UI de MLflow no muestra objetos."""
        return {
            clave: valor
            for clave, valor in params.items()
            if valor is None or isinstance(valor, _TIPOS_SIMPLES)
        }

    @classmethod
    def aplanar(cls, datos: dict[str, Any], prefijo: str = "") -> dict[str, Any]:
        """Convierte params.yaml anidado en claves planas: 'split.test_size'."""
        plano: dict[str, Any] = {}
        for clave, valor in datos.items():
            nombre = f"{prefijo}{clave}"
            if isinstance(valor, dict):
                plano.update(cls.aplanar(valor, f"{nombre}."))
            else:
                plano[nombre] = valor
        return plano

    # --------------------------------------------------------------- corridas

    @contextmanager
    def corrida_principal(
        self, nombre: str, params: dict[str, Any], tags: dict[str, str]
    ) -> Iterator[Any]:
        """Corrida padre: agrupa la ejecución completa del pipeline."""
        if not self.activo:
            yield None
            return
        with mlflow.start_run(run_name=nombre) as corrida:
            mlflow.set_tags(tags)
            mlflow.log_params(self._limpiar_params(params))
            yield corrida

    @contextmanager
    def corrida_modelo(self, nombre: str) -> Iterator[Any]:
        """Corrida hija anidada: un modelo candidato."""
        if not self.activo:
            yield None
            return
        with mlflow.start_run(run_name=nombre, nested=True) as corrida:
            yield corrida

    # -------------------------------------------------------------- registros

    def registrar_params(self, params: dict[str, Any]) -> None:
        if self.activo:
            mlflow.log_params(self._limpiar_params(params))

    def registrar_tags(self, tags: dict[str, str]) -> None:
        if self.activo:
            mlflow.set_tags(tags)

    def registrar_metricas(self, metricas: dict[str, float]) -> None:
        if self.activo:
            mlflow.log_metrics({k: float(v) for k, v in metricas.items()})

    def registrar_artefacto(self, ruta: str | Path, carpeta: str = "reportes") -> None:
        if self.activo:
            mlflow.log_artifact(str(ruta), artifact_path=carpeta)

    def registrar_modelo(
        self,
        modelo: Any,
        x_ejemplo: Any,
        y_ejemplo: Any,
        nombre_artefacto: str = "modelo",
    ):
        """Guarda el estimador con firma e input_example. Devuelve ModelInfo."""
        if not self.activo:
            return None
        x_firma = x_ejemplo.astype("float64")
        firma = infer_signature(x_firma, y_ejemplo)
        return mlflow.sklearn.log_model(
            sk_model=modelo,
            name=nombre_artefacto,          # MLflow 3: sustituye a artifact_path
            signature=firma,
            input_example=x_firma.head(3),
        )

    def metrica_campeon(self, metrica: str) -> float | None:
        """Métrica del campeón vigente, o None si aún no hay ninguno."""
        if not self.activo:
            return None
        try:
            version = self.cliente.get_model_version_by_alias(
                name=self.registro_modelo, alias=self.alias_produccion
            )
        except Exception:
            return None
        return self.cliente.get_run(version.run_id).data.metrics.get(metrica)

    def promover_si_mejora(
        self, model_uri: str, metrica: str, valor: float
    ) -> tuple[str | None, bool, float | None]:
        """Registra siempre la versión; mueve el alias solo si supera al campeón."""
        if not self.activo:
            return None, False, None

        referencia = self.metrica_campeon(metrica)
        version = mlflow.register_model(model_uri=model_uri, name=self.registro_modelo)
        promovido = referencia is None or valor > referencia

        if promovido:
            self.cliente.set_registered_model_alias(
                name=self.registro_modelo,
                alias=self.alias_produccion,
                version=version.version,
            )
        return version.version, promovido, referencia

    def cargar_campeon(self):
        """Carga el modelo marcado como campeón para inferencia."""
        uri = f"models:/{self.registro_modelo}@{self.alias_produccion}"
        return mlflow.sklearn.load_model(uri)