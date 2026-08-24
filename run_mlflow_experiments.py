"""Ejecuta los dos experimentos de MLflow del proyecto.

Experimento 1 (`mobile-price-model-comparison`):
    Un run por cada modelo candidato (Regresión Logística, Random Forest,
    SVM), con sus hiperparámetros, métricas y matriz de confusión.

Experimento 2 (`mobile-price-production-model`):
    Reentrena el modelo ganador (según `metrica_seleccion`, por defecto
    accuracy) y lo registra como candidato de producción, incluyendo el
    artefacto `.pkl` generado por el pipeline existente.

Uso:
    python run_mlflow_experiments.py

Para ver los resultados:
    mlflow ui --backend-store-uri file:./mlruns
"""

from src.config import MODEL_PATH, TRAIN_PATH
from src.models.mlflow_tracking import RegistradorExperimentos
from src.models.pipeline import PipelineTelefonos


def main() -> None:
    registrador = RegistradorExperimentos()

    resultado = registrador.ejecutar_desde_csv(TRAIN_PATH, MODEL_PATH)

    print("=== Experimento 1: comparación de modelos ===")
    print(resultado["resultados_comparacion"].to_string(index=False))

    produccion = resultado["produccion"]
    print("\n=== Experimento 2: modelo de producción ===")
    print(f"Modelo ganador: {produccion['nombre_modelo']}")
    print(f"Run ID: {produccion['run_id']}")
    print(f"Model URI: {produccion['model_uri']}")
    print(f"Métricas: {produccion['metricas']}")

    # Mantiene el artefacto .pkl consistente con el resto del proyecto
    # (main.py, src/models/predict.py) usando el pipeline ya existente.
    pipeline, _ = PipelineTelefonos.entrenar_desde_csv(TRAIN_PATH, MODEL_PATH)
    print(f"\nArtefacto .pkl actualizado en: {MODEL_PATH}")
    print(f"Mejor modelo (pipeline clásico): {pipeline.mejor_modelo.nombre}")


if __name__ == "__main__":
    main()
