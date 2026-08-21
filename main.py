import mlflow
from src.config import MODEL_PATH, TRAIN_PATH
from src.models.pipeline import PipelineTelefonos


def main() -> None:
    # Configurar MLflow para que apunte al contenedor (vía red local)
    mlflow.set_tracking_uri("http://localhost:5008")
    mlflow.set_experiment("Proyecto_Mobile_Price_Liz")

    pipeline, resultados = PipelineTelefonos.entrenar_desde_csv(
        TRAIN_PATH, MODEL_PATH
    )
    print(resultados.to_string(index=False))
    print(f"\nMejor modelo: {pipeline.mejor_modelo.nombre}")
    print(f"Artefacto guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()

