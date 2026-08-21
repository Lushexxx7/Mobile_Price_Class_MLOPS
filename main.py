from src.config import MODEL_PATH, TRAIN_PATH
from src.models.pipeline import PipelineTelefonos
from src.models.evaluate import EvaluadorModelo
from src.config import METRICS_PATH, PARAMS


def main() -> None:
    pipeline, resultados = PipelineTelefonos.entrenar_desde_csv(
        TRAIN_PATH, MODEL_PATH
    )
    EvaluadorModelo.guardar_comparacion(
        resultados, METRICS_PATH, str(PARAMS["selection"]["metric"])
    )
    print(resultados.to_string(index=False))
    print(f"\nMejor modelo: {pipeline.mejor_modelo.nombre}")
    print(f"Artefacto guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()
