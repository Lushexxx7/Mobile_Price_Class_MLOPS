from src.config import MODEL_PATH, TRAIN_PATH
from src.pipeline import PipelineTelefonos


def main() -> None:
    pipeline, resultados = PipelineTelefonos.entrenar_desde_csv(
        TRAIN_PATH, MODEL_PATH
    )
    print(resultados.to_string(index=False))
    print(f"\nMejor modelo: {pipeline.mejor_modelo.nombre}")
    print(f"Artefacto guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
