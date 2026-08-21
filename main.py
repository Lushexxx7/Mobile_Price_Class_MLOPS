import json

from src.config import METRICS_PATH, MODEL_PATH, TRAIN_PATH
from src.models.pipeline import PipelineTelefonos


def main() -> None:
    pipeline, resultados = PipelineTelefonos.entrenar_desde_csv(TRAIN_PATH, MODEL_PATH)
    print(resultados.to_string(index=False))

    ganador = resultados.iloc[0]
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as archivo:
        json.dump(
            {
                "modelo": str(ganador["modelo"]),
                "accuracy": float(ganador["accuracy"]),
                "precision": float(ganador["precision"]),
                "recall": float(ganador["recall"]),
                "f1": float(ganador["f1"]),
            },
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nMejor modelo: {pipeline.mejor_modelo.nombre}")
    print(f"Artefacto guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()