import json

import pandas as pd

from src.config import (
    COMPARACION_PLOT_PATH,
    CONFUSION_PLOT_PATH,
    IMPORTANCIAS_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    TRAIN_PATH,
)
from src.models.pipeline import PipelineTelefonos
from src.models.train import ModeloRandomForest


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
        
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Comparación de modelos (barras)
    resultados.to_csv(COMPARACION_PLOT_PATH, index=False)

    # 2) Matriz de confusión del modelo ganador
    nombre_ganador = str(ganador["modelo"])
    pd.DataFrame(
        {
            "real": pipeline.y_val.astype(str).to_numpy(),
            "prediccion": pipeline.predicciones[nombre_ganador].astype(str),
        }
    ).to_csv(CONFUSION_PLOT_PATH, index=False)

    # 3) Importancia de variables del Random Forest
    random_forest = next(
        modelo for modelo in pipeline.modelos if isinstance(modelo, ModeloRandomForest)
    )
    pd.DataFrame(
        {
            "variable": pipeline.columnas,
            "importancia": random_forest.obtener_importancias(),
        }
    ).sort_values("importancia", ascending=False).to_csv(
        IMPORTANCIAS_PLOT_PATH, index=False
    )

    print(f"\nMejor modelo: {pipeline.mejor_modelo.nombre}")
    print(f"Artefacto guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()