import json

import pandas as pd

from src.config import (
    COMPARACION_PLOT_PATH,
    CONFUSION_PLOT_PATH,
    IMPORTANCIAS_PLOT_PATH,
    METRICS_PATH,
    MLFLOW_EXPERIMENTO,
    MODEL_PATH,
    PARAMS,
    PLOTS_DIR,
    TRAIN_PATH,
)
from src.models.pipeline import PipelineTelefonos
from src.models.train import ModeloRandomForest
from src.tracking.mlflow_tracker import LinajeDatos, RastreadorMLflow

_METRICAS = ("accuracy", "precision", "recall", "f1")


def main() -> None:
    rastreador = RastreadorMLflow()
    linaje = LinajeDatos.obtener()

    pipeline, resultados = PipelineTelefonos.entrenar_desde_csv(TRAIN_PATH, MODEL_PATH)
    print(resultados.to_string(index=False))

    ganador = resultados.iloc[0]
    nombre_ganador = str(ganador["modelo"])

    # ------------------------------------------------ salidas versionadas por DVC
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8", newline="\n") as archivo:
        json.dump(
            {"modelo": nombre_ganador, **{m: float(ganador[m]) for m in _METRICAS}},
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Comparación de modelos (barras)
    resultados.to_csv(COMPARACION_PLOT_PATH, index=False, lineterminator="\n")

    # 2) Matriz de confusión del modelo ganador
    pd.DataFrame(
        {
            "real": pipeline.y_val.astype(str).to_numpy(),
            "prediccion": pipeline.predicciones[nombre_ganador].astype(str),
        }
    ).to_csv(CONFUSION_PLOT_PATH, index=False, lineterminator="\n")

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
        IMPORTANCIAS_PLOT_PATH, index=False, lineterminator="\n"
    )

    # ------------------------------------------------------- tracking en MLflow
    params_modelado = {clave: valor for clave, valor in PARAMS.items() if clave != "mlflow"}

    with rastreador.corrida_principal(
        nombre=f"pipeline_{linaje['git_commit']}",
        params=RastreadorMLflow.aplanar(params_modelado),
        tags={
            "proyecto": MLFLOW_EXPERIMENTO,
            "modelo_ganador": nombre_ganador,
            "metrica_seleccion": pipeline.metrica_seleccion,
            "n_clases": str(pipeline.y_val.nunique()),
            **linaje,
        },
    ):
        # Métricas del ganador, visibles en la corrida padre
        rastreador.registrar_metricas({m: float(ganador[m]) for m in _METRICAS})

        for ruta in (
            METRICS_PATH,
            COMPARACION_PLOT_PATH,
            CONFUSION_PLOT_PATH,
            IMPORTANCIAS_PLOT_PATH,
        ):
            rastreador.registrar_artefacto(ruta, carpeta="reportes")

        # Una corrida hija anidada por cada modelo candidato
        info_ganador = None
        for modelo in pipeline.modelos:
            fila = resultados.loc[resultados["modelo"] == modelo.nombre].iloc[0]
            with rastreador.corrida_modelo(nombre=modelo.nombre):
                rastreador.registrar_tags({"algoritmo": modelo.nombre, **linaje})
                rastreador.registrar_params(modelo.modelo.get_params())
                rastreador.registrar_metricas({m: float(fila[m]) for m in _METRICAS})

                info = rastreador.registrar_modelo(
                    modelo=modelo.modelo,
                    x_ejemplo=pipeline.x_val,
                    y_ejemplo=pipeline.predicciones[modelo.nombre],
                )
                if modelo.nombre == nombre_ganador:
                    info_ganador = info

        # Solo el ganador entra al Model Registry
        if info_ganador is not None:
            version, promovido, referencia = rastreador.promover_si_mejora(
                model_uri=info_ganador.model_uri,
                metrica=pipeline.metrica_seleccion,
                valor=float(ganador[pipeline.metrica_seleccion]),
            )
            if promovido:
                print(
                    f"[MLflow] {rastreador.registro_modelo} v{version} "
                    f"-> alias @{rastreador.alias_produccion} "
                    f"({pipeline.metrica_seleccion}={float(ganador[pipeline.metrica_seleccion]):.4f})"
                )
            else:
                print(
                    f"[MLflow] v{version} registrada pero NO promovida: "
                    f"{float(ganador[pipeline.metrica_seleccion]):.4f} no supera "
                    f"al campeón vigente ({referencia:.4f})."
                )

    print(f"\nMejor modelo: {pipeline.mejor_modelo.nombre}")
    print(f"Artefacto guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()