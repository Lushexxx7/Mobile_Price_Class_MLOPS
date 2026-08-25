import mlflow

from src.config import METRICS_PATH, MODEL_PATH, PARAMS, TRAIN_PATH
from src.data.load_data import CargadorDatos
from src.models.evaluate import EvaluadorModelo
from src.models.pipeline import PipelineTelefonos
from src.models.tracking import RastreadorMLflow


def main() -> None:
    datos = CargadorDatos(TRAIN_PATH).cargar()
    metrica = str(PARAMS["selection"]["metric"])
    pipeline = PipelineTelefonos(metrica_seleccion=metrica)
    resultados = pipeline.entrenar(datos)
    pipeline.guardar(MODEL_PATH)
    EvaluadorModelo.guardar_comparacion(resultados, METRICS_PATH, metrica)

    x, y = pipeline.preprocesador.separar_variables(datos)
    _, x_validacion, _, y_validacion = pipeline.preprocesador.dividir_datos(x, y)
    rastreador = RastreadorMLflow()
    registros = []
    with mlflow.start_run(run_name="comparacion_modelos") as parent:
        mlflow.set_tags({"tipo": "baseline", **rastreador.linaje()})
        for modelo in pipeline.modelos:
            registros.append(
                rastreador.registrar_modelo(
                    modelo,
                    x_validacion,
                    y_validacion,
                    pipeline.evaluador,
                    run_name=f"baseline_{modelo.nombre}",
                )
            )
        mejor = max(registros, key=lambda item: item["metricas"][metrica])
        mlflow.log_metric(f"best_{metrica}", mejor["metricas"][metrica])
        mlflow.log_param("best_child_run_id", mejor["run_id"])
        version = rastreador.registrar_y_asignar_alias(
            mejor["model_uri"],
            "champion",
            {"tipo": "baseline", "parent_run_id": parent.info.run_id},
        )

    print(resultados.to_string(index=False))
    print(f"\nMejor modelo: {pipeline.mejor_modelo.nombre}")
    print(f"Artefacto guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")
    print(f"MLflow: versión {version} registrada con alias champion")


if __name__ == "__main__":
    main()
