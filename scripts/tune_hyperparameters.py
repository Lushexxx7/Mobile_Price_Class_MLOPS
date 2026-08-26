import json
import mlflow

from src.config import MLFLOW_ALIAS_CHALLENGER, PARAMS, PROJECT_ROOT, TRAIN_PATH
from src.data.load_data import CargadorDatos
from src.data.preprocessing import PreprocesadorTelefonos
from src.models.evaluate import EvaluadorModelo
from src.models.tracking import RastreadorMLflow
from src.models.train import (
    ModeloRandomForest,
    ModeloRegresionLogistica,
    ModeloSVM,
)


def candidatos():
    busqueda = PARAMS["hyperparameter_search"]
    for c in busqueda["logistic_regression"]["C"]:
        modelo = ModeloRegresionLogistica()
        modelo.modelo.set_params(classifier__C=float(c))
        yield modelo, {"C": c}
    for configuracion in busqueda["random_forest"]["trials"]:
        n_estimators = configuracion["n_estimators"]
        max_depth = configuracion["max_depth"]
        modelo = ModeloRandomForest()
        modelo.modelo.set_params(
            n_estimators=int(n_estimators), max_depth=max_depth
        )
        yield modelo, dict(configuracion)
    for c in busqueda["svm"]["C"]:
        modelo = ModeloSVM()
        modelo.modelo.set_params(classifier__C=float(c))
        yield modelo, {"C": c, "gamma": PARAMS["svm"]["gamma"]}


def main() -> None:
    datos = CargadorDatos(TRAIN_PATH).cargar()
    preprocesador = PreprocesadorTelefonos()
    x, y = preprocesador.separar_variables(datos)
    x_train, x_val, y_train, y_val = preprocesador.dividir_datos(x, y)
    evaluador = EvaluadorModelo()
    rastreador = RastreadorMLflow()
    resultados = []
    with mlflow.start_run(run_name="busqueda_hiperparametros") as parent:
        mlflow.set_tags({"tipo": "hyperparameter_search", **rastreador.linaje()})
        for indice, (modelo, parametros) in enumerate(candidatos(), start=1):
            modelo.entrenar(x_train, y_train)
            registro = rastreador.registrar_modelo(
                modelo,
                x_val,
                y_val,
                evaluador,
                run_name=f"trial_{indice}_{modelo.nombre}",
                parametros=parametros,
            )
            resultados.append({**registro, "parametros": parametros})
        metrica = str(PARAMS["selection"]["metric"])
        mejor = max(resultados, key=lambda item: item["metricas"][metrica])
        rastreador.registrar_resumen(resultados, "busqueda_hiperparametros")
        mlflow.log_metric(f"best_{metrica}", mejor["metricas"][metrica])
        mlflow.log_param("best_child_run_id", mejor["run_id"])
        version = rastreador.registrar_y_asignar_alias(
            mejor["model_uri"],
            MLFLOW_ALIAS_CHALLENGER,
            {"tipo": "hyperparameter_search", "parent_run_id": parent.info.run_id},
        )
    resumen = {
        "metrica_seleccion": metrica,
        "mejor_modelo": mejor["modelo"],
        "mejor_valor": mejor["metricas"][metrica],
        "mejores_parametros": mejor["parametros"],
        "mlflow_run_id": mejor["run_id"],
        "mlflow_model_version": version,
        "total_experimentos": len(resultados),
    }
    destino = PROJECT_ROOT / "reports" / "hyperparameter_search.json"
    # newline="\n": ver nota en scripts/validate_data.py sobre los hashes de DVC.
    destino.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
