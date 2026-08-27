"""Genera las predicciones de test.csv con el modelo ya entrenado."""

import pandas as pd

from src.config import MODEL_PATH, PREDICTIONS_PATH, TEST_PATH
from src.data.load_data import CargadorDatos
from src.models.pipeline import PipelineTelefonos


def main() -> None:
    """Predice sobre test.csv y guarda el CSV que versiona DVC."""
    datos = CargadorDatos(TEST_PATH).cargar()
    salida = pd.DataFrame({"price_range": PipelineTelefonos.predecir(datos, MODEL_PATH)})
    if "id" in datos.columns:
        salida.insert(0, "id", datos["id"].to_numpy())
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n": to_csv usa CRLF en Windows por defecto, y ese CSV es
    # una salida cacheada por DVC. Sin fijarlo, entrenar en Windows y en el
    # contenedor daba hashes distintos para predicciones identicas.
    salida.to_csv(PREDICTIONS_PATH, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
