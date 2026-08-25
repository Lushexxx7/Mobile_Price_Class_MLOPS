import pandas as pd

from src.config import MODEL_PATH, PREDICTIONS_PATH, TEST_PATH
from src.data.load_data import CargadorDatos
from src.models.pipeline import PipelineTelefonos


def main() -> None:
    datos = CargadorDatos(TEST_PATH).cargar()
    salida = pd.DataFrame({"price_range": PipelineTelefonos.predecir(datos, MODEL_PATH)})
    if "id" in datos.columns:
        salida.insert(0, "id", datos["id"].to_numpy())
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(PREDICTIONS_PATH, index=False)


if __name__ == "__main__":
    main()
