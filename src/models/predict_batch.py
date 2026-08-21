import pandas as pd

from src.config import MODEL_PATH, PREDICTIONS_PATH, TEST_PATH
from src.data.load_data import CargadorDatos
from src.models.pipeline import PipelineTelefonos


def main() -> None:
    datos = CargadorDatos(TEST_PATH).cargar()
    predicciones = PipelineTelefonos.predecir(datos, MODEL_PATH)

    resultado = pd.DataFrame({"id": datos["id"], "price_range": predicciones})
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Predicciones generadas: {len(resultado)}")
    print(f"Guardadas en: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()