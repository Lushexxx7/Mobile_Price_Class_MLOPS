import pandas as pd

from src.models.train import ModeloRegresionLogistica
from src.models.pipeline import PipelineTelefonos


def test_pipeline_entrena_guarda_y_predice(tmp_path):
    datos = pd.DataFrame(
        {
            "ram": [512, 600, 1000, 1200, 2000, 2200, 3000, 3200] * 4,
            "battery_power": [500, 600, 800, 900, 1100, 1200, 1500, 1600] * 4,
            "price_range": [0, 0, 1, 1, 2, 2, 3, 3] * 4,
        }
    )
    ruta = tmp_path / "modelo.pkl"
    pipeline = PipelineTelefonos(modelos=[ModeloRegresionLogistica()], test_size=0.25)

    resultados = pipeline.entrenar(datos)
    pipeline.guardar(ruta)
    predicciones = PipelineTelefonos.predecir(datos.drop(columns="price_range"), ruta)

    assert resultados["modelo"].tolist() == ["Regresión Logística"]
    assert ruta.is_file()
    assert len(predicciones) == len(datos)
