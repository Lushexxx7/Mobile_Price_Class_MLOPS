import os
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import MODEL_PATH
from src.models.pipeline import PipelineTelefonos

app = FastAPI(title="API de Inferencia de Precios de Móviles", version="1.0.0")

# Definir la estructura de entrada esperada (Payload POST)
# Adaptado a las posibles variables de x (features de los móviles)
# Se recibirá un diccionario con los valores de las características
class PredictionRequest(BaseModel):
    features: dict[str, float | int | str]

@app.on_event("startup")
def load_model():
    """Se ejecuta una sola vez cuando la API arranca."""
    try:
        # Se asegura que la ruta base existe, aunque en realidad solo verificaremos si está el file
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️ Advertencia: No se encontró el modelo en {MODEL_PATH}. Debes ejecutar main.py primero.")
        else:
            print(f"✅ Modelo listo para ser cargado en memoria desde {MODEL_PATH}.")
    except Exception as e:
        print(f"Error en startup: {e}")

@app.post("/predict")
def predict(req: PredictionRequest):
    """Endpoint para hacer inferencia."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Modelo no disponible. Verifica que se haya entrenado.")
    
    try:
        # Convertimos las features a DataFrame, que es lo que espera `PipelineTelefonos.predecir`
        datos = pd.DataFrame([req.features])
        pred = PipelineTelefonos.predecir(datos, ruta=MODEL_PATH)
        return {"prediction": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")
