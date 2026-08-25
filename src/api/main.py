import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Importamos las funciones centralizadas de MLOps
from src.api.model_loader import load_model, get_model_metadata, MODEL_NAME, FEATURE_COLUMNS

# ==========================================
# 1. INICIALIZACIÓN DE FASTAPI Y CARGA
# ==========================================
app = FastAPI(
    title="API de Clasificación de Precios de Teléfonos",
    description="API MLOps para predecir el rango de precio de un teléfono móvil según sus características técnicas.",
    version="1.0.0",
)

model = None
model_version_info = {"version": "Desconocida", "run_id": "Desconocido"}

# Etiquetas legibles para cada clase de price_range (0 a 3)
ETIQUETAS_PRECIO = {
    0: "Precio Bajo",
    1: "Precio Medio",
    2: "Precio Alto",
    3: "Precio Muy Alto",
}


@app.on_event("startup")
def startup_event():
    global model, model_version_info
    model = load_model()
    model_version_info = get_model_metadata()
    if model:
        print(f"[FastAPI] ¡Modelo v{model_version_info['version']} cargado exitosamente en producción!")
    else:
        print("[FastAPI ERROR] El modelo inició en None.")


# ==========================================
# 2. ESQUEMAS DE ENTRADA (Pydantic)
# ==========================================
class PhoneFeatures(BaseModel):
    battery_power: float
    blue: int
    clock_speed: float
    dual_sim: int
    fc: float
    four_g: int
    int_memory: float
    m_dep: float
    mobile_wt: float
    n_cores: int
    pc: float
    px_height: float
    px_width: float
    ram: float
    sc_h: float
    sc_w: float
    talk_time: float
    three_g: int
    touch_screen: int
    wifi: int


class PredictionRequest(BaseModel):
    data: List[PhoneFeatures]


# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "model_name": MODEL_NAME,
        "production_version": model_version_info["version"],
        "run_id": model_version_info["run_id"],
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo no está cargado en memoria o no se encontró en el Model Registry.",
        )

    try:
        input_data = pd.DataFrame([item.dict() for item in payload.data])
        input_data = input_data[FEATURE_COLUMNS]

        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None

        results = []
        for i, pred in enumerate(predictions):
            pred_int = int(pred)
            confidence = float(max(probabilities[i])) if probabilities is not None else None

            detalle_probabilidades = None
            if probabilities is not None:
                detalle_probabilidades = {
                    ETIQUETAS_PRECIO[clase]: round(float(probabilities[i][clase]), 4)
                    for clase in range(probabilities.shape[1])
                }

            results.append(
                {
                    "index": i,
                    "prediction_code": pred_int,
                    "price_range": ETIQUETAS_PRECIO.get(pred_int, "Desconocido"),
                    "confidence_score": round(confidence * 100, 2) if confidence else None,
                    "probabilities_detail": detalle_probabilidades,
                }
            )

        return {
            "model_metadata": {
                "name": MODEL_NAME,
                "version": model_version_info["version"],
                "run_id": model_version_info["run_id"],
            },
            "total_predictions": len(predictions),
            "results": results,
            "message": "Inferencia completada con éxito.",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")
