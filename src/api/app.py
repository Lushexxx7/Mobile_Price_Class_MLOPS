"""API de inferencia para la clasificación de teléfonos por rango de precio."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.model_loader import (
    MODEL_NAME,
    cargar_modelo,
    obtener_columnas,
    obtener_metadatos,
)
from src.api.schemas import (
    ETIQUETAS,
    PeticionPrediccion,
    RespuestaPrediccion,
    ResultadoPrediccion,
)
from src.data.preprocessing import PreprocesadorTelefonos

_estado: dict[str, Any] = {"modelo": None}


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    """Carga el modelo una sola vez, al arrancar el proceso."""
    _estado["modelo"] = cargar_modelo()
    metadatos = obtener_metadatos()
    if _estado["modelo"] is not None:
        print(f"[API] Modelo v{metadatos['version']} cargado ({metadatos['origen']}).")
    else:
        print("[API] ATENCIÓN: arranqué sin modelo. /predict devolverá 503.")
    yield
    _estado.clear()


app = FastAPI(
    title="API de Clasificación de Precio de Móviles",
    description="Sirve el modelo campeón registrado en MLflow.",
    version="1.0.0",
    lifespan=ciclo_vida,
)


@app.get("/")
def raiz() -> dict[str, Any]:
    return {
        "status": "online",
        "modelo": MODEL_NAME,
        **obtener_metadatos(),
        "features_esperadas": obtener_columnas(),
    }


@app.get("/health")
def salud():
    """Readiness probe: 200 solo si hay modelo en memoria."""
    if _estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    return {"status": "ready", "version": obtener_metadatos()["version"]}


@app.post("/predict", response_model=RespuestaPrediccion)
def predecir(peticion: PeticionPrediccion) -> RespuestaPrediccion:
    modelo = _estado["modelo"]
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado. Ejecuta `dvc repro` y reinicia la API.",
        )

    try:
        crudo = pd.DataFrame([fila.model_dump() for fila in peticion.data])
        # Reutiliza la validación de columnas que ya usa el pipeline batch.
        x = PreprocesadorTelefonos.preparar_inferencia(crudo, obtener_columnas())
        x = x.astype("float64")  # coincide con la firma registrada en MLflow

        predicciones = modelo.predict(x)
        proba = modelo.predict_proba(x) if hasattr(modelo, "predict_proba") else None
        clases = [int(c) for c in modelo.classes_]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Fallo en la inferencia: {error}") from error

    resultados = []
    for i, prediccion in enumerate(predicciones):
        codigo = int(prediccion)
        detalle = None
        confianza = None
        if proba is not None:
            detalle = {
                ETIQUETAS.get(clase, str(clase)): round(float(valor), 4)
                for clase, valor in zip(clases, proba[i])
            }
            confianza = round(float(max(proba[i])) * 100, 2)

        resultados.append(
            ResultadoPrediccion(
                index=i,
                price_range=codigo,
                etiqueta=ETIQUETAS.get(codigo, "desconocido"),
                confianza=confianza,
                probabilidades=detalle,
            )
        )

    return RespuestaPrediccion(
        model_metadata={"name": MODEL_NAME, **obtener_metadatos()},
        total=len(resultados),
        results=resultados,
    )