"""API de inferencia para la clasificación de teléfonos por rango de precio.

Sustituye a `src/api/main.py`, que colisionaba de nombre con el `main.py` de la
raíz (el entrenamiento). El punto de entrada ahora es `src.api.app:app`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from src.api.model_loader import (
    FEATURE_COLUMNS,
    MODEL_ALIAS,
    MODEL_NAME,
    get_model_metadata,
    load_model,
)
from src.api.schemas import (
    ETIQUETAS,
    PeticionPrediccion,
    RespuestaPrediccion,
    ResultadoPrediccion,
)
from src.api.security import revisar_configuracion, verificar_api_key
from src.data.preprocessing import PreprocesadorTelefonos

# El modelo vive aquí y no en una global suelta: `lifespan` lo escribe al
# arrancar y los endpoints lo leen. Con `global` era fácil que un test dejara
# el módulo contaminado para el siguiente.
_estado: dict[str, Any] = {"modelo": None}


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    """Carga el modelo una sola vez, al arrancar el proceso.

    Reemplaza a `@app.on_event("startup")`, que FastAPI marca como obsoleto.

    :param app: la aplicacion que esta arrancando
    """
    _estado["modelo"] = load_model()
    metadatos = get_model_metadata()

    if _estado["modelo"] is not None:
        print(f"[API] Modelo cargado desde {metadatos['origen']} (versión {metadatos['version']}).")
    else:
        print("[API] Arranqué sin modelo: /health devolverá 503 hasta que lo haya.")

    aviso = revisar_configuracion()
    if aviso is not None:
        print(f"[API] AVISO: {aviso}")

    yield
    _estado.clear()


app = FastAPI(
    title="API de Clasificación de Precios de Teléfonos",
    description=(
        "API MLOps para predecir el rango de precio de un teléfono móvil según "
        "sus características técnicas. Sirve el modelo con alias "
        f"@{MODEL_ALIAS} del Model Registry de MLflow y, si el Registry no está "
        "disponible, cae al artefacto local."
    ),
    version="1.0.0",
    lifespan=ciclo_vida,
)


@app.get("/")
def raiz() -> dict[str, Any]:
    """Estado del servicio y que modelo esta sirviendo.

    Responde 200 aunque no haya modelo: para saber si esta listo, /health.

    :return: estado, nombre y alias del modelo, su origen y las features
    """
    return {
        "status": "online",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        **get_model_metadata(),
        "features_esperadas": FEATURE_COLUMNS,
    }


@app.get("/health")
def salud() -> dict[str, str]:
    """Readiness probe: 200 sólo si hay un modelo en memoria.

    Existe porque `GET /` respondía 200 incluso sin modelo cargado, así que no
    servía para el HEALTHCHECK del contenedor ni para que otros servicios
    esperaran a que la API estuviera realmente lista.

    :return: el estado, el origen del modelo y su version
    :raises HTTPException: 503 mientras no haya modelo cargado
    """
    if _estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    metadatos = get_model_metadata()
    return {"status": "ready", "origen": metadatos["origen"], "version": metadatos["version"]}


@app.post(
    "/predict",
    response_model=RespuestaPrediccion,
    dependencies=[Depends(verificar_api_key)],
)
def predecir(peticion: PeticionPrediccion) -> RespuestaPrediccion:
    """Predice el rango de precio de uno o varios telefonos.

    :param peticion: lote con las 20 caracteristicas de cada telefono
    :return: una prediccion por fila, con etiqueta y probabilidades
    :raises HTTPException: 503 sin modelo, 422 si la entrada no vale, 500 si
        falla la inferencia
    """
    modelo = _estado["modelo"]
    if modelo is None:
        # 503 y no 500: el servicio está sano, lo que falta es el modelo. Un
        # 500 le dice al balanceador que el proceso está roto.
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado. Ejecuta el entrenamiento y reinicia la API.",
        )

    try:
        crudo = pd.DataFrame([fila.model_dump() for fila in peticion.data])
        # Reutiliza la validación de columnas del pipeline batch en vez de
        # duplicar aquí el reordenado.
        x = PreprocesadorTelefonos.preparar_inferencia(crudo, FEATURE_COLUMNS)
        x = x.astype("float64")  # coincide con la firma registrada en MLflow

        predicciones = modelo.predict(x)
        proba = modelo.predict_proba(x) if hasattr(modelo, "predict_proba") else None
        clases = [int(c) for c in getattr(modelo, "classes_", [])]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Fallo en la inferencia: {error}") from error

    resultados = []
    for indice, prediccion in enumerate(predicciones):
        codigo = int(prediccion)
        detalle = None
        confianza = None

        if proba is not None:
            detalle = {
                ETIQUETAS.get(clase, str(clase)): round(float(valor), 4)
                for clase, valor in zip(clases, proba[indice])
            }
            confianza = round(float(max(proba[indice])) * 100, 2)

        # `confianza` se asigna sólo dentro del `if proba`, así que queda en
        # None cuando el modelo no expone predict_proba. Antes el código hacía
        # `if confidence else None`, que descartaba una confianza de 0.0 —
        # valor legítimo— y la convertía en ausencia de dato.
        resultados.append(
            ResultadoPrediccion(
                index=indice,
                price_range=codigo,
                etiqueta=ETIQUETAS.get(codigo, "desconocido"),
                confianza=confianza,
                probabilidades=detalle,
            )
        )

    return RespuestaPrediccion(
        model_metadata={"name": MODEL_NAME, "alias": MODEL_ALIAS, **get_model_metadata()},
        total=len(resultados),
        results=resultados,
    )
