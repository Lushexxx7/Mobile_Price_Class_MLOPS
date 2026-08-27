"""Esquemas de entrada y salida de la API de inferencia.

Estaban embebidos en el modulo de la app. Separarlos permite que los tests y
la documentacion los importen sin arrastrar el arranque de FastAPI, que carga
el modelo como efecto secundario.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Etiquetas legibles de price_range. El modelo predice 0-3.
ETIQUETAS: dict[int, str] = {
    0: "Precio Bajo",
    1: "Precio Medio",
    2: "Precio Alto",
    3: "Precio Muy Alto",
}


class CaracteristicasTelefono(BaseModel):
    """Las 20 caracteristicas tecnicas con las que se entreno el modelo."""

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


class PeticionPrediccion(BaseModel):
    """Lote de telefonos a predecir. Tiene que traer al menos uno.

    :ivar data: las filas del lote
    """

    # min_length=1: un lote vacio llegaba hasta el modelo y reventaba dentro de
    # sklearn con un error opaco. Ahora lo rechaza la validacion con un 422.
    data: list[CaracteristicasTelefono] = Field(..., min_length=1)


class ResultadoPrediccion(BaseModel):
    """Prediccion de una fila del lote.

    :ivar index: posicion de la fila dentro del lote enviado
    :ivar price_range: clase predicha, de 0 a 3
    :ivar etiqueta: esa misma clase en texto legible
    :ivar confianza: probabilidad de la clase ganadora, en porcentaje
    :ivar probabilidades: probabilidad de cada clase, None si el modelo no
        expone predict_proba
    """

    index: int
    price_range: int
    etiqueta: str
    confianza: float | None = None
    probabilidades: dict[str, float] | None = None


class RespuestaPrediccion(BaseModel):
    """Respuesta completa de /predict.

    :ivar model_metadata: que modelo respondio y de donde salio
    :ivar total: numero de filas predichas
    :ivar results: una entrada por fila
    """

    model_metadata: dict[str, str]
    total: int
    results: list[ResultadoPrediccion]
