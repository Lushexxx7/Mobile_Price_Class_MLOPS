"""Contratos de entrada y salida de la API (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# 0 = bajo | 1 = medio | 2 = alto | 3 = muy alto
ETIQUETAS: dict[int, str] = {
    0: "bajo",
    1: "medio",
    2: "alto",
    3: "muy_alto",
}


class TelefonoFeatures(BaseModel):
    """Las 20 variables de data/raw/train.csv, sin la columna objetivo."""

    model_config = ConfigDict(extra="forbid")

    battery_power: int = Field(..., ge=0, description="Capacidad de la batería en mAh")
    blue: int = Field(..., ge=0, le=1, description="Tiene Bluetooth")
    clock_speed: float = Field(..., gt=0, description="GHz del procesador")
    dual_sim: int = Field(..., ge=0, le=1)
    fc: int = Field(..., ge=0, description="Megapíxeles de la cámara frontal")
    four_g: int = Field(..., ge=0, le=1)
    int_memory: int = Field(..., ge=0, description="Memoria interna en GB")
    m_dep: float = Field(..., gt=0, description="Grosor en cm")
    mobile_wt: int = Field(..., gt=0, description="Peso en gramos")
    n_cores: int = Field(..., gt=0)
    pc: int = Field(..., ge=0, description="Megapíxeles de la cámara principal")
    px_height: int = Field(..., ge=0)
    px_width: int = Field(..., ge=0)
    ram: int = Field(..., ge=0, description="RAM en MB")
    sc_h: int = Field(..., ge=0, description="Alto de pantalla en cm")
    sc_w: int = Field(..., ge=0, description="Ancho de pantalla en cm")
    talk_time: int = Field(..., ge=0)
    three_g: int = Field(..., ge=0, le=1)
    touch_screen: int = Field(..., ge=0, le=1)
    wifi: int = Field(..., ge=0, le=1)


class PeticionPrediccion(BaseModel):
    data: list[TelefonoFeatures] = Field(..., min_length=1, max_length=1000)


class ResultadoPrediccion(BaseModel):
    index: int
    price_range: int
    etiqueta: str
    confianza: float | None = None
    probabilidades: dict[str, float] | None = None


class RespuestaPrediccion(BaseModel):
    model_metadata: dict[str, str]
    total: int
    results: list[ResultadoPrediccion]