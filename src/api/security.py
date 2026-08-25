"""Autenticación por API key para los endpoints de inferencia."""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# Cadena vacía = autenticación desactivada (desarrollo local y pytest).
API_KEY = os.getenv("API_KEY", "")

_cabecera = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar_api_key(clave: str | None = Security(_cabecera)) -> None:
    """Deja pasar si no hay API_KEY configurada; si la hay, la exige.

    `compare_digest` compara en tiempo constante: evita que un atacante
    deduzca la clave midiendo cuánto tarda el rechazo.
    """
    if not API_KEY:
        return
    if clave is None or not secrets.compare_digest(clave, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente (cabecera X-API-Key).",
            headers={"WWW-Authenticate": "ApiKey"},
        )