"""Autenticacion opcional por API key para los endpoints de inferencia."""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# Cadena vacia o variable ausente = autenticacion desactivada. Es lo que
# queremos en local y en pytest; en el contenedor se inyecta desde .env.
#
# El `.strip()` no es cosmetico: un `.env` guardado en CRLF le pega un `\r`
# invisible al valor, y docker compose lo inyecta tal cual. La comparacion
# fallaba entonces contra una clave que era imposible de teclear.
API_KEY = os.getenv("API_KEY", "").strip()

_cabecera = APIKeyHeader(name="X-API-Key", auto_error=False)


def revisar_configuracion() -> str | None:
    """Devuelve un aviso si la API key configurada tiene mala pinta.

    No corrige nada ni aborta el arranque: solo describe el problema para que
    salga en el log. Un 401 sin contexto es carisimo de diagnosticar; este
    aviso convierte media hora de curl a ciegas en una linea leible.

    :return: el texto del aviso, o None si la clave no tiene mala pinta
    """
    if not API_KEY:
        return None

    if API_KEY.startswith("<") and API_KEY.endswith(">"):
        return (
            "API_KEY conserva los angulos < > del generador. Se esta usando "
            "literalmente, angulos incluidos: quitalos de .env y recrea el "
            "contenedor con `docker compose up -d --force-recreate api`."
        )

    if any(caracter.isspace() for caracter in API_KEY):
        return (
            "API_KEY contiene espacios o saltos de linea en medio. La cabecera "
            "X-API-Key tendria que reproducirlos exactamente para dar 200."
        )

    return None


def verificar_api_key(clave: str | None = Security(_cabecera)) -> None:
    """Deja pasar si no hay API_KEY configurada; si la hay, la exige.

    `compare_digest` compara en tiempo constante: evita que alguien deduzca la
    clave caracter a caracter midiendo cuanto tarda el rechazo.

    :param clave: valor de la cabecera X-API-Key, o None si no vino
    :raises HTTPException: 401 si hay clave configurada y no coincide
    """
    if not API_KEY:
        return

    if clave is None or not secrets.compare_digest(clave, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key invalida o ausente (cabecera X-API-Key).",
            headers={"WWW-Authenticate": "ApiKey"},
        )
