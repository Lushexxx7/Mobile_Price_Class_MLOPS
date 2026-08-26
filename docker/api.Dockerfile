# syntax=docker/dockerfile:1
#
# API de inferencia (FastAPI + uvicorn).
#
# El modelo NO se hornea en la imagen: llega en tiempo de ejecucion desde el
# Model Registry de MLflow o, como respaldo, desde models/modelo_final.pkl
# montado como volumen. Asi la imagen no caduca cada vez que se reentrena.

# ------------------------------------------------------------ etapa: dependencias
FROM python:3.11-slim AS deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Se copian solo los requirements para que esta capa quede cacheada y no se
# reinstale nada cuando cambie el codigo de src/.
COPY docker/requirements-train.txt docker/requirements-api.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt

# ---------------------------------------------------------------- etapa: runtime
FROM python:3.11-slim

# El entorno virtual viaja ya resuelto; en la imagen final no queda ni pip cache
# ni los ficheros de requirements.
COPY --from=deps /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY src/ ./src/
COPY params.yaml ./

# Usuario sin privilegios: el proceso no corre como root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# /health devuelve 503 mientras no haya modelo cargado, asi que urlopen lanza
# excepcion y el contenedor se marca unhealthy. Se usa urllib para no meter
# curl en la imagen solo para esto.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
