# syntax=docker/dockerfile:1
#
# Pipeline de entrenamiento.
#
# Escribe el modelo, las metricas y los plots en directorios montados desde el
# host, de modo que DVC sigue versionando las salidas desde fuera del
# contenedor. Gracias al fin de linea LF forzado en el pipeline, los md5 que
# produce aqui son identicos a los que produce el host: `dvc status` sigue
# limpio despues de entrenar en Docker.

FROM python:3.11-slim AS deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY docker/requirements-train.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt

FROM python:3.11-slim

COPY --from=deps /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY src/ ./src/
COPY main.py params.yaml ./

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Sobrescribible: `docker compose run --rm trainer python -m src.models.predict_batch`
CMD ["python", "main.py"]
