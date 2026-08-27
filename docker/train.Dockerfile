# syntax=docker/dockerfile:1
#
# Pipeline de entrenamiento.
#
# Escribe el modelo y las metricas en directorios montados desde el host, de
# modo que DVC los sigue versionando desde fuera del contenedor.
#
# El pipeline fuerza LF en todas sus salidas, asi que las de texto
# (reports/*.json y data/processed/*.csv) salen identicas byte a byte a las
# que produce Windows: ahi no hay reproducciones espurias.
#
# El .pkl es otra historia. joblib/pickle no serializa igual byte a byte en
# Linux y en Windows -- mismo tamano, mismas metricas, hash distinto -- asi
# que tras entrenar en Docker `dvc status` marca `train` y `predict` como
# modificados aunque el modelo sea equivalente. No es un fallo de este
# Dockerfile ni de la configuracion: es como se comporta pickle entre
# plataformas. Para descartar el cambio: `dvc checkout models/modelo_final.pkl`.

# ------------------------------------------------------------ etapa: dependencias
FROM python:3.11-slim AS deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Solo los requirements en esta capa: asi queda cacheada y no se reinstala
# nada cuando cambia el codigo de src/.
COPY docker/requirements-train.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt

# ---------------------------------------------------------------- etapa: runtime
FROM python:3.11-slim

# El entorno virtual viaja ya resuelto; en la imagen final no queda cache de
# pip ni los ficheros de requirements.
COPY --from=deps /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py params.yaml ./

# Usuario sin privilegios: el proceso no corre como root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Sobrescribible para las demas etapas del pipeline, por ejemplo:
#   docker compose run --rm trainer python -m scripts.predict
CMD ["python", "main.py"]
