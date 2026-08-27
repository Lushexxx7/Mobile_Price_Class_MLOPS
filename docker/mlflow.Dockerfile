# syntax=docker/dockerfile:1
#
# Servidor de tracking y Model Registry de MLflow.
#
# Arranca con --serve-artifacts: con esa opcion el servidor hace de proxy de
# artefactos y entrega a los clientes URIs portables del tipo
# mlflow-artifacts:/..., en lugar de rutas absolutas del sistema de ficheros.
# Ese es justo el motivo por el que el historial local no sirve aqui: sus
# artefactos apuntan a C:/Users/... y esa ruta no existe en un contenedor.

FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# mlflow completo y no mlflow-skinny: el servidor necesita sqlalchemy y alembic
# para el backend SQL, que es requisito del Model Registry.
RUN pip install --no-cache-dir mlflow==3.15.1

# Los volumenes nombrados heredan el propietario que tenga el directorio en la
# imagen, asi que se crean antes de cambiar de usuario.
RUN useradd --create-home --uid 1000 mlflowuser \
    && mkdir -p /mlflow /mlartifacts \
    && chown -R mlflowuser:mlflowuser /mlflow /mlartifacts

USER mlflowuser
WORKDIR /mlflow

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=10 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status == 200 else 1)"

# --allowed-hosts: MLflow 3 valida la cabecera Host para frenar ataques de DNS
# rebinding, y por defecto solo admite localhost e IPs privadas. El nombre de
# servicio de la red de compose ("mlflow") no encaja ahi y la API recibia 403.
# Se enumeran los hosts legitimos en vez de abrir con "*".
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlartifacts", \
     "--allowed-hosts", "mlflow:5000,mlops-mlflow:5000,localhost:5000,127.0.0.1:5000"]
