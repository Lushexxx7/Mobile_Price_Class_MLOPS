.PHONY: install test train repro push pull notebooks mlflow-ui serve \n        docker-build docker-up docker-down docker-train docker-predict \n        docker-logs docker-ps docker-clean

install:
	pip install -r requirements.txt

test:
	pytest

train: repro

repro:
	dvc repro

push:
	dvc push

pull:
	dvc pull

notebooks:
	jupyter notebook

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------------- Docker
# Stack completo: mlflow (tracking + registry), trainer (a demanda) y api.

docker-build:
	docker compose build
	docker compose build trainer

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Entrena dentro del contenedor y reinicia la api para que recargue el campeon
docker-train:
	docker compose run --rm trainer
	docker compose restart api

docker-predict:
	docker compose run --rm trainer python -m src.models.predict_batch

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

# CUIDADO: -v borra los volumenes, es decir el historial de MLflow del stack.
docker-clean:
	docker compose down -v
