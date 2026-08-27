.PHONY: install test lint format train tune dvc-repro mlflow notebooks serve \
        docker-build docker-up docker-down docker-train docker-predict \
        docker-logs docker-ps docker-clean

install:
	pip install -r requirements.txt

test:
	pytest

# Revisa el estilo sin tocar nada.
lint:
	black --check src tests scripts main.py
	flake8 src tests scripts main.py

# Aplica el formato. Luego conviene un dvc status.
format:
	black src tests scripts main.py

train:
	python main.py

tune:
	python -m scripts.tune_hyperparameters

dvc-repro:
	dvc repro

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

notebooks:
	jupyter lab

serve:
	uvicorn src.api.app:app --reload --port 8000

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
	docker compose run --rm trainer python -m scripts.predict

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

# CUIDADO: -v borra los volumenes, es decir el historial de MLflow del stack.
docker-clean:
	docker compose down -v
