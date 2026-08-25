.PHONY: install test train repro push pull notebooks mlflow-ui serve

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