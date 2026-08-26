.PHONY: install test train tune dvc-repro mlflow notebooks

install:
	pip install -r requirements.txt

test:
	pytest

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
