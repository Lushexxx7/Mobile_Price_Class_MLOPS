.PHONY: install test train repro push pull notebooks

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