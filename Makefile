.PHONY: install test train notebooks

install:
	pip install -r requirements.txt

test:
	pytest

train:
	python main.py

notebooks:
	jupyter notebook
