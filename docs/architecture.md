# Arquitectura

El proyecto adopta una estructura inspirada en Cookiecutter Data Science y
separa datos, notebooks, código reutilizable, scripts, modelos, pruebas,
referencias y reportes.

El flujo reproducible es:

```text
train.csv + test.csv
        |
        v
validación de datos
        |
        v
entrenamiento y evaluación
        |
        v
selección y persistencia
        |
        v
predicción por lotes
```

`PipelineTelefonos` orquesta las clases del dominio y mantiene separadas la
carga, preparación, evaluación, persistencia e inferencia del modelo.
