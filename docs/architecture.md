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

`PipelineTelefonos` orquesta las clases del dominio. DVC describe las
dependencias y salidas de cada etapa en `dvc.yaml`; Git conserva el código y
los metadatos, mientras el almacenamiento DVC conserva los datos y artefactos.
