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

DVC conecta estas responsabilidades mediante las etapas `validate`, `train` y
`predict` declaradas en `dvc.yaml`. Git conserva código y metadatos; Google
Drive conserva los datos y artefactos versionados.
