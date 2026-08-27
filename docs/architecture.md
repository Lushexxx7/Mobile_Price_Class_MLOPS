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
entrenamiento y evaluación ------> MLflow: runs, métricas y Model Registry
        |                                          |
        v                                          | alias @champion
selección y persistencia                           |
        |                                          v
        +----> predicción por lotes        API de inferencia
```

`PipelineTelefonos` orquesta las clases del dominio y mantiene separadas la
carga, preparación, evaluación, persistencia e inferencia del modelo.

DVC conecta estas responsabilidades mediante las etapas `validate`, `train`,
`predict` y `tune` declaradas en `dvc.yaml`. Git conserva código y metadatos;
Google Drive conserva los datos y artefactos versionados.

Cada etapa declara como dependencia las subrutas de `src/` que realmente la
afectan, y no el directorio completo. Es deliberado: la API consume el
artefacto pero no participa en producirlo, así que publicar un endpoint no
debe invalidar el modelo entrenado.

## Serving

`src/api/` expone el modelo por HTTP y se reparte en tres piezas: `app.py` con
los endpoints, `schemas.py` con los contratos de entrada y salida, y
`security.py` con la autenticación opcional por API key. `model_loader.py`
resuelve de dónde sale el modelo.

Esa resolución tiene dos niveles a propósito. Primero pide al Model Registry la
versión con alias `champion`, de modo que la API sirve siempre el último modelo
promovido sin tocar código. Si el Registry no responde, cae al `.pkl` local. La
alternativa —fijar un número de versión— obligaría a un despliegue por cada
reentrenamiento.

## Contenedores

```text
        +---------------------+
        |  mlflow             |  tracking + Model Registry
        |  :5000              |  volúmenes: mlflow_db, mlflow_artifacts
        +---------------------+
           ^               ^
           | registra      | lee @champion
           |               |
   +---------------+   +---------------+
   |  trainer      |   |  api          |  :8000
   |  (a demanda)  |   |  (permanente) |
   +---------------+   +---------------+
           |                   |
           | bind-mount        | bind-mount solo lectura
           v                   v
      ./data ./models ./reports      ./models
```

El modelo no se hornea en las imágenes: llega en tiempo de ejecución desde el
Registry o desde el volumen. Así una imagen construida hoy sigue sirviendo el
modelo que se entrene mañana.

El servidor de MLflow arranca con `--serve-artifacts`, lo que hace que entregue
URIs portables del tipo `mlflow-artifacts:/...` en lugar de rutas absolutas del
sistema de ficheros. Sin eso, los artefactos registrados desde Windows apuntan
a `C:/Users/...`, una ruta que no existe dentro del contenedor.

Las salidas del entrenamiento se escriben en el host por bind-mount para que
DVC las siga versionando desde fuera. Eso exige que los hashes coincidan entre
sistemas, y por eso el pipeline fuerza LF en todas sus salidas de texto: sin
ello, reproducir en Linux invalidaba lo reproducido en Windows y viceversa, en
bucle.
