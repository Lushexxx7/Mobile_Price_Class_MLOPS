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

## Arquitectura de contenedores

El mismo flujo, empaquetado en tres servicios de `docker-compose.yml`:

```text
   ┌──────────────────────────────────────────────────────────────┐
   │                     red de docker compose                    │
   │                                                              │
   │   ┌───────────┐   registra corridas   ┌──────────────────┐   │
   │   │  trainer  │──────────────────────▶│      mlflow      │   │
   │   │ main.py   │   promueve @champion  │  servidor :5000  │   │
   │   └─────┬─────┘                       │  sqlite + proxy  │   │
   │         │                             │  de artefactos   │   │
   │         │                             └────────▲─────────┘   │
   │         │                                      │             │
   │         │                     models:/...@champion           │
   │         │                                      │             │
   │         │                             ┌────────┴─────────┐   │
   │         │                             │       api        │   │
   │         │                             │  FastAPI :8000   │   │
   │         │                             └──────────────────┘   │
   └─────────┼──────────────────────────────────────┼─────────────┘
             │                                      │
             ▼                                      ▼
      data/  models/  metrics/  plots/        models/ (solo lectura)
             (volumenes del host, versionados por DVC)
```

`mlflow` guarda su base y sus artefactos en volúmenes propios, no en el
sistema de ficheros del host, y los sirve a través de su propio proxy: los
clientes reciben URIs `mlflow-artifacts:/` en lugar de rutas absolutas, que
son las que impedían cargar un modelo registrado desde otra máquina o desde
otro sistema operativo.

`trainer` no arranca con `up`; queda bajo un perfil de compose y se invoca a
demanda. Escribe sus salidas en carpetas montadas desde el host, de modo que
DVC las sigue versionando desde fuera del contenedor.

`api` carga el modelo una sola vez al arrancar y en dos niveles: primero el
Model Registry y, si falla, el artefacto local de DVC. Nunca lanza excepción
durante el arranque, así que un fallo de MLflow degrada el servicio en lugar
de tumbarlo.
