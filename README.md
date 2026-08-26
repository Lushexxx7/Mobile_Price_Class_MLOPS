# Mobile_Price_Classification_Project

# Integrantes

1. **Canaviri Yanahuaya Sergio Alexander**
2. **Hualca Yavi Lizbeth**
3. **Sanchez Calle Maria Yesica**
4. **Sotillo Sanchez Luis Antonio**

# Clasificación de precios de teléfonos

Proyecto educativo de Machine Learning para clasificar teléfonos en cuatro
rangos de precio (`price_range`: 0, 1, 2 o 3) a partir de 20 características
técnicas. Implementa un flujo reproducible orientado a objetos que carga,
preprocesa, entrena, compara, selecciona, persiste y reutiliza modelos.

En un mercado móvil competitivo, clasificar correctamente un dispositivo por
su rango de precio puede apoyar las decisiones de lanzamiento de fabricantes y
empresas emergentes. El sistema analiza características como RAM, batería,
memoria interna y resolución; no intenta predecir un precio monetario exacto.

## Objetivo

Construir una solución reproducible para clasificar teléfonos en cuatro rangos
de precio a partir de 20 características técnicas. El proyecto aplica una
estructura clásica de Cookiecutter Data Science, programación orientada a
objetos y pruebas automatizadas.

## Modelos y resultados

Se comparan tres clasificadores mediante la misma interfaz polimórfica:

| Modelo                | Accuracy | Precision | Recall |    F1 |
| --------------------- | -------: | --------: | -----: | ----: |
| Regresión Logística |    0.965 |     0.965 |  0.965 | 0.965 |
| SVM                   |    0.890 |     0.890 |  0.890 | 0.890 |
| Random Forest         |    0.880 |     0.880 |  0.880 | 0.880 |

La selección se realiza por `accuracy`. En la ejecución actual ganó Regresión
Logística. Random Forest se conserva para analizar la importancia de las
características y SVM como alternativa no lineal. Las métricas provienen de una
división estratificada 80/20 con `random_state=42`.

## Preparación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Los datos originales deben permanecer en `data/raw/train.csv` y
`data/raw/test.csv`.

## Ejecución

Ejecutar el flujo completo desde la terminal:

```powershell
python main.py
```

Ejecutar los notebooks en orden:

```powershell
jupyter notebook
```

1. `notebooks/01_eda.ipynb`: exploración y calidad de datos.
2. `notebooks/02_entrenamiento.ipynb`: comparación y persistencia.
3. `notebooks/03_prediccion.ipynb`: inferencia sin reentrenamiento.

Ejecutar las pruebas:

```powershell
pytest
```

## Estructura

- `data/raw/`: datasets originales.
- `data/external/`: fuentes externas sin modificar.
- `data/interim/`: datos intermedios.
- `data/processed/`: predicciones y salidas procesadas.
- `notebooks/`: exploración, entrenamiento y predicción.
- `src/data/`: carga, validación y preprocesamiento de datos.
- `src/features/`: construcción de características.
- `src/models/`: entrenamiento, evaluación, pipeline y predicción.
- `tests/`: pruebas unitarias.
- `models/`: artefactos de modelos entrenados.
- `reports/`: validaciones, informes y figuras.
- `references/`: diccionarios y documentación de datos.
- `docs/`: documentación adicional.

Esta distribución sigue la variante clásica de Cookiecutter Data Science y
conserva la arquitectura POO implementada. `Makefile`, `LICENSE`,
`pyproject.toml`, `docs`, `references` y `reports/figures` completan los
componentes estándar del proyecto.

## Diseño orientado a objetos

- `CargadorDatos`: carga y validación básica de CSV.
- `PreprocesadorTelefonos`: separación, división estratificada e inferencia.
- `ModeloClasificacion`: interfaz común para entrenar y predecir.
- `ModeloRegresionLogistica`, `ModeloRandomForest` y `ModeloSVM`: clases
  especializadas que demuestran herencia y polimorfismo.
- `EvaluadorModelo`: métricas y diagnósticos homogéneos.
- `PipelineTelefonos`: orquestación, selección, persistencia e inferencia.

El artefacto `models/modelo_final.pkl` incluye el estimador ganador, las
columnas esperadas, la variable objetivo, la métrica de selección y los
resultados de comparación. Las predicciones se escriben en
`data/processed/predicciones.csv`.

## Versionado de datos con DVC

Para este proyecto usamos **DVC (Data Version Control)** porque los datasets y el modelo entrenado son archivos pesados y no queremos guardarlos directamente en GitHub.

La idea es sencilla:

* **GitHub** guarda nuestro código y los archivos `.dvc`.
* **DVC** se encarga de controlar las versiones de los datos y del modelo.
* **Google Drive** almacena los archivos reales.

De esta manera podemos trabajar con Git y DVC juntos sin tener que subir los archivos pesados al repositorio.

### 1. Instalar DVC

Primero instalamos DVC con soporte para Google Drive:

```powershell
pip install "dvc[gdrive]"
```

Esta dependencia también está incluida en `requirements.txt`.

### 2. Inicializar DVC

Dentro del proyecto inicializamos DVC:

```powershell
dvc init
```

Esto crea la carpeta `.dvc`, donde DVC guarda su configuración.

Después hacemos el commit de esta configuración:

```powershell
git add .dvc .gitignore
git commit -m "chore: initialize DVC"
```

### 3. Configurar Google Drive

Creamos un remote de DVC para utilizar Google Drive como almacenamiento:

```powershell
dvc remote add -d gdrive_remote "ID_DE_LA_CARPETA_DE_GOOGLE_DRIVE"
```

El `ID_DE_LA_CARPETA_DE_GOOGLE_DRIVE` corresponde a la carpeta donde DVC almacenará los archivos.

La configuración del remote queda en `.dvc/config`. Esta información no contiene nuestras credenciales personales.

### 4. Configurar las credenciales de Google

Para poder conectarnos a Google Drive configuramos las credenciales de OAuth de manera local:

```powershell
dvc remote modify --local gdrive_remote gdrive_client_id "TU_CLIENT_ID"
dvc remote modify --local gdrive_remote gdrive_client_secret "TU_CLIENT_SECRET"
```

La opción `--local` es importante porque hace que estas credenciales se guarden en:

```text
.dvc/config.local
```

Este archivo **no se sube a GitHub**.

Para utilizar el remote también es necesario tener acceso a la carpeta de Google Drive y, dependiendo de la configuración del proyecto de Google Cloud, estar agregado como usuario de prueba de OAuth.

### 5. Agregar los datos a DVC

Los archivos que queremos versionar con DVC son:

```text
data/raw/train.csv
data/raw/test.csv
models/modelo_final.pkl
```

Para agregarlos utilizamos:

```powershell
dvc add data/raw/train.csv
dvc add data/raw/test.csv
dvc add models/modelo_final.pkl
```

Por ejemplo, al ejecutar:

```powershell
dvc add data/raw/train.csv
```

DVC deja de necesitar que Git controle directamente el archivo y crea un archivo:

```text
data/raw/train.csv.dvc
```

El archivo `.dvc` funciona como un pequeño puntero que le indica a DVC dónde encontrar la versión correspondiente del dataset.

> **Importante:** los archivos originales no deben estar siendo rastreados directamente por Git. Si Git ya los estaba siguiendo, primero hay que quitarlos del seguimiento de Git:

```powershell
git rm --cached data/raw/train.csv
git rm --cached data/raw/test.csv
git rm --cached models/modelo_final.pkl
```

Después hacemos el commit:

```powershell
git add .
git commit -m "chore: move data and model tracking to DVC"
```

### 6. Subir los archivos a Google Drive

Una vez que DVC está configurado, subimos los archivos reales al remote:

```powershell
dvc push
```

Aquí ocurre algo importante:

* GitHub recibe los archivos `.dvc`.
* Google Drive recibe los archivos reales.

Por eso los datasets no aparecen directamente dentro del repositorio de GitHub.

Finalmente subimos los cambios de Git:

```powershell
git push
```

### 7. Descargar los datos después de clonar el proyecto

Cuando otra persona clona el proyecto, los archivos grandes no se descargan automáticamente porque no están almacenados en GitHub.

Primero clona el repositorio:

```powershell
git clone https://github.com/Lushexxx7/Mobile_Price_Class_MLOPS.git
cd Mobile_Price_Class_MLOPS
```

Después configura las credenciales locales de Google Drive y ejecuta:

```powershell
dvc pull
```

DVC se encarga de buscar los archivos en Google Drive y descargarlos en las ubicaciones correspondientes.

La primera vez puede abrirse el navegador para iniciar sesión con Google y autorizar el acceso.

### 8. ¿Qué hacemos cuando modificamos un dataset?

Este es el flujo que utilizamos normalmente cuando cambiamos un archivo:

```powershell
dvc add data/raw/train.csv
```

DVC detecta la nueva versión del archivo y actualiza:

```text
data/raw/train.csv.dvc
```

Después guardamos el nuevo puntero en Git:

```powershell
git add data/raw/train.csv.dvc
git commit -m "feat(data): update train dataset"
git push
```

Y subimos la nueva versión real del archivo a Google Drive:

```powershell
dvc push
```

En resumen, el flujo es:

```text
Modificar datos
      ↓
    dvc add
      ↓
Actualizar archivo .dvc
      ↓
  git commit
      ↓
   git push
      ↓
   dvc push
      ↓
Google Drive guarda la nueva versión
```

### 9. Volver a una versión anterior

Una de las ventajas de utilizar DVC es que podemos recuperar una versión anterior de nuestros datos.

Primero cambiamos a un commit anterior de Git:

```powershell
git checkout <commit>
```

Después ejecutamos:

```powershell
dvc checkout
```

DVC revisa el archivo `.dvc` correspondiente a esa versión y recupera los datos correctos.

### 10. ¿Por qué usamos DVC en este proyecto?

Usamos DVC porque nuestro proyecto no solamente tiene código. También tenemos datasets y un modelo entrenado que necesitamos conservar y versionar.

Podemos saber **qué versión de los datos corresponde a cada versión del código**, descargar nuevamente los archivos cuando sea necesario y evitar llenar el repositorio de GitHub con archivos pesados.

### Resumen del flujo

Los comandos que usamos principalmente son:

```powershell
# Agregar o actualizar un archivo
dvc add data/raw/train.csv

# Subir los datos al almacenamiento remoto
dvc push

# Descargar los datos
dvc pull

# Recuperar la versión de datos correspondiente al commit actual
dvc checkout

# Guardar el puntero de DVC en Git
git add .
git commit -m "update data"
git push
```

**En pocas palabras:** Git controla el código y la versión del proyecto, mientras que DVC controla las versiones de los datos y del modelo, utilizando Google Drive para almacenar los archivos reales.

## Project Organization

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         classification_model and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── classification_model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes classification_model a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling        
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models  
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```


## Autenticación de la API

El endpoint `/predict` acepta una API key opcional por cabecera. Se activa
definiendo la variable de entorno `API_KEY`:

```powershell
$env:API_KEY = "tu-clave"
uvicorn src.api.app:app --port 8000
```

## Ejecución con Docker

El proyecto se levanta completo en contenedores: servidor de MLflow,
entrenamiento y API de inferencia.

| Servicio  | Definición                  | Puerto | Función                                  |
| --------- | --------------------------- | -----: | ---------------------------------------- |
| `mlflow`  | `docker/mlflow.Dockerfile`  |   5000 | Tracking y Model Registry                |
| `trainer` | `docker/train.Dockerfile`   |      — | Ejecuta el pipeline; se invoca a demanda |
| `api`     | `docker/api.Dockerfile`     |   8000 | Sirve el modelo campeón por HTTP         |

### Puesta en marcha

Requisito: Docker Desktop en marcha.

1. Crear el fichero de variables de entorno a partir de la plantilla:

```bash
cp .env.example .env
```

Editar `.env` y poner una `API_KEY`. Si se deja vacía, `/predict` queda sin
autenticación.

2. Recuperar los datos. No viajan dentro de la imagen: el contenedor de
   entrenamiento los recibe por volumen desde `data/`.

```bash
dvc pull
```

3. Construir las imágenes y levantar el stack:

```bash
docker compose build && docker compose build trainer
```

```bash
docker compose up -d
```

Con esto quedan en marcha `mlflow` en el puerto 5000 y `api` en el 8000. La
API arranca aunque todavía no exista ningún modelo: cae al artefacto local de
DVC y, si tampoco lo hay, `/health` responde 503 sin que el contenedor entre
en un bucle de reinicios.

4. Entrenar dentro del contenedor y recargar la API:

```bash
docker compose run --rm trainer
```

```bash
docker compose restart api
```

El entrenamiento registra los tres modelos candidatos en MLflow, promueve al
ganador con el alias `@champion` y escribe el artefacto, las métricas y los
plots en las carpetas del host.

### Comprobación

```bash
curl -s localhost:8000/
```

La respuesta debe incluir `"origen": "registry@champion"`. Si en su lugar
aparece `"origen": "artefacto_local"`, la API no encontró el modelo en el
Registry y está sirviendo el respaldo: revisar que el paso 4 terminó bien y
reiniciar la API.

Para predecir:

```bash
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -H "X-API-Key: TU_CLAVE" -d @tests/payload_ejemplo.json
```

La interfaz de MLflow queda en `http://localhost:5000`.

### Otros comandos

```bash
docker compose run --rm trainer python -m src.models.predict_batch
```

```bash
docker compose logs -f
```

```bash
docker compose down
```

Si además se quieren borrar los volúmenes, es decir el historial de MLflow del
stack:

```bash
docker compose down -v
```

El `Makefile` recoge estos mismos comandos como atajos (`docker-build`,
`docker-up`, `docker-train`, `docker-down`, `docker-clean`) para quien tenga
`make` instalado.

### Notas de diseño

**El modelo no se hornea en la imagen.** Llega en tiempo de ejecución desde el
Model Registry o, como respaldo, desde `models/modelo_final.pkl` montado como
volumen. Así reentrenar no obliga a reconstruir la imagen.

**El stack usa su propia base de MLflow.** El historial local de `mlflow.db`
guarda rutas absolutas del estilo `file:///C:/Users/...` que no existen dentro
de un contenedor Linux, de modo que sus modelos no se pueden cargar desde ahí.
El servidor del stack arranca con `--serve-artifacts`, que entrega URIs
portables `mlflow-artifacts:/`, y guarda todo en volúmenes propios. El
historial de la máquina local no se toca ni se borra.

**Entrenar en Windows y en Linux no da un artefacto idéntico.** Las métricas,
los plots y el `eval.json` sí coinciden byte a byte, pero los coeficientes del
modelo difieren en torno a 5e-15 porque las ruedas de numpy para cada
plataforma traen versiones distintas de BLAS. El modelo es equivalente y las
predicciones son las mismas, pero `dvc status` marcará `modelo_final.pkl` como
modificado si se entrena alternando plataforma. Conviene fijar un entorno de
entrenamiento canónico y usar siempre ese para generar el `dvc.lock` que se
commitea.
