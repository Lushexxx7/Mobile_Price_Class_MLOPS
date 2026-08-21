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

| Modelo              | Accuracy | Precision | Recall |    F1 |
| ------------------- | -------: | --------: | -----: | ----: |
| Regresión Logística |    0.965 |     0.965 |  0.965 | 0.965 |
| SVM                 |    0.890 |     0.890 |  0.890 | 0.890 |
| Random Forest       |    0.880 |     0.880 |  0.880 | 0.880 |

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

Los datos originales (`train.csv`, `test.csv`) y el modelo entrenado
(`modelo_final.pkl`) **no viven en Git**: se versionan con DVC y se
descargan desde Google Drive. Después de clonar el repositorio, sigue la
sección [Versionado de datos con DVC](#versionado-de-datos-con-dvc) antes de
ejecutar el proyecto.

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

- `data/raw/`: datasets originales (versionados con DVC, no con Git).
- `data/processed/`: futuros datos transformados.
- `data/raw/`: datasets originales.
- `data/external/`: fuentes externas sin modificar.
- `data/interim/`: datos intermedios.
- `data/processed/`: predicciones y salidas procesadas.
- `notebooks/`: exploración, entrenamiento y predicción.
- `src/data/`: carga, validación y preprocesamiento de datos.
- `src/features/`: construcción de características.
- `src/models/`: entrenamiento, evaluación, pipeline y predicción.
- `tests/`: pruebas unitarias.
- `models/`: artefactos de modelos entrenados (versionados con DVC).
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

- **GitHub** guarda nuestro código y los archivos `.dvc`.
- **DVC** se encarga de controlar las versiones de los datos y del modelo.
- **Google Drive** almacena los archivos reales.

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

- GitHub recibe los archivos `.dvc`.
- Google Drive recibe los archivos reales.

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

## Alcance MLOps

Este proyecto usa [DVC](https://dvc.org) para versionar los datasets
(`data/raw/train.csv`, `data/raw/test.csv`) y el modelo final
(`models/modelo_final.pkl`). Git solo guarda el código y los archivos
puntero `.dvc`; el contenido real pesado vive en un remote de Google Drive.

### 1. Instalar DVC

```powershell
pip install "dvc[gdrive]"
```

(ya incluido en `requirements.txt`)

### 2. Clonar el repositorio y descargar los datos

```powershell
git clone https://github.com/Lushexxx7/Mobile_Price_Class_MLOPS.git
cd Mobile_Price_Class_MLOPS
```

Configura las credenciales de OAuth de forma **local** (nunca se suben a
Git, se guardan en `.dvc/config.local`):

```powershell
dvc remote modify --local gdrive_remote gdrive_client_id 'TU_CLIENT_ID'
dvc remote modify --local gdrive_remote gdrive_client_secret 'TU_CLIENT_SECRET'
```

Pide el `client_id` y `client_secret` a quien administra el proyecto de
Google Cloud (`DVC-MobilePrice`). También necesitas:

- Ser agregado como colaborador de la carpeta de Google Drive del remote.
- Ser agregado como **usuario de prueba** en la pantalla de consentimiento
  OAuth del proyecto de Google Cloud (si no, Google bloqueará el acceso).

Luego descarga los datos y el modelo reales:

```powershell
dvc pull
```

La primera vez se abrirá el navegador para autenticarte con tu cuenta de
Google.

### 3. Flujo de trabajo diario

Después de modificar o regenerar datasets/modelos:

```powershell
dvc add data/raw/train.csv        # o el archivo que hayas cambiado
git add data/raw/train.csv.dvc
git commit -m "feat(data): update train.csv"
dvc push                          # sube el archivo real a Drive
git push                          # sube el puntero .dvc a GitHub
```

Para volver a un estado anterior:

```powershell
git checkout <commit-o-rama>
dvc checkout
```

### Notas de seguridad

- `.dvc/config` (versionado en Git) solo contiene la URL del remote — **sin
  secretos**.
- `.dvc/config.local` guarda las credenciales de OAuth y está en
  `.gitignore` por defecto. Nunca lo subas manualmente ni cambies eso.
- Si alguna vez ves un secreto en un commit antes de hacer `git push`,
  detente y reescribe el historial (`git reset --soft` al commit anterior)
  en lugar de solo corregirlo en un commit nuevo — GitHub revisa todos los
  commits del push, no solo el estado final.
