# Clasificación de precios de teléfonos

## Integrantes

1. Canaviri Yanahuaya Sergio Alexander
2. Hualca Yavi Lizbeth
3. Sanchez Calle Maria Yesica
4. Sotillo Sanchez Luis Antonio

Proyecto MLOps para clasificar teléfonos en cuatro rangos de precio
(`price_range`: 0, 1, 2 o 3) a partir de 20 características técnicas. Integra
una arquitectura orientada a objetos, pruebas automatizadas, DVC para datos y
artefactos, y MLflow para experimentos y registro de modelos.

## Modelos

El proyecto utiliza una interfaz común para comparar:

- Regresión Logística como modelo base.
- Random Forest para modelos de ensamble e importancia de características.
- SVM como alternativa no lineal.

La selección se realiza por `accuracy` sobre una división estratificada 80/20
con `random_state=42`. Los parámetros se administran desde `params.yaml`.

## Estructura

```text
data/                 Datos raw, interim, external y processed
docs/                 Documentación de arquitectura
models/               Modelo final administrado por DVC
notebooks/            EDA, entrenamiento y predicción
references/           Documentación del dataset
reports/              Métricas, validaciones y figuras
scripts/              Validación, predicción y búsqueda de hiperparámetros
src/data/             Carga y preprocesamiento
src/features/         Construcción de características
src/models/           Modelos, evaluación, pipeline y tracking
tests/                Pruebas automatizadas
```

Esta distribución sigue una variante de Cookiecutter Data Science.

## Instalación

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## DVC y Google Drive

Git conserva el código y los archivos puntero `.dvc`. Los datasets, el modelo
final y las predicciones se conservan en la caché DVC y en el remoto
`gdrive_remote` de Google Drive.

Las credenciales OAuth deben estar únicamente en `.dvc/config.local`, archivo
ignorado por Git. Para configurarlas en una instalación nueva:

```powershell
dvc remote modify --local gdrive_remote gdrive_client_id "TU_CLIENT_ID"
dvc remote modify --local gdrive_remote gdrive_client_secret "TU_CLIENT_SECRET"
dvc pull
```

Comandos principales:

```powershell
dvc pull          # descargar datos y artefactos
dvc repro         # reproducir las etapas modificadas
dvc metrics show  # consultar métricas
dvc status        # revisar el estado local
dvc status -c     # comparar caché local y remoto
dvc push          # subir artefactos nuevos a Google Drive
```

El pipeline definido en `dvc.yaml` contiene:

1. `validate`: valida `train.csv` y `test.csv`.
2. `train`: compara los modelos base, selecciona y persiste el ganador.
3. `predict`: genera las predicciones del conjunto de prueba.
4. `tune`: ejecuta la búsqueda de hiperparámetros con MLflow.

## MLflow

La implementación central está en `src/models/tracking.py`. Utiliza SQLite
local (`mlflow.db`) y el experimento `telefonos_price_classification`.

Cada ejecución hija registra:

- Parámetros y métricas `accuracy`, `precision`, `recall` y `f1`.
- Matriz de confusión en JSON y PNG.
- Reporte de clasificación.
- Modelo, firma y ejemplo de entrada.
- Rama y commit de Git, además de hashes DVC de los datasets.

Cada ejecución padre registra una tabla y una gráfica comparativa. La etapa
`tune` realiza exactamente dos entrenamientos de Regresión Logística, dos de
Random Forest y dos de SVM. El ganador base recibe el alias `champion` y el
ganador de la búsqueda recibe `challenger` en el modelo registrado
`TelefonosPriceClassifier`.

Ejecutar únicamente los seis ensayos:

```powershell
python -m scripts.tune_hyperparameters
```

Iniciar la interfaz web:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir `http://127.0.0.1:5000` y seleccionar **Entrenamiento de modelos**. La
base `mlflow.db`, `mlruns/` y `mlartifacts/` son locales y están ignorados por
Git.

## Ejecución

Flujo reproducible completo:

```powershell
dvc pull
dvc repro
dvc metrics show
```

Ejecución directa del entrenamiento base:

```powershell
python main.py
```

Notebooks, en orden:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_entrenamiento.ipynb`
3. `notebooks/03_prediccion.ipynb`

## Pruebas

```powershell
pytest
```

## Seguridad

- No subir `.dvc/config.local`, credenciales OAuth ni secretos.
- No agregar directamente a Git los CSV, modelos, `mlflow.db` o artefactos.
- Actualizar los datos con `dvc add`, confirmar el puntero `.dvc` en Git y
  ejecutar `dvc push`.
