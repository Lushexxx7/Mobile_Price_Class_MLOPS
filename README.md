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
- `notebooks/`: exploración, entrenamiento y predicción.
- `src/`: código reutilizable.
- `tests/`: pruebas unitarias.
- `models/`: artefactos de modelos entrenados (versionados con DVC).

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
