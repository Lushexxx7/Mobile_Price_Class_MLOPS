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
- `data/processed/`: futuros datos transformados.
- `notebooks/`: exploración, entrenamiento y predicción.
- `src/`: código reutilizable.
- `tests/`: pruebas unitarias.
- `models/`: artefactos de modelos entrenados.

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

## Alcance MLOps

La base actual cubre reproducibilidad, modularidad, pruebas y persistencia del
modelo. Una fase posterior puede incorporar seguimiento de experimentos con
MLflow, una API, Docker, integración continua y monitoreo sin modificar las
interfaces centrales.
