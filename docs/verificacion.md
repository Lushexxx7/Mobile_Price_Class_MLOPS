# Guía de verificación end-to-end

Cómo comprobar, desde cero y en orden, que las cuatro piezas del proyecto
funcionan: **DVC** (datos y pipeline), **MLflow** (tracking y registry),
**FastAPI** (serving) y **Docker** (todo empaquetado).

Cada bloque lleva el comando exacto, qué hace por dentro y qué tienes que ver
en pantalla para dar el paso por bueno. Si un paso falla, no sigas al
siguiente: cada nivel asume que el anterior quedó verde.

Los comandos son de PowerShell, ejecutados desde la raíz del repositorio.

---

## Nivel 0 — Entorno

### 0.1 Situarte en la rama y crear el entorno

```powershell
git status
```

Confirma en qué rama estás y que no hay cambios sin guardar. Todo lo de esta
guía se verifica sobre `main`.

```powershell
python -m venv .venv
```

Crea un intérprete de Python aislado dentro de `.venv/`. Aísla las
dependencias del proyecto de las que tengas instaladas en el sistema: si otro
trabajo tuyo usa scikit-learn 1.5, aquí puede convivir con la 1.9 que este
repo necesita.

```powershell
.\.venv\Scripts\Activate.ps1
```

Activa el entorno: a partir de aquí `python`, `pip`, `pytest`, `dvc` y
`mlflow` son los de `.venv`. Lo sabes porque el prompt pasa a empezar por
`(.venv)`. Si PowerShell bloquea el script, ejecuta una vez
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

```powershell
pip install -r requirements.txt
```

Instala las dependencias directas del proyecto. Las versiones están fijadas a
propósito: el modelo se serializa con `joblib`, y ese formato es sensible a la
versión de scikit-learn con la que se escribió. Si cada uno instala la suya,
el `.pkl` de uno no se abre en la máquina del otro.

### 0.2 Comprobar que las herramientas responden

```powershell
python -V; dvc version; mlflow --version; pytest --version
```

Cuatro comprobaciones en una línea. Esperado: Python **3.11.x**, DVC 3.67.1,
MLflow 3.15.1, pytest 9.1.1. Si `python -V` dice 3.14 o similar, el entorno no
está activado y estás usando el Python del sistema.

---

## Nivel 1 — Código (pytest)

Antes de tocar datos, modelos o contenedores: ¿el código está sano?

```powershell
pytest
```

Ejecuta las 24 pruebas de `tests/`. Cubren el cargador de datos, el
preprocesamiento, los tres modelos, el pipeline de selección, el tracking de
MLflow y los endpoints de la API.

Esperado: `24 passed` (o `23 passed, 1 skipped` si aún no descargaste los
datos). La suite está diseñada para pasar en un clon recién hecho, **sin**
`dvc pull` y **sin** haber entrenado: las pruebas de la API inyectan un modelo
falso y la que necesita los CSV se salta sola.

```powershell
pytest -m datos
```

Ejecuta solo las pruebas marcadas con `@pytest.mark.datos`, las que sí exigen
los CSV reales de `data/raw/`. Úsalo **después** del `dvc pull` del nivel 2
para exigir que no se salte nada.

```powershell
pytest -v --durations=5
```

Modo verboso: lista el nombre de cada prueba y las 5 más lentas. Es el que
conviene proyectar al explicar la suite, porque los nombres de las pruebas se
leen como una especificación de lo que el sistema garantiza.

---

## Nivel 2 — DVC (datos y pipeline reproducible)

DVC hace dos cosas distintas en este proyecto y conviene explicarlas por
separado: **versiona los datos** (que no caben en Git) y **orquesta el
pipeline** (qué etapa hay que re-ejecutar cuando algo cambia).

### 2.1 Los datos

```powershell
dvc remote list
```

Muestra dónde vive el almacenamiento remoto. Esperado:
`gdrive_remote  gdrive://1s2ueXS2ubqD9_Tk1hs4k1DP4B_AKOVuY`. Git guarda solo
los punteros (`data/raw/train.csv.dvc`, un fichero de texto con el hash md5);
los CSV de verdad están en esa carpeta de Drive.

```powershell
dvc pull
```

Lee los ficheros `.dvc` del repo y descarga del remoto los datos cuyo hash
coincide. Es el equivalente de `git pull` pero para datos. La primera vez abre
el navegador para autorizar Google Drive.

```powershell
dvc status
```

Compara el hash de lo que tienes en disco con lo que registran los `.dvc` y el
`dvc.lock`. Esperado: `Data and pipelines are up to date.` Si sale
`changed outs`, algún fichero de datos o salida se modificó fuera del pipeline.

```powershell
Get-FileHash data\raw\train.csv -Algorithm MD5; Get-Content data\raw\train.csv.dvc
```

La demostración visual de que DVC no versiona el contenido sino la huella: el
md5 que calcula PowerShell es el mismo que está escrito en el `.dvc` que sí
está en Git.

### 2.2 El pipeline

```powershell
dvc dag
```

Dibuja en ASCII el grafo de dependencias de `dvc.yaml`: qué etapa alimenta a
cuál. Cuatro etapas: `validate` (control de calidad de los CSV), `train`
(entrena los tres modelos y guarda el campeón), `predict` (inferencia batch
sobre `test.csv`) y `tune` (búsqueda de hiperparámetros).

```powershell
dvc repro
```

El comando central. Recorre el grafo y **re-ejecuta solo lo que hace falta**:
para cada etapa compara los hashes de sus dependencias declaradas (datos,
código, parámetros de `params.yaml`) con los guardados en `dvc.lock`. Si nada
cambió, la salta y lo dice (`Stage ... didn't change, skipping`). Eso es la
reproducibilidad: dos personas con el mismo commit y los mismos datos obtienen
exactamente los mismos artefactos.

```powershell
dvc repro --force
```

Fuerza la re-ejecución de todo ignorando la caché. Útil para demostrar en vivo
que el pipeline corre de verdad de principio a fin (tarda unos minutos: entrena
3 modelos base + 6 configuraciones de la búsqueda).

```powershell
dvc metrics show
```

Lee los ficheros declarados como `metrics:` en `dvc.yaml`
(`reports/metrics.json` y `reports/hyperparameter_search.json`) y los imprime
en tabla. Son las métricas del último entrenamiento.

```powershell
dvc metrics diff main
```

Compara las métricas actuales contra las de la rama `main`. Este es el
argumento de venta de DVC frente a "entrenar a mano": puedes responder
*"¿este cambio mejoró el modelo?"* con un diff, igual que con el código.

**La prueba de que el pipeline es sensible a los cambios** (vale la pena
enseñarla, porque es lo que separa a DVC de un script suelto):

```powershell
dvc repro; dvc status
```

Primera pasada. Ahora edita `params.yaml` y cambia `random_forest.n_estimators`
de `300` a `250`, guarda, y repite:

```powershell
dvc repro
```

DVC detecta que el parámetro `random_forest` —declarado en la sección `params:`
de la etapa `train`— cambió, y re-ejecuta `train`, `predict` y `tune`, pero
**no** `validate`, porque esa etapa no depende de ese parámetro. Deja el valor
como estaba (`300`) y vuelve a ejecutar `dvc repro` para dejar el repo limpio.

---

## Nivel 3 — MLflow (tracking y Model Registry)

DVC te dice *qué* se ejecutó y con qué datos. MLflow te dice *cómo le fue*:
guarda cada entrenamiento con sus parámetros, sus métricas, sus gráficas y el
modelo entrenado, y mantiene un registro de qué versión está en producción.

### 3.1 Levantar la interfaz

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Arranca el servidor de MLflow leyendo la base local `mlflow.db` (la que este
repo usa por defecto, ver `params.yaml -> mlflow.tracking_uri`). Deja esta
terminal ocupada y abre [http://127.0.0.1:5000](http://127.0.0.1:5000) en el navegador. También vale
`make mlflow`.

### 3.2 Generar un experimento

En **otra** terminal, con el entorno activado:

```powershell
python main.py
```

El entrenamiento base. Entrena Regresión Logística, Random Forest y SVM,
elige el mejor por `accuracy` y lo promueve. En MLflow crea:

- Un **run padre** `comparacion_modelos` con las etiquetas de linaje
  (`git_commit`, `git_branch`, `dvc_train_md5`) — o sea, la trazabilidad de
  qué commit y qué versión de los datos produjeron ese modelo.
- Tres **runs hijos** anidados: `baseline_Regresión Logística`,
  `baseline_Random Forest`, `baseline_SVM`. Cada uno con sus parámetros,
  4 métricas, la matriz de confusión en JSON y en PNG, y el modelo serializado.
- Una **nueva versión** en el Model Registry `TelefonosPriceClassifier` con el
  alias `@champion` movido a ella.

```powershell
python -m scripts.tune_hyperparameters
```

La búsqueda de hiperparámetros. Recorre las combinaciones de
`params.yaml -> hyperparameter_search` y crea un run padre
`busqueda_hiperparametros` con **6 runs hijos** (`trial_1` … `trial_6`). El
ganador se registra con el alias `@challenger`, no con `@champion`: el
candidato no reemplaza al modelo en producción hasta que alguien lo decide.

### 3.3 Verificar en la interfaz

En [http://127.0.0.1:5000](http://127.0.0.1:5000):

1. **Experiments -> `telefonos_price_classification`**. Verás las filas de los
   runs padre. Si te parece que "solo se registra uno", es porque los hijos
   vienen **plegados** dentro del padre: pulsa la flecha `>` a la izquierda de
   `comparacion_modelos`, o desmarca la opción de agrupar por run padre, y
   aparecen los tres baselines.
2. Marca las casillas de dos o tres runs hijos y pulsa **Compare**: MLflow
   pone sus parámetros y métricas lado a lado y marca las diferencias.
3. Entra en un run -> pestaña **Artifacts**: ahí están
   `graficas/matriz_confusion.png`, `diagnosticos/clasificacion.json`,
   `resumen/comparacion_modelos.json` y la carpeta `model/` con la firma de
   entrada/salida.
4. **Models -> `TelefonosPriceClassifier`**: la lista de versiones y qué
   versión tiene el alias `@champion` y cuál `@challenger`.

### 3.4 Verificarlo por consola

Cuando no quieras depender del navegador:

```powershell
python -c "import mlflow; from src.config import MLFLOW_TRACKING_URI as u; mlflow.set_tracking_uri(u); df = mlflow.search_runs(experiment_names=['telefonos_price_classification']); print(df[['tags.mlflow.runName','metrics.accuracy','status']].to_string())"
```

Lista todos los runs del experimento con su nombre, su accuracy y su estado.
Esto demuestra el punto anterior: aunque la interfaz muestre pocas filas, aquí
salen todos.

```powershell
python -c "from mlflow import MlflowClient; from src.config import MLFLOW_TRACKING_URI as u, MLFLOW_REGISTERED_MODEL_NAME as n; c = MlflowClient(tracking_uri=u); [print(v.version, v.aliases, v.run_id) for v in c.search_model_versions(f\"name='{n}'\")]"
```

Lista las versiones registradas del modelo con sus alias y el run que las
produjo. Es la forma programática de contestar *"¿qué modelo exacto está
sirviendo la API ahora mismo?"*.

---

## Nivel 4 — FastAPI (serving)

```powershell
uvicorn src.api.app:app --reload --port 8000
```

Arranca la API. Al levantarse ejecuta `lifespan`, que carga el modelo **una
sola vez**: primero intenta el Model Registry pidiendo
`models:/TelefonosPriceClassifier@champion` y, si el Registry no está
disponible, cae a `models/modelo_final.pkl`. `--reload` reinicia el proceso
cuando cambia el código. Deja la terminal ocupada; usa otra para lo que sigue.

En el arranque verás una de estas dos líneas, y conviene señalarla:
`[API] Modelo cargado desde mlflow-registry (versión N)` o
`[API] Arranqué sin modelo: /health devolverá 503 hasta que lo haya.`

### 4.1 Los tres endpoints

```powershell
curl.exe http://127.0.0.1:8000/
```

Metadatos del servicio: nombre y alias del modelo, de dónde se cargó, qué
versión, y la lista de las 20 características que espera. Usa `curl.exe` y no
`curl` a secas: en PowerShell `curl` es un alias de `Invoke-WebRequest`, que
tiene otra sintaxis.

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

La sonda de readiness. `-i` muestra la cabecera de estado. **200** = hay modelo
cargado y la API puede predecir. **503** = el proceso está vivo pero no hay
modelo, así que no debería recibir tráfico. Esta distinción es la que usa el
`HEALTHCHECK` del contenedor: `/` respondería 200 aunque no hubiera modelo, y
por eso no serviría como sonda.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@tests/payload_ejemplo.json"
```

La predicción. Envía los dos teléfonos de ejemplo (uno de gama alta, otro de
gama baja) y devuelve para cada uno el `price_range` (0–3), su etiqueta legible,
el porcentaje de confianza y la probabilidad de cada clase. El `@` delante de
la ruta le dice a curl que lea el cuerpo desde ese fichero.

Esperado: el primer teléfono (3800 MB de RAM) sale en rango alto y el segundo
(512 MB) en rango bajo. La RAM es con diferencia la característica más
determinante del dataset, así que ese contraste es una buena demo.

### 4.2 La documentación interactiva

Abre [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Es el Swagger UI que FastAPI genera solo a
partir de los esquemas Pydantic de `src/api/schemas.py`. Desde ahí puedes
lanzar un `POST /predict` con el botón **Try it out** sin escribir curl, que
para una demo en vivo suele funcionar mejor.

### 4.3 Los casos de error

Merece la pena enseñar que la API no solo funciona con la entrada buena:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"data\":[{\"battery_power\":1900}]}"
```

Faltan 19 características. Esperado: **422** con el detalle de qué campos
faltan, generado por la validación de Pydantic antes de que el modelo vea nada.

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"data\":[]}"
```

Lista vacía. Esperado: **422**, porque el esquema exige al menos una fila.

### 4.4 La autenticación opcional

Por defecto la API key está desactivada (variable vacía = sin autenticación),
que es lo cómodo en local y en pytest. Para verificar que el mecanismo existe,
para uvicorn y arráncalo con la variable puesta.

Antes de nada, el aviso importante: si el stack de Docker está levantado, el
contenedor `mlops-api` ya está publicando el **8000**. Windows deja que uvicorn
se ate a `127.0.0.1:8000` aunque Docker tenga tomado `0.0.0.0:8000`, así que
acabas con dos servidores distintos respondiendo al mismo `localhost:8000`
según el momento, cada uno con su propia `API_KEY`. Es una tarde perdida
mandando la clave correcta al servidor equivocado.

Así que o paras el contenedor:

```powershell
docker compose stop api
```

...o arrancas uvicorn en otro puerto (el resto de esta sección usa el 8000; si
eliges el 8001, cámbialo también en los `curl`):

```powershell
$env:API_KEY = "clave-de-prueba"; uvicorn src.api.app:app --port 8001
```

Para saber quién escucha en cada puerto, con el nombre del proceso:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object LocalAddress,OwningProcess,@{n='Proceso';e={(Get-Process -Id $_.OwningProcess).ProcessName}}
```

`com.docker.backend` y `wslrelay` son de Docker Desktop; cualquier `python` de
esa lista es un uvicorn tuyo. Ojo con filtrar por `netstat -ano | findstr ":8000"`:
si tu dirección IPv6 pública contiene el grupo `8000`, engancha decenas de
conexiones del navegador que no tienen nada que ver.

Con uvicorn ya arrancado, desde otra terminal:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@tests/payload_ejemplo.json"
```

Esperado: **401**, porque falta la cabecera. Y con ella:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -H "X-API-Key: clave-de-prueba" -d "@tests/payload_ejemplo.json"
```

Esperado: **200** con las predicciones. Al terminar, limpia la variable con
`Remove-Item Env:API_KEY` para que la siguiente terminal no la herede, y vuelve
a levantar el contenedor si lo paraste: `docker compose start api`.

La suite de pytest ya no se ve afectada por esa variable: un fixture autouse en
`tests/conftest.py` desactiva la autenticación durante los tests. Olvidarse la
variable puesta hacía fallar seis tests de la API con 401 por un motivo que no
tenía nada que ver con el código que estaban probando.

Si la clave se rechaza y estás seguro de haberla escrito bien, mira el log del
arranque: la API avisa cuando `API_KEY` conserva los ángulos `< >` del
generador o trae espacios y saltos de línea dentro.

---

## Nivel 5 — Docker (el stack completo)

Hasta aquí todo corría contra tu `.venv` y tu `mlflow.db`. Docker demuestra que
el proyecto se levanta en una máquina limpia sin instalar nada más que Docker.

El stack tiene tres servicios: **mlflow** (tracking + registry, con su propio
volumen), **api** (serving) y **trainer** (a demanda, no arranca solo).

```powershell
docker compose config
```

Valida el `docker-compose.yml` y lo imprime ya resuelto, con las variables
sustituidas. Es la comprobación barata de sintaxis antes de construir nada.

```powershell
docker compose build; docker compose build trainer
```

Construye las imágenes. `trainer` va aparte porque está en el perfil `tools`:
no arranca con `up`, solo se invoca a demanda. Las imágenes de `api` y
`trainer` son multi-etapa: las dependencias se instalan en una etapa `deps` que
queda cacheada, y a la imagen final solo viaja el entorno virtual ya resuelto,
sin caché de pip. La primera construcción tarda varios minutos.

```powershell
docker compose up -d
```

Levanta `mlflow` y `api` en segundo plano (`-d` = detached). La API **no**
espera a que exista un modelo: arranca igual y `/health` devuelve 503 hasta que
lo haya. Esa decisión es la que evita que el contenedor entre en un bucle de
reinicios en el primer despliegue.

```powershell
docker compose ps
```

Estado de los servicios. Espera unos 30 segundos y vuelve a ejecutarlo: la
columna STATUS debe decir `healthy` para `mlops-mlflow`. La API estará
`unhealthy` mientras no haya modelo — es lo correcto, todavía no has entrenado
dentro del stack.

```powershell
docker compose logs -f api
```

Sigue los logs de la API en vivo (`Ctrl+C` para salir, no para el contenedor).
Aquí verás la línea de `[model_loader]` explicando de dónde cargó el modelo o
por qué no pudo.

```powershell
docker compose run --rm trainer
```

Entrena **dentro** del contenedor. Es el paso interesante para explicar:

- `--rm` borra el contenedor al terminar; el trainer es un trabajo puntual, no
  un servicio.
- Escribe los runs en el MLflow del stack, no en tu `mlflow.db` local, porque
  el servicio inyecta `MLFLOW_TRACKING_URI=http://mlflow:5000` y
  `src/config.py` da prioridad al entorno sobre `params.yaml`.
- Escribe `models/`, `reports/` y `data/` en tu disco, montados como volumen,
  para que DVC los siga versionando desde fuera.

> **Ojo con `dvc status` despues de este paso.** Las salidas de texto
> (`reports/*.json`, `data/processed/*.csv`) salen identicas byte a byte a las
> que produce Windows, porque el pipeline fuerza LF. El `.pkl` no: joblib no
> serializa igual en Linux y en Windows, asi que sale con el mismo tamano y las
> mismas metricas pero distinto hash, y `dvc status` marca `train` y `predict`
> como modificados. Es comportamiento de pickle entre plataformas, no un fallo
> del stack. Si no quieres versionar el modelo entrenado en el contenedor,
> descarta el cambio con:
>
> ```powershell
> dvc checkout --force models/modelo_final.pkl
> ```

```powershell
docker compose restart api
```

Reinicia la API para que vuelva a ejecutar `lifespan` y recargue el nuevo
`@champion` desde el Registry. El modelo no está horneado en la imagen: por eso
promover un modelo nuevo no obliga a reconstruir nada.

```powershell
docker compose ps
```

Ahora sí: ambos servicios `healthy`.

### 5.1 Verificar el stack desde fuera

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Esperado: **200**, y en el cuerpo `"origen": "mlflow-registry"` — la prueba de
que la API está sirviendo desde el Model Registry del contenedor y no del
`.pkl` de respaldo.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@tests/payload_ejemplo.json"
```

La misma predicción del nivel 4, ahora contra el contenedor. Mismo resultado =
el empaquetado no cambió el comportamiento.

Abre [http://127.0.0.1:5000](http://127.0.0.1:5000): es el MLflow del stack, con su propia base. Está
vacío salvo por el entrenamiento que acabas de lanzar con `trainer`, y eso es
intencionado: el historial de tu máquina no se copia porque sus artefactos
apuntan a rutas `C:/Users/...` que dentro de un contenedor no existirían.

```powershell
docker compose run --rm trainer python -m scripts.predict
```

Ejecuta la inferencia batch dentro del contenedor sobrescribiendo el comando
por defecto de la imagen. Genera `data/processed/predicciones.csv` en tu disco.

```powershell
docker compose down
```

Para y elimina los contenedores, **conservando** los volúmenes: el historial de
MLflow del stack sigue ahí para el siguiente `up`.

```powershell
docker compose down -v
```

Lo mismo pero borrando los volúmenes: se pierde el historial de MLflow del
stack. Úsalo solo para empezar de cero a propósito.

---

## Para el equipo: verificación desde un clon limpio

Estos son los pasos mínimos para que cada integrante compruebe en su máquina
que la rama funciona. Cada uno debería llegar hasta el nivel que le toque
defender.

### Requisitos previos

- Python 3.11 (**no** 3.12+; las versiones fijadas están verificadas en 3.11)
- Git
- Docker Desktop, solo para la parte de contenedores
- Acceso a la carpeta de Google Drive del remoto de DVC

### Paso 1 — Clonar y preparar

```powershell
git clone https://github.com/Lushexxx7/Mobile_Price_Class_MLOPS.git
```

```powershell
cd Mobile_Price_Class_MLOPS; git checkout main
```

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

### Paso 2 — Probar el código sin datos

```powershell
pytest
```

Esta es la primera señal de vida y no necesita ni datos ni credenciales.
Esperado: `24 passed` (o 23 + 1 skipped). **Si esto falla, para y avisa**: es
un problema de entorno, no del resto del pipeline.

### Paso 3 — Configurar DVC y traer los datos

El remoto de Google Drive necesita credenciales que **no** están en el repo
(viven en `.dvc/config.local`, que está en `.gitignore` precisamente para que
no se suban claves a GitHub). Pídeselas a Luis y crea el fichero:

```powershell
dvc remote modify --local gdrive_remote gdrive_client_id "<CLIENT_ID>"
```

```powershell
dvc remote modify --local gdrive_remote gdrive_client_secret "<CLIENT_SECRET>"
```

`--local` escribe en `.dvc/config.local` en vez de en `.dvc/config`, que sí
está versionado. Es la separación entre configuración compartida (la URL del
remoto) y secretos personales (las claves).

```powershell
dvc pull
```

Descarga los CSV. La primera vez abre el navegador para autorizar la cuenta de
Google. Esperado: 2 ficheros traídos a `data/raw/`.

```powershell
dvc status; pytest -m datos
```

Verifica que los datos llegaron íntegros y que las pruebas que dependen de
ellos ahora sí se ejecutan en vez de saltarse.

### Paso 4 — Reproducir el pipeline

```powershell
dvc repro
```

Esperado la primera vez: se ejecutan las cuatro etapas. La segunda vez, todas
se saltan. Que se salten **es** el resultado correcto: significa que tu
ejecución coincide con la registrada en `dvc.lock`.

```powershell
dvc metrics show
```

Compara tus métricas con las del compañero. Deben coincidir: `random_state: 42`
está fijado en `params.yaml` y se propaga a los tres modelos y a la partición
train/validación.

### Paso 5 — MLflow y API

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Cada integrante tiene su propio `mlflow.db` local (está en `.gitignore`, no se
comparte). Verás solo tus propios runs, y es lo esperado: el historial
compartido de verdad es el del stack de Docker.

```powershell
uvicorn src.api.app:app --reload --port 8000
```

Y desde otra terminal, la prueba de humo:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@tests/payload_ejemplo.json"
```

### Paso 6 — Docker

```powershell
docker compose build; docker compose build trainer; docker compose up -d
```

```powershell
docker compose run --rm trainer; docker compose restart api
```

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Esperado: 200 con `"origen": "mlflow-registry"`.

### Qué reportar

Al terminar, cada uno pega en el grupo:

- La línea final de `pytest` (`N passed`)
- La salida de `dvc status`
- La tabla de `dvc metrics show`
- El `accuracy` del mejor modelo y qué modelo ganó
- La respuesta de `/health` con Docker levantado
- Cualquier paso donde se atascó, con el mensaje de error completo

---

## Añadir más experimentos a MLflow

### Por qué parece que "solo se registra uno"

No lo es. Cada `python main.py` crea **4 runs** (1 padre + 3 hijos) y cada
`python -m scripts.tune_hyperparameters` crea **7** (1 padre + 6 trials). Lo
que pasa es que la interfaz de MLflow **agrupa los runs anidados bajo su
padre** y los muestra plegados, así que la tabla enseña una sola fila por
ejecución. Pulsa la flecha `>` a la izquierda del nombre del run padre, o
desactiva la agrupación en la configuración de la tabla, y aparecen todos.

Para confirmarlo sin la interfaz:

```powershell
python -c "import mlflow; from src.config import MLFLOW_TRACKING_URI as u; mlflow.set_tracking_uri(u); print(len(mlflow.search_runs(experiment_names=['telefonos_price_classification'])), 'runs')"
```

`search_runs` no agrupa nada: devuelve la lista plana y ahí se ven todos.

### La forma prevista de añadir experimentos: `params.yaml`

**No hay que tocar código.** El bloque `hyperparameter_search` de `params.yaml`
es la superficie de configuración: `scripts/tune_hyperparameters.py` lo recorre
y genera un run por combinación. Añadir experimentos = añadir entradas ahí.

Estado actual (6 combinaciones):

```yaml
hyperparameter_search:
  logistic_regression:
    C: [0.1, 10.0]
  random_forest:
    trials:
      - n_estimators: 200
        max_depth: null
      - n_estimators: 300
        max_depth: 20
  svm:
    C: [1.0, 10.0]
```

Ampliado a 12 combinaciones:

```yaml
hyperparameter_search:
  logistic_regression:
    C: [0.01, 0.1, 1.0, 10.0]        # 4 runs en vez de 2
  random_forest:
    trials:                           # 4 runs en vez de 2
      - n_estimators: 200
        max_depth: null
      - n_estimators: 300
        max_depth: 20
      - n_estimators: 500
        max_depth: 30
      - n_estimators: 150
        max_depth: 10
  svm:
    C: [0.5, 1.0, 10.0, 100.0]       # 4 runs en vez de 2
```

Después:

```powershell
dvc repro tune
```

DVC ve que el parámetro `hyperparameter_search` —declarado en la sección
`params:` de la etapa `tune` en `dvc.yaml`— cambió, y re-ejecuta esa etapa y
solo esa. En MLflow aparecerá un nuevo run padre `busqueda_hiperparametros` con
12 hijos, y el ganador se registrará como `@challenger`.

O sin pasar por DVC, si solo quieres verlo en MLflow:

```powershell
python -m scripts.tune_hyperparameters
```

### Cómo se reparte el trabajo el equipo

Cada integrante puede explorar un modelo distinto sin pisarse, porque cada uno
edita un bloque separado de `hyperparameter_search`:

| Integrante | Bloque de`params.yaml`  | Qué varía                      |
| ---------- | ------------------------- | -------------------------------- |
| A          | `logistic_regression.C` | fuerza de la regularización     |
| B          | `random_forest.trials`  | `n_estimators` y `max_depth` |
| C          | `svm.C`                 | margen del clasificador          |

Flujo por persona:

```powershell
git checkout -b exp/<tu-nombre>-<modelo>
```

Edita tu bloque de `params.yaml`, y luego:

```powershell
dvc repro tune; dvc metrics show
```

```powershell
git add params.yaml dvc.lock reports/hyperparameter_search.json; git commit -m "exp: amplia la busqueda de <modelo>"
```

Al comparar ramas, `dvc metrics diff` responde quién encontró la mejor
configuración:

```powershell
dvc metrics diff main
```

### Añadir un hiperparámetro que hoy no se explora

Si quieres variar algo que el script todavía no recorre —por ejemplo el
`kernel` del SVM, que hoy sale fijo de `params.yaml`— hay que tocar el
generador `candidatos()` de `scripts/tune_hyperparameters.py`. El patrón es
siempre el mismo: `yield` de una pareja (modelo configurado, diccionario de
parámetros que se registrará en MLflow).

Para el kernel del SVM, en `params.yaml`:

```yaml
  svm:
    C: [1.0, 10.0]
    kernel: [rbf, linear, poly]
```

Y en `candidatos()`, sustituyendo el bucle del SVM:

```python
    for kernel in busqueda["svm"].get("kernel", [PARAMS["svm"]["kernel"]]):
        for c in busqueda["svm"]["C"]:
            modelo = ModeloSVM()
            modelo.modelo.set_params(
                classifier__C=float(c), classifier__kernel=str(kernel)
            )
            yield modelo, {"C": c, "kernel": kernel, "gamma": PARAMS["svm"]["gamma"]}
```

Eso pasa de 2 a 6 runs de SVM. `.get(..., [por defecto])` mantiene la
compatibilidad: si alguien no añade la clave `kernel` a su `params.yaml`, el
script sigue funcionando con el valor único de siempre.

Cuando toques el script, verifica que no rompiste el tracking:

```powershell
pytest tests/test_tracking.py -v
```

### Añadir un modelo nuevo al conjunto base

Si el equipo quiere probar un cuarto algoritmo (Gradient Boosting, KNN, …), el
sitio es `src/models/train.py`: se hereda de `ModeloClasificacion`, se añade a
`crear_modelos()`, y tanto `main.py` como MLflow lo recogen automáticamente —
ambos iteran sobre la lista, no sobre nombres escritos a mano. Añade también su
bloque de parámetros a `params.yaml` para no dejar valores fijos en el código,
y su entrada en `dvc.yaml -> train -> params` para que DVC sepa que ese modelo
tiene parámetros que invalidan el entrenamiento cuando cambian.

---

## Resumen: la ruta corta para una demo

Si tienes que enseñarlo todo en 10 minutos, esta es la secuencia:

```powershell
pytest
```

```powershell
dvc dag; dvc repro; dvc metrics show
```

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

```powershell
python main.py
```

```powershell
docker compose up -d; docker compose run --rm trainer; docker compose restart api
```

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@tests/payload_ejemplo.json"
```
