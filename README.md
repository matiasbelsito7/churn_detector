# Churn Detector

Sistema end-to-end de detección y predicción de churn de clientes, construido
sobre el dataset **IBM Telco Customer Churn**. Proyecto de portfolio de
Data Science/MLOps tratado como un sistema real de producción.

## Objetivo

El dato es lo primero: la fase inicial se centra en data curation, data
quality, preprocessing, tratamiento de valores faltantes, EDA y feature
engineering antes de entrenar cualquier modelo. El proyecto evoluciona hacia
un sistema reproducible y desplegable (pipelines de ML, SQL/PostgreSQL,
MLflow, API, Docker, testing, CI/CD y monitoring).

## Documentación

La documentación de gobierno vive en los siguientes archivos y es de consulta
obligatoria antes de cada intervención (`AGENTS.md` es el punto de entrada):

| Documento | Propósito |
|---|---|
| [AGENTS.md](AGENTS.md) | Contexto operativo, responsabilidades y convenciones; punto de entrada. |
| [docs/constitution.md](docs/constitution.md) | Principios y reglas que gobiernan el desarrollo del proyecto. |
| [docs/specs.md](docs/specs.md) | Especificación funcional y técnica del sistema. |
| [docs/tasks.md](docs/tasks.md) | Roadmap de implementación con dependencias y criterios de completitud. |
| [docs/data_contract.md](docs/data_contract.md) | Contrato y diccionario de datos del dataset de churn. |
| [docs/missing_values.md](docs/missing_values.md) | Decisiones de tratamiento de valores faltantes, justificadas. |
| [docs/feature_dictionary.md](docs/feature_dictionary.md) | Diccionario de features derivadas del EDA (T-10). |
| [docs/monitoring.md](docs/monitoring.md) | Plan de monitoreo: métricas, umbrales de alerta y acciones (T-22). |

## Decisión de arquitectura

El sistema se organiza en capas conceptuales separadas (ver `AGENTS.md`,
sección 2): **datos**, **análisis**, **modelado** e **industrialización**. Las
decisiones principales derivan de `docs/constitution.md`, `docs/specs.md` y de
los hallazgos del EDA:

- **El dato es lo primero**: los datos crudos se adquieren inmutables y
  versionados (`data/raw/` + `PROVENANCE.yaml`), y pasan checks explícitos de
  calidad antes de avanzar. Ningún artefacto se produce sobre datos no
  validados.
- **Reproducibilidad**: todo transformador se ajusta únicamente con datos de
  entrenamiento (estructura de `src/modeling/preprocessing.py`), las semillas se
  centralizan en `src/seeds.py` y el pipeline completo
  `datos → features → modelo → predicción` se regenera con un único comando.
- **Tracking**: los experimentos se registran en MLflow (SQLite), con el modelo
  seleccionado registrado en el Model Registry y exportado a un artefacto
  standalone (`data/models/`) para el despliegue.
- **Industrialización**: API FastAPI para inferencia, PostgreSQL para
  persistencia consultable y trazable, Docker para el despliegue reproducible,
  GitHub Actions para CI/CD y un plan de monitoreo con umbrales y acciones.

## Estructura del repositorio

```
docs/                 Documentación de gobierno del proyecto.
data/
  raw/                Datos crudos inmutables y versionados (PROVENANCE.yaml).
  processed/          Artefactos derivados reproducibles (limpieza).
  features/           Datos con feature engineering.
  splits/             Particiones reproducibles train/validation/test (log + CSVs).
  models/             Modelo seleccionado exportado (joblib).
  predictions/        Predicciones del conjunto (CSV + log) y persistencia SQL.
src/
  seeds.py            Semillas aleatorias centralizadas del proyecto.
  data/               Capa de datos: ingestion, curation, cleaning, calidad.
  analysis/           Capa de análisis: EDA y feature engineering.
  modeling/           Capa de modelado: preprocessing, entrenamiento, evaluación.
  serving/            Capa de industrialización: pipeline, API, persistencia, monitoreo.
notebooks/            EDA exploratorio versionado y reproducible.
scripts/              Utilidades y orquestación.
tests/                Pruebas de las distintas capas.
reports/              Informes generados (audit, calidad, EDA, selección, monitoreo).
.github/workflows/    CI (lint/typecheck/tests) y CD (imagen Docker).
docker-compose.yml    PostgreSQL y API para el entorno local.
Dockerfile            Imagen de la API con el modelo embebido.
```

## Instalación

Se usa `uv` (Python >= 3.11). Para instalar el entorno desde cero:

```powershell
uv sync --extra dev
```

Las versiones exactas quedan fijadas en `uv.lock`. Para activar los hooks de
pre-commit (también ejecutados en CI):

```powershell
uv run pre-commit install
```

Para levantar la base PostgreSQL local (necesaria para persistencia):

```powershell
docker compose up -d postgres
```

## Uso

### Pipeline reproducible (T-15)

El pipeline completo `datos → features → modelo → predicción` se orquesta desde
un único comando:

```powershell
uv run python -m src.serving.pipeline
```

También permite ejecutar etapas sueltas, por ejemplo:

```powershell
uv run python -m src.serving.pipeline features prediccion
```

La ejecución regenera desde el raw versionado todos los artefactos (procesado,
features, particiones, experimentos, tracking y modelos, selección y
predicciones). Con los mismos datos y configuración los resultados son idénticos
(semillas fijas); los artefactos se versionan en `data/predictions/`.

### Tracking de experimentos (T-17)

El registro de parámetros, métricas y modelos de cada experimento se integra con
MLflow:

```powershell
uv run python -m src.modeling.tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

La base de tracking (`mlflow.db`) y los artefactos (`mlruns/`) no se versionan;
se regeneran ejecutando el módulo.

### API de predicción (T-18)

La API FastAPI expone dos endpoints conforme a `docs/specs.md` §8.2:

```powershell
uv run python -m src.serving.api
```

- `GET /health` — disponibilidad del servicio.
- `POST /predict` — predicción por cliente. Recibe el esquema procesado del
  cliente (sin target) y devuelve `customerID`, `churn_prob` y `churn_class`.
  Aplica el mismo feature engineering que en entrenamiento y el modelo
  seleccionado con su pipeline, cargados desde MLflow (o desde el artefacto
  `MODEL_FILE` si se define).
- Errores: entrada inválida devuelve `422` con código `invalid_request` y
  detalle de campos; modelo no disponible devuelve `503` con `model_unavailable`;
  errores internos devuelven `500` sin exponer detalles.

Documentación interactiva (OpenAPI) en `http://127.0.0.1:8000/docs`.

Ejemplo:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"customerID":"0001-ABCD","gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":24,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":74.5,"TotalCharges":1788.0}'
```

Se puede desplegar la API con Docker (comparte red con PostgreSQL):

```powershell
docker compose up -d --build api
```

### Persistencia en PostgreSQL (T-16)

Carga las predicciones del pipeline en las tablas `predictions` y
`source_files` (trazables a su origen versionado) y verifica el esquema por SQL:

```powershell
uv run python -m src.serving.persist
```

Configurable por variables de entorno (`POSTGRES_HOST`, `POSTGRES_PORT`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`). El resultado queda en
`reports/persistence.md` y `reports/persistence.json`.

### Monitoreo (T-22)

Evalúa drift de features, distribución de predicciones y desempeño respecto al
conjunto de referencia, con los umbrales definidos en `docs/monitoring.md`:

```powershell
uv run python -m src.serving.monitor
```

El reporte se escribe en `reports/monitoring.md` y `reports/monitoring.json`.

### Calidad de datos y validación

```powershell
uv run python -m src.data.audit
uv run python -m src.data.quality_checks
```

(o los equivalentes `audit-data` y `quality-check` expuestos por pyproject).

### Tests, lint y typecheck

```powershell
uv run pytest
uv run black --check src tests
uv run ruff check src tests
uv run mypy src
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`): en cada push/PR a `main` ejecuta black,
  ruff, mypy y pytest con el lockfile congelado.
- **CD** (`.github/workflows/cd.yml`): en cada push a `main` construye la imagen
  Docker de la API. En local se puede reproducir con `docker compose up --build`.

## Estado

Sistema implementado de extremo a extremo (fases 1 a 19, T-01 a T-23). Las
fases, dependencias y criterios de completitud se definen en `docs/tasks.md`;
los reportes generados por cada fase viven en `reports/`.

## Contacto

Proyecto de portfolio personal. Repositorio:
`https://github.com/matiasbelsito7/churn_detector`. Consultas e incidencias:
abrir un issue en el repositorio o gestionar a través de la documentación del
proyecto.
