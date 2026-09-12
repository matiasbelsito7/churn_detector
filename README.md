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

| Documento | Propósito |
|---|---|
| [AGENTS.md](AGENTS.md) | Contexto operativo, responsabilidades y convenciones; punto de entrada. |
| [docs/constitution.md](docs/constitution.md) | Principios y reglas que gobiernan el desarrollo del proyecto. |
| [docs/specs.md](docs/specs.md) | Especificación funcional y técnica del sistema. |
| [docs/tasks.md](docs/tasks.md) | Roadmap de implementación con dependencias y criterios de completitud. |
| [docs/data_contract.md](docs/data_contract.md) | Contrato y diccionario de datos del dataset de churn. |

## Estructura del repositorio

La estructura respeta las capas conceptuales del proyecto (ver `AGENTS.md`,
sección 2):

```
docs/                 Documentación de gobierno del proyecto.
data/
  raw/                Datos crudos inmutables y versionados.
  processed/          Artefactos derivados reproducibles (limpieza).
  features/           Datos con feature engineering.
src/
  data/               Capa de datos: ingestion, curation, cleaning, calidad.
  analysis/           Capa de análisis: EDA y feature engineering.
  modeling/           Capa de modelado: preprocessing, entrenamiento, evaluación.
  serving/            Capa de industrialización: pipeline, API, persistencia, monitoreo.
notebooks/            EDA exploratorio versionado y reproducible.
scripts/              Utilidades y orquestación.
tests/                Pruebas de las distintas capas.
reports/              Informes generados (audit, calidad, EDA, monitoreo).
```

## Entorno reproducible

Se usa `uv` (Python >= 3.11). Para instalar el entorno desde cero:

```powershell
uv sync --extra dev
```

Las versiones exactas quedan fijadas en `uv.lock`. Las semillas aleatorias se
centralizan en `src/seeds.py`.

## Estado

En preparación. Las fases y tareas se definen en `docs/tasks.md`.

## Contacto

Proyecto de portfolio personal. Consultas e incidencias: gestionar a través
de la documentación del proyecto.
