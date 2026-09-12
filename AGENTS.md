# AGENTS.md

## 1. Contexto y objetivo general

Proyecto de portfolio de Data Science/MLOps: sistema end-to-end de detección y
predicción de churn de clientes. Dataset principal: **IBM Telco Customer Churn**.

La fase inicial se centra en **datos**: data curation, data quality,
preprocessing, tratamiento de valores faltantes, EDA y feature engineering,
antes de entrenar cualquier modelo.

El proyecto debe evolucionar hasta convertirse en un **sistema reproducible y
desplegable**, incorporando SQL/PostgreSQL, pipelines de ML, MLflow, FastAPI,
Docker, testing, CI/CD y monitoring.

Aspectos que gobiernan todo el trabajo:

- El proyecto se trata como un **sistema real de producción**, no como un notebook.
- El **dato es lo primero**: no se entrena ni despliega nada sobre datos que no
  hayan sido comprendidos, curados y validados.
- La calidad, comprensión y trazabilidad de los datos tienen prioridad sobre
  el modelo y la infraestructura.

## 2. Estructura conceptual del trabajo

El proyecto se organiza en capas conceptuales que deben permanecer separadas:

- **Datos**: adquisición, versionado, curation, limpieza y validación de calidad.
- **Análisis**: EDA y feature engineering, con decisiones basadas en evidencia.
- **Modelado**: preprocessing, entrenamiento, evaluación y selección de modelo.
- **Industrialización**: pipelines reproducibles, persistencia, API, infraestructura
  y monitoreo.

Los agentes deben respetar esa separación al introducir cambios: el código de
cada capa vive en su espacio y no se mezcla con el de otras capas.

La descomposición concreta del trabajo en fases y tareas es responsabilidad de
`docs/tasks.md`; en este documento se define únicamente la lógica conceptual
que lo ordena.

## 3. Responsabilidades esperadas de los agentes

- **Consultar antes de actuar**: leer los documentos de referencia (sección 6)
  antes de empezar cualquier tarea.
- **Trabajar dentro del alcance**: ejecutar la tarea asignada sin ampliar el
  alcance ni inventar requisitos; cualquier requisito nuevo se propone, no se
  asume.
- **Derivar de los documentos**: las decisiones deben derivarse de
  `docs/constitution.md`, `docs/specs.md` y `docs/tasks.md`, y no de criterios
  personales.
- **Producir código reproducible**: transformaciones e imports de datos
  reproducibles, con versionado de datos y dependencias.
- **Producir código testeable y documentado**: toda funcionalidad nueva debe
  acompañarse de pruebas y documentación mínima.
- **Verificar el trabajo**: correr tests, lint y/o typecheck antes de dar una
  tarea por completada.
- **Declarar bloqueos**: si una tarea no puede completarse (datos faltantes,
  ambigüedad, dependencia no resuelta), exponer el bloqueo en lugar de
  improvisar una solución.

## 4. Convenciones de colaboración

- **Rutina de cambio**: leer documentos de referencia → identificar la tarea en
  `docs/tasks.md` → implementar → verificar (tests/lint) → revisar el diff →
  declarar completitud.
- **Cierre de cambio**: la entrega de una tarea terminada en git sigue la skill
  `submit-change` (ver sección 7).
- **Un cambio por tarea**: evitar commits que mezclen tareas o responsabilidades
  de capas distintas.
- **Idioma**: el contenido de documentación, mensajes de commit y comentarios se
  redacta en español; identificadores y artefactos de código siguen convenciones
  del lenguaje que los define.
- **No sobre-ingeniería**: en las decisiones de implementación se aplica el
  principio homónimo de `docs/constitution.md` (sección 4): evitar complejidad
  tecnológica innecesaria y elegir la alternativa más simple que cumpla la tarea.
- **Actualización de la documentación**: si un cambio altera el sistema descrito
  en `docs/specs.md` o las tareas en `docs/tasks.md`, se propone la
  actualización del documento en lugar de silenciar la divergencia.

## 5. Criterios generales para modificar archivos

- **Documentos de gobierno**: `docs/constitution.md`, `docs/specs.md` y
  `docs/tasks.md` no se modifican sin autorización explícita; se revisan y se
  proponen cambios, no se editan directamente.
- **Datos**: no se modifican los datos crudos; el versionado y transformación de
  datos debe ser trazable y reproducible.
- **Confidencialidad**: no se suben claves, credenciales ni datos personales a
  repositorios ni archivos de configuración.
- **Estructura del repositorio**: se respeta la organización de capas y
  directorios que define la documentación del proyecto.
- **Dependencias**: no se agregan librerías o servicios sin necesidad justificada.
- **Pruebas**: una funcionalidad nueva llega con sus pruebas; los cambios que
  rompen pruebas existentes deben corregirse antes de declararse completos.

## 6. Documentos de referencia

Toda la documentación del proyecto vive en los siguientes archivos, y cada
agente debe consultarlos antes de cada intervención:

| Documento | Propósito |
|---|---|
| `AGENTS.md` | Este archivo: contexto operativo, responsabilidades y convenciones; punto de entrada a la documentación. |
| `docs/constitution.md` | Principios y reglas que gobiernan el desarrollo del proyecto. |
| `docs/specs.md` | Especificación funcional y técnica de lo que debe cumplir el sistema. |
| `docs/tasks.md` | Roadmap de implementación: tareas ordenadas, dependencias y criterios de completitud. |

Todos los documentos de esta sección son de consulta obligatoria antes de cada
intervención. `AGENTS.md` es el punto de entrada; `docs/constitution.md`,
`docs/specs.md` y `docs/tasks.md` no se editan directamente: los cambios se
proponen y se aprueban antes de aplicarse (ver sección 5).

## 7. Skills del proyecto

Las skills son procedimientos reutilizables definidos en `.opencode/skills/`.
Un agente debe cargarlas cuando la tarea coincida con su descripción.

| Skill | Ruta | Cuándo usarla |
|---|---|---|
| `submit-change` | `.opencode/skills/submit-change/SKILL.md` | Entregar y publicar un cambio de una tarea `T-NN`: verificar (black, ruff, mypy, pytest, pre-commit), revisar diff, commitear, pushear y validar CI/CD tras el push. |
