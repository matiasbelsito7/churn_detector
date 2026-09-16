# tasks.md

## 1. Propósito y notación

Este documento transforma la especificación funcional y técnica de `specs.md`
en un roadmap de implementación concreto, ordenado por fases y dependencias.

Las tareas se derivan de `specs.md` y respetan los principios de
`constitution.md` y las convenciones de `AGENTS.md`.

**Formato de cada tarea:**

- `T-NN` — identificador único.
- Descripción concreta.
- Dependencias (`T-XX` o "ninguna").
- Criterio de completitud verificable.

**Gobierno:** este documento no se modifica sin autorización explícita (per
`AGENTS.md`, sección 5). Las modificaciones se proponen y justifican, no se
aplican directamente.

## 2. Mapa de fases

```
T-01 → T-02
T-01 → T-03 → T-04 → T-05 → T-06 → T-07 → T-08
T-08 → T-09 → T-10 → T-11 (particiones y baseline) → T-12 (pipeline) → T-13
T-13 → T-17 (MLflow)
T-17, T-13 → T-14 (evaluación y selección)
T-14 → T-15 → T-16
T-15, T-17 → T-18 → T-19
T-15, T-18 → T-20
T-19, T-20 → T-21
T-18 → T-22
todas → T-23
T-23 → T-24 → T-25 → T-26 → T-27
```

Nota: las particiones (`T-11`) se crean **antes** del pipeline de preprocessing
(`T-12`), y el tracking (`T-17`) se integra **antes** de la evaluación y
selección (`T-14`) para que los criterios de registro y auditoría sean
verificables.

## 3. Tareas

---

### Fase 1 — Preparación del repositorio

#### T-01

**Descripción:** Inicializar el repositorio git, establecer la estructura de
directorios por capas (datos, análisis, modelado, industrialización, tests,
documentación) y crear un `README` con índice a la documentación del proyecto.

**Dependencias:** ninguna.

**Criterio de completitud:**
- Repo inicializado y clonable.
- Estructura de directorios creada y documentada en `README`.
- No se versionan secretos ni credenciales; la política de versionado de datos
  crudos se define en `T-03`.

#### T-02

**Descripción:** Declarar el entorno reproducible: archivo de dependencias
(`pyproject.toml` o `requirements.txt`), configuración de lint, typecheck y
tests, y fijación de semillas aleatorias.

**Dependencias:** T-01.

**Criterio de completitud:**
- El entorno se instala de forma reproducible desde cero.
- El lint, typecheck y la suite de tests base se ejecutan sin errores.

---

### Fase 2 — Adquisición y versionado de datos

#### T-03

**Descripción:** Adquirir el dataset original y guardarlo como **raw inmutable**.
Dado su tamaño reducido, el raw se **versiona en el repositorio** junto con su
registro de procedencia (fuente, fecha de descarga, checksum, código o commit de
la fuente). Si el volumen creciera y lo justificara, se reemplaza por un
mecanismo de almacenamiento externo (p. ej. DVC) sin alterar el resto del flujo
ni el contrato de datos.

**Dependencias:** T-01.

**Criterio de completitud:**
- Archivo raw presente en el directorio de datos crudos y versionado en el repo.
- Registro de procedencia (checksum, fecha, fuente) versionado junto al
  proyecto.
- Ningún script modifica el raw; toda transformación opera sobre derivados.

---

### Fase 3 — Data audit y curation

#### T-04

**Descripción:** Documentar el contrato de datos: columnas esperadas, tipos,
dominios, rangos, cardinalidad y definición explícita del target. Crear o
completar el diccionario de datos.

**Dependencias:** T-03.

**Criterio de completitud:**
- Archivo de contrato/diccionario de datos creado.
- Cada campo tiene tipo, dominio/rango y descripción definidos.

#### T-05

**Descripción:** Realizar un data audit inicial sobre los datos crudos:
detectar tipos reales vs esperados, nulos, valores atípicos, cardinalidad,
duplicados y coherencia. Producir un informe de hallazgos.

**Dependencias:** T-04.

**Criterio de completitud:**
- Informe de audit escrito con hallazgos concretos por campo.
- Hallazgos cruzados con el contrato de datos (desviaciones documentadas).

#### T-06

**Descripción:** Implementar checks de calidad de datos automatizados
(dominios, rangos, tipos, consistencia entre campos, duplicados). Ejecutarlos
sobre el dataset y producir un informe de resultado.

**Dependencias:** T-05.

**Criterio de completitud:**
- Checks ejecutables con resultado explícito (aprobado/rechazado) por regla.
- Pruebas unitarias de los propios checks.
- Dataset que no pasa validación queda bloqueado para la siguiente fase.

---

### Fase 4 — Cleaning y missing values

#### T-07

**Descripción:** Normalizar tipos y nombres, resolver hallazgos del audit
duplicados o de coherencia, y producir un dataset procesado como artefacto
derivado trazable. El raw permanece intacto.

**Dependencias:** T-06.

**Criterio de completitud:**
- Artefacto derivado generado desde raw, con registro de pasos y semillas.
- Raw sin alteraciones.
- Proceso reproducible desde cero.

#### T-08

**Descripción:** Analizar la presencia de valores faltantes, decidir el
tratamiento de cada caso (imputar, descartar, marcar) con justificación
basada en los datos, e implementarlo como parte del pipeline de limpieza.

**Dependencias:** T-07.

**Criterio de completitud:**
- Documento de justificación de cada tratamiento de missing values.
- Implementación reproducible, aplicable de forma idéntica en inferencia.
- Transformaciones integradas al pipeline (no scripts sueltos).

---

### Fase 5 — EDA

#### T-09

**Descripción:** Realizar el EDA completo: perfil univariado y bivariado de
variables, análisis del desbalanceo del target, tasas de churn por segmento,
distribución de variables clave, relación de missing values con el target y
con otras variables. Documentar hallazgos y decisiones que sustentan
feature engineering y preprocessing. El EDA se implementa de forma
**versionada y reproducible** (scripts o notebooks revisables) sobre los datos
procesados de `T-08`; no depende de ejecución manual ni de estado de sesión.

**Dependencias:** T-08.

**Criterio de completitud:**
- Documento de EDA con visualizaciones y tablas, generado y versionado.
- Cada hallazgo relevante vinculado a una decisión posterior (features o
  preprocessing).
- Desbalanceo del target cuantificado y analizado.
- El EDA es reproducible ejecutando su código sobre los datos versionados.

---

### Fase 6 — Feature engineering

#### T-10

**Descripción:** Diseñar e implementar features derivadas del EDA, definiendo
para cada una: origen, justificación, definición y consistencia entre
entrenamiento e inferencia.

**Dependencias:** T-09.

**Criterio de completitud:**
- Cada feature justificada por un hallazgo del EDA.
- Mismas features disponibles en train y en inferencia (mismo esquema).
- Versionado del diccionario de features.

---

### Fase 7 — Particiones y baseline

#### T-11

**Descripción:** Definir y ejecutar las particiones de datos
(entrenamiento/validación/prueba) sobre el dataset con features (`T-10`), con
semillas fijas. Establecer un baseline simple con las métricas definidas en la
sección 7 de `specs.md`.

**Dependencias:** T-10.

**Criterio de completitud:**
- Particiones reproducibles (semillas fijas, mismo resultado en cada
  ejecución).
- Baseline medido y registrado con las métricas del problema.

---

### Fase 8 — Preprocessing

#### T-12

**Descripción:** Implementar el pipeline de transformadores (imputación,
escalado, codificación, selección) ajustados únicamente sobre datos de
entrenamiento, usando las particiones de `T-11`. Estructurar el pipeline para
que aplique de forma idéntica en entrenamiento, validación, prueba e inferencia.

**Dependencias:** T-11.

**Criterio de completitud:**
- Pipeline compuesto como unidad reutilizable.
- Ningún transformador ajustado sobre datos de validación, prueba o inferencia
  (verificable por diseño).
- Pruebas que validan que train, validación y prueba reciben el mismo
  procesamiento.

---

### Fase 9 — Entrenamiento

#### T-13

**Descripción:** Entrenar al menos dos candidatos de modelo distintos,
evaluados bajo las mismas condiciones (partición, métricas, semillas).

**Dependencias:** T-12.

**Criterio de completitud:**
- Múltiples experimentos ejecutados.
- Cada experimento registrado con datos, configuración y métricas.
- Sin evidencia de data leakage en el procedimiento.

---

### Fase 10 — Evaluación y selección del modelo

#### T-14

**Descripción:** Comparar candidatos con métricas adecuadas al problema
desbalanceado (recall, precisión, F1 o AUC-PR sobre la clase de interés).
Seleccionar el modelo y documentar la justificación.

**Dependencias:** T-13, T-17.

**Criterio de completitud:**
- Tabla comparativa de candidatos con métricas consistentes.
- Selección documentada y alineada con `specs.md` sección 7.
- Resultado auditable a partir del tracking de experimentos.

---

### Fase 11 — Pipeline reproducible

#### T-15

**Descripción:** Orquestar el pipeline end-to-end reproducible que ejecuta:
datos → features → modelo → predicción. Comandos documentados para
regenerarlo desde cero.

**Dependencias:** T-14.

**Criterio de completitud:**
- Pipeline ejecutable de extremo a extremo con un único comando o secuencia.
- Documentación del comando de regeneración.
- Mismos resultados al ejecutar con los mismos datos y configuración.

---

### Fase 12 — SQL / PostgreSQL

#### T-16

**Descripción:** Implementar la persistencia de datos procesados y/o
resultados de predicción en PostgreSQL. Definir esquemas, cargar datos y
verificar que son consultables por SQL.

**Dependencias:** T-15.

**Criterio de completitud:**
- Esquemas definidos y datos cargados.
- Consultas de verificación que devuelven resultados coherentes.
- Trazabilidad de los datos persistidos a su origen versionado.

---

### Fase 13 — MLflow

#### T-17

**Descripción:** Integrar MLflow para tracking de experimentos (parámetros,
métricas, artefactos) y registro del modelo seleccionado. Se integra **antes**
de la evaluación y selección (`T-14`) para que los criterios de registro y
auditoría de `T-13`/`T-14` sean verificables.

**Dependencias:** T-13.

**Criterio de completitud:**
- Experimentos visibles en la interfaz de MLflow con métricas y configuración.
- Cada experimento de `T-13` registrado en el tracking.
- Modelo registrado y recuperable para inferencia.
- Artefactos asociados accesibles desde MLflow.

---

### Fase 14 — API con FastAPI

#### T-18

**Descripción:** Desarrollar la API de predicción: endpoint de disponibilidad
del servicio, endpoint de predicción por cliente, respuesta ante errores
explícitos y entrada inválida. Carga el modelo y su pipeline desde
MLflow/almacenamiento.

**Dependencias:** T-15, T-17.

**Criterio de completitud:**
- Endpoints operativos y verificados con requests de prueba.
- Errores devueltos con código y mensaje claro; sin exposición de detalles
  internos.
- Contrato de entrada/salida consistente con `specs.md`.

---

### Fase 15 — Docker

#### T-19

**Descripción:** Crear el contenedor Docker que empaqueta la API y el modelo
seleccionado con su pipeline de datos.

**Dependencias:** T-18.

**Criterio de completitud:**
- Imagen construible sin errores.
- Contenedor funcional que expone los endpoints de la API.

---

### Fase 16 — Testing

#### T-20

**Descripción:** Consolidar y ampliar la suite de tests: unitarios de
transformaciones, integración de la API (contrato de entrada/salida, errores),
validación de datos. Verificar que pasan todos.

**Dependencias:** T-15, T-18.

**Criterio de completitud:**
- Suite de tests completa ejecutable con todos los tests en verde.
- Cobertura mínima sobre transformaciones, pipeline y endpoints.

---

### Fase 17 — CI/CD

#### T-21

**Descripción:** Configurar pipeline de CI que ejecute lint, typecheck y
tests en cada cambio. Definir el pipeline de CD para construcción de imagen
y/o despliegue.

**Dependencias:** T-19, T-20.

**Criterio de completitud:**
- Pipeline CI activo que se ejecuta y pasa en el repo.
- Pipeline CD definido que produce la imagen o artefacto desplegable.

---

### Fase 18 — Monitoring

#### T-22

**Descripción:** Definir qué se monitorea (distribución de features,
distribución de predicciones, desempeño cuando se dispone del target), los
umbrales de alerta y las acciones a tomar ante desviaciones. Implementar la
recolección y el reporte.

**Dependencias:** T-18.

**Criterio de completitud:**
- Documento de monitoring con métricas, umbrales y acciones.
- Script o servicio que produce el reporte de monitoreo.

---

### Fase 19 — Documentación y cierre

#### T-23

**Descripción:** Asegurar que `AGENTS.md`, `constitution.md`, `specs.md` y
`tasks.md` están alineados con el sistema implementado. Completar `README`
con instrucciones de uso, decisión de arquitectura y punto de contacto.

**Dependencias:** todas.

**Criterio de completitud:**
- La documentación refleja fielmente el sistema implementado.
- `README` contiene instrucciones de instalación, uso y arquitectura.
- Los criterios de aceptación de `specs.md` se cumplen.

---

### Fase 20 — Refactor SOLID de la capa de código

#### T-24

**Descripción:** Centralizar la definición de rutas de archivos en
`src/paths.py` y el cálculo de hashes (PEP 247) en `src/hashing.py`, eliminando
las definiciones duplicadas de `PROJECT_ROOT` y los imports cruzados de
`src.data.clean`. Cada módulo conserva sus nombres públicos de rutas (p. ej.
`OUTPUT_FILE`, `LOG_FILE`, `REPORT_FILE`) como alias de las constantes
canónicas, de modo que los tests que parchean atributos de módulo sigan
funcionando sin cambios.

**Dependencias:** T-23.

**Criterio de completitud:**
- `src/paths.py` define todas las rutas canónicas y `src/hashing.py` define
  `sha256_file`.
- Ningún módulo redefine `PROJECT_ROOT` ni importa hashing desde `src.data.clean`.
- Suite completa en verde (pytest, black, ruff, mypy, pre-commit) sin modificar
  el comportamiento de los módulos ni los nombres públicos usados por los tests.

#### T-25

**Descripción:** Aplicar OCP al entrenamiento de candidatos: en
`src/modeling/train.py`, centralizar la selección de modelos en un registro
(`CANDIDATES`) que mapea cada nombre de candidato a un constructor
parametrizable, de modo que añadir un nuevo candidato no requiera modificar
`build_candidate`. Conservar `CANDIDATE_NAMES` y `build_candidate` como API
pública por compatibilidad con los tests y con `tracking.py`.

**Dependencias:** T-24.

**Criterio de completitud:**
- `build_candidate(name, seed, **overrides)` resuelve el modelo desde
  `CANDIDATES` y lanza `ValueError` para nombres desconocidos.
- `CANDIDATE_NAMES` sigue disponible y refleja las claves del registro.
- Suite completa en verde sin cambios de comportamiento en el entrenamiento.

#### T-26

**Descripción:** Aplicar OCP a los checks de calidad de datos: en
`src/data/quality_checks.py`, centralizar la ejecución de reglas en un
registro (`CHECKS`) que mapea cada `rule_id` a su función de validación, de
modo que añadir una regla nueva no requiera modificar `run_all`. Conservar
`run_all` y `build_report` como API pública.

**Dependencias:** T-24.

**Criterio de completitud:**
- `run_all` ejecuta únicamente los checks registrados en `CHECKS`, en orden.
- `run_all` y `build_report` mantienen su firma y comportamiento.
- Suite completa en verde sin cambios de comportamiento en los checks.

#### T-27

**Descripción:** Aplicar responsabilidad única y delegación al servicio de
predicción: extraer la lógica de inferencia a `src/serving/prediction_service.py`
(`PredictionService` con carga lazy del modelo y método `predict`), dejando a
`src/serving/api.py` solo la validación del contrato HTTP y el manejo de
errores. `create_app` acepta un servicio inyectable y conserva
`load_pipeline`/`threshold` por compatibilidad.

**Dependencias:** T-25.

**Criterio de completitud:**
- `PredictionService` encapsula feature engineering + predicción y carga el
  modelo una única vez.
- `api.py` delega en el servicio; los endpoints y códigos de error
  (`invalid_request`, `model_unavailable`, `internal_error`) no cambian.
- Suite completa en verde; los tests existentes de la API siguen pasando.
