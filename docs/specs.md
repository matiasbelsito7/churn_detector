# specs.md

## 1. Propósito y alcance

Este documento define la especificación funcional y técnica del sistema de
detección y predicción de churn. Describe **qué** debe cumplir el sistema, no
**cómo** implementarlo.

Su punto de entrada es `AGENTS.md`; se apoya en los principios de
`constitution.md`. La descomposición del trabajo en tareas es responsabilidad
de `tasks.md`.

## 2. Objetivo del sistema

El sistema debe detectar y predecir la probabilidad de churn de un cliente,
de modo que un área de negocio pueda tomar acciones de retención antes de que
se produzca la baja.

Objetivos que debe cumplir:

- Producir, por cada cliente, una **probabilidad de churn** y la **clase
  predicha** derivada de ella.
- Soportar tanto **predicción por cliente** (consulta puntual) como **evaluación
  de cohortes o del conjunto** (análisis y reporting).

## 3. Problema de negocio

Se define como problema de **clasificación binaria desbalanceada**: la clase de
interés (churn) es minoritaria respecto a la clase de no-churn.

Requisito derivado: la evaluación y selección del sistema debe reflejar el
desempeño sobre la clase de interés y no quedar enmascarada por el balance de
las clases.

## 4. Dataset y target

- Fuente de datos principal: dataset de telco referencial (IBM Telco Customer
  Churn). Los detalles concretos del dataset se validan y documentan durante la
  fase de datos, no se asumen aquí.
- **Target**: variable binaria de churn del cliente. Debe definirse de forma
  explícita y coherente con el problema de negocio, y documentarse como parte
  del contrato de datos.
- Requisito: ningún artefacto (EDA, modelo, monitoreo) puede producirse sobre
  el dataset sin que este haya pasado la validación de calidad definida en la
  sección 5.

## 5. Requisitos de datos

### 5.1 Ingestion

- El sistema debe adquirir los datos desde su origen sin alterarlos: los **datos
  crudos no se modifican**.
- Cada import debe ser **reproducible** y quedar **trazable** (fecha, origen,
  versión o código de la fuente).
- Los datos crudos se **versionan**; toda transformación posterior debe operar
  sobre copias derivadas.

### 5.2 Data quality

- El sistema debe validar los datos con checks explícitos antes de usarlos:
  tipos de datos, dominios y rangos válidos, cardinalidad y valores únicos,
  duplicados y coherencia interna entre campos.
- Toda validación debe producir un **informe de calidad** con el resultado
  (aprobado/no aprobado) y las anomalías detectadas.
- Un dataset que no pasa validación **no avanza** a modelado.

### 5.3 Curation

- Debe existir un **diccionario de datos** y un **contrato de datos** (esquema
  esperado: columnas, tipos, dominios, target) que sirvan de base para las
  validaciones y para el resto de las capas.
- Los nombres, tipos y unidades de los campos deben ser **consistentes** entre
  conjuntos (entrenamiento e inferencia).

### 5.4 Tratamiento de valores faltantes

- El tratamiento debe ser **explícito y justificable**: cada decisión
  (descartar, imputar, marcar) se documenta con su motivo y su evidencia.
- No se aplica un método de imputación por defecto; la elección se deriva del
  análisis de los datos.
- Las reglas de imputación/tratamiento deben **reproducirse** en inferencia con
  exactitud respecto a las definidas en entrenamiento.

### 5.5 EDA requerido

El sistema debe producir (y el equipo documentar) análisis que sustenten las
decisiones posteriores:

- Perfil **univariado** de cada variable: distribución, valores extremos,
  atípicos y dominios.
- Perfil **bivariado**: relación de cada variable (o grupo) con el target, y
  correlaciones relevantes entre variables.
- Análisis del **desbalanceo** del target y de las tasas de churn por segmento.
- Análisis de **datos faltantes**: cantidad, patrón y relación con el target y
  con otras variables.
- Distribución de variables clave de dinero/duración del servicio y su relación
  con el target.
- Cada decisión posterior (features, preprocessing, modelo) debe poder trazarse
  a un hallazgo del EDA.

## 6. Feature engineering y preprocessing

- Las **features** a construir o transformar se derivan del EDA; ninguna se
  introduce sin evidencia que la justifique.
- El esquema de features debe ser **idéntico entre entrenamiento e inferencia**
  (mismo nombre, definición y orden).
- Todo transformador (imputación, escalado, codificación, selección) se **ajusta
  únicamente con datos de entrenamiento**; el flujo debe garantizarlo
  estructuralmente para prevenir data leakage u optimismo de validación.
- Las transformaciones deben componerse en **pipelines** que se apliquen de
  forma idéntica en entrenamiento, validación e inferencia.

## 7. Modelado, evaluación y selección

- El sistema debe partir de un **baseline** simple y medible antes de modelos
  complejos.
- Deben definirse **particiones** de datos (entrenamiento/validación/prueba)
  reproducibles, con semillas fijas.
- La evaluación debe usar **métricas adecuadas al problema desbalanceado**:
  recall, precisión y/o métricas derivadas (F1, AUC-PR) sobre la clase de
  interés; métricas globales como accuracy no son criterio suficiente por sí
  solas.
- La **selección del modelo** se realiza comparando candidatos bajo la misma
  partición, las mismas métricas y las mismas condiciones de ejecución.
- Cada experimento debe quedar registrado (ver tracking, sección 9.3) con su
  configuración, datos, métricas y artefactos, de modo que la evaluación sea
  auditable.

## 8. Inferencia y API

### 8.1 Inferencia

- El sistema debe producir predicciones sobre clientes **sin que se conozca el
  target**, aplicando exactamente el mismo procesamiento que en entrenamiento.
- Salida definida: **probabilidad de churn**, **clase predicha** e identificador
  del cliente. La salida debe ser interpretable y consumible por un sistema de
  negocio.

### 8.2 Requisitos mínimos de API

- **Funcional** (sin restricción tecnológica impuesta):
  - Exponer un endpoint de **verificación de disponibilidad** del servicio.
  - Exponer un endpoint de **predicción por cliente** que reciba los campos de
    entrada y devuelva probabilidad, clase e identificador.
  - Devolver **errores explícitos** ante entrada inválida o incompleta (código
    de error, mensaje claro), sin exponer detalles internos.
- El artefacto de inferencia debe ser el modelo seleccionado en la sección 7,
  con su pipeline de preprocessing asociado.

## 9. Industrialización

### 9.1 Persistencia y SQL

- El sistema debe poder **persistir** estructuras de datos relevantes (datos
  procesados y/o resultados de predicción) en una base de datos relacional
  (PostgreSQL).
- Los datos persistidos deben ser **consultables** mediante SQL y **trazables**
  a su origen versionado.

### 9.2 Reproducibilidad

- El sistema debe permitir regenerar su pipeline completo (datos → features →
  modelo → predicción) a partir de entradas versionadas.
- Las **dependencias** (bibliotecas y versiones) deben estar declaradas y
  reproducibles.
- **Semillas fijas** en toda ejecución que las requiera.

### 9.3 Tracking de experimentos

- El sistema debe registrar los experimentos de modelado (parámetros, métricas,
  artefactos y modelo) en una herramienta de tracking (MLflow).
- Los modelos candidatos y el modelo seleccionado deben poder **localizarse y
  recuperarse** desde el tracking para inferencia o auditoría.

### 9.4 Testing

- Funcionalidad nueva: **pruebas unitarias** de las transformaciones y de los
  pipelines (reproducibilidad y consistencia train/inferencia).
- **Pruebas de integración** del servicio de predicción (contrato de entrada y
  salida, manejo de errores).
- **Validación de datos** (checks de calidad) con sus propias pruebas.

### 9.5 Monitoring

- El sistema en producción debe permitir observar su **desempeño** y la
  **distribución de sus entradas** (detección de drift sobre features y, cuando
  aplique, sobre el target).
- Debe existir una definición de **qué se monitorea, qué umbral alerta y qué se
  hace ante la alerta**.

## 10. Criterios de aceptación

El sistema se considera conforme a esta especificación cuando:

- Los datos crudos están **versionados** y la ingestión es reproducible y trazable.
- Existe contrato, diccionario de datos e **informe de validación de calidad**
  para el dataset utilizado.
- Los valores faltantes están tratados con decisiones **documentadas y
  justificadas**.
- El EDA está documentado y sus hallazgos sustentan las decisiones de
  features/preprocessing.
- El pipeline de datos → features → modelo está **reproducido de extremo a
  extremo** y el mismo procesamiento se aplica en inferencia.
- Existe un baseline y un **modelo seleccionado con criterios comparables**, con
  su evaluación registrada y sin evidencia de data leakage.
- La API expone los endpoints mínimos y responde errores explícitos.
- Los experimentos están **trackeados**, los datos procesados persisten en la
  base relacional y son consultables.
- La funcionalidad nueva cuenta con **pruebas** que se ejecutan y pasan.
- Está documentado el **plan de monitoreo** con umbrales y acciones.
- La documentación de `AGENTS.md`/`constitution.md`/`specs.md`/`tasks.md` está
  alineada con el sistema implementado.
