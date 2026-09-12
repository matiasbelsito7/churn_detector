# constitution.md

## 1. Propósito

Este documento define los principios y reglas que gobiernan el desarrollo del
proyecto. Derivados del contexto de `AGENTS.md`, son el fundamento sobre el que
se definen los requisitos en `specs.md` y el roadmap en `tasks.md`.

Los principios son concretos y aplicables: están pensados para decidir entre
alternativas durante el desarrollo. Ante una decisión técnica, estos principios
tienen prioridad sobre criterios personales o de conveniencia.

## 2. Principios de datos y reproducibilidad

- **El dato es lo primero**: no se entrena ni despliega ningún modelo sobre
  datos que no hayan sido comprendidos, curados y validados. La calidad y
  comprensión de los datos tienen prioridad sobre el modelo y la
  infraestructura.
- **Reproducibilidad**: cualquier import, transformación o entrenamiento debe
  poder regenerarse a partir de datos versionados y dependencias declaradas. Se
  fijan semillas aleatorias y versiones de librerías cuando el resultado lo
  requiera.
- **Trazabilidad**: se registra el origen de los datos, el historial de sus
  transformaciones y la justificación de cada decisión de cleaning. Los datos
  crudos no se modifican; toda limpieza se produce como artefacto derivado
  reproducible.

## 3. Principios de calidad y preprocessing

- **Validación de calidad de datos**: antes de entrenar, los datos se someten a
  checks explícitos (tipos, dominios, rangos, valores únicos, coherencia con el
  contrato de datos). Un dataset que no pasa validación no avanza a modelado.
- **Tratamiento explícito y justificable de valores faltantes**: cada decisión
  sobre missing values (imputar, descartar, marcar) se documenta con su motivo
  y se respalda con evidencia; no se aplica un método por defecto.
- **Preprocessing basado en evidencia**: las transformaciones se deciden a
  partir del EDA y de los datos observados, no por costumbre ni heurística no
  verificada.

## 4. Principios de diseño

- **Separación de responsabilidades**: datos, lógica de negocio y modelos
  residen en capas (o espacios) separados; el código de una capa no se mezcla
  con el de otra.
- **Prevención de data leakage**: los ajustes de transformadores, imputaciones
  y escalado se realizan únicamente con datos de entrenamiento. El flujo lo
  garantiza estructuralmente (pipelines/componentes), no por disciplina manual.
- **No sobre-ingeniería**: se elige la alternativa más simple que cumple la
  necesidad del sistema sin violar los principios de este documento.

## 5. Principios de modelado

- **Evaluación con métricas adecuadas al problema**: dado un problema de churn
  desbalanceado, la evaluación prioriza métricas que reflejen el desempeño en
  la clase minoritaria (p. ej. recall, precisión, AUC-PR o F1) frente a métricas
  globales como accuracy.
- **Selección de modelo con criterios comparables**: los modelos compiten bajo
  la misma partición de datos, las mismas métricas y las mismas semillas; la
  selección se basa en el resultado de esa comparación.
- **Auditabilidad de experimentos**: cada corrida registra los datos y
  transformaciones usados, la configuración, las métricas y los artefactos
  generados (tracking de experimentos).

## 6. Principios de código

- **Testeable, mantenible y documentado**: toda funcionalidad nueva se acompaña
  de pruebas y documentación mínima; el código se mantiene legible y respeta las
  convenciones del proyecto.
- **Cambios verificados**: ningún cambio se da por completado sin correr las
  pruebas, lint y/o typecheck correspondientes.
