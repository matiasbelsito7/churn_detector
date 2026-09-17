# Selección del modelo (T-14)

Procedimiento: los candidatos de `T-13` se comparan con las mismas métricas sobre `validation` (registradas en el tracking de MLflow por `T-17`). El ganador se carga desde el Model Registry como `churn-logistic-regression/production`, se evalúa sobre `test` (partición usada una sola vez) y la corrida queda registrada.

| Modelo | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |
|---|---|---|---|---|---|---|
| logistic-regression | 0.8421 | 0.6686 | 0.8043 | 0.5056 | 0.6209 | 0.7389 |
| random-forest | 0.8415 | 0.6362 | 0.7687 | 0.5243 | 0.6234 | 0.7531 |

Candidato seleccionado: **logistic-regression**.

Justificación: Métrica primaria: AUC-PR sobre la clase `Yes`; desempate por recall sobre `Yes`. Métricas adecuadas al problema desbalanceado (specs.md §7).

Evaluación del seleccionado sobre `test` (1057 filas, sha256 `7f1688151384d94b7dba71461511ef3035a78dc2ee0b5d45f6bed63d69a8c241`):

| Métrica | Valor (clase `Yes`) |
|---|---|
| AUC-ROC | 0.8560 |
| AUC-PR | 0.6698 |
| Recall | 0.8071 |
| Precisión | 0.5033 |
| F1 | 0.6200 |
| Accuracy (referencia) | 0.7379 |

Matriz de confusión (umbral 0.5):

- TP=226, FN=54, FP=223, TN=554.

Auditabilidad: experimento `churn-detector` en el tracking de MLflow (candidatos con su configuración, métricas y artefactos); modelo `churn-logistic-regression` con alias `production`. El registro en `reports/experiments.json` conserva la trazabilidad de datos.

---
---

## Reversión de adopción (T-30)

El candidato adoptado en `T-30` no superó al champion anterior sobre `test` (AUC-PR peor); la adopción se revirtió y el champion volvió a la configuración por defecto de `T-14`. Detalle y evidencia: `reports/adoption.md`.

Champion vigente: `churn-logistic-regression` v3.
