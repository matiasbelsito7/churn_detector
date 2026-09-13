# Entrenamiento de candidatos (T-13)

Candidatos entrenados bajo condiciones idénticas: partición `train` de `T-11`, pipeline de preprocessing de `T-12` (ajustado solo en train) y evaluación sobre `validation` con las métricas del problema. La partición `test` queda reservada para `T-14`.

| Modelo | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |
|---|---|---|---|---|---|---|
| logistic-regression | 0.8421 | 0.6686 | 0.8043 | 0.5056 | 0.6209 | 0.7389 |
| random-forest | 0.8415 | 0.6362 | 0.7687 | 0.5243 | 0.6234 | 0.7531 |

Referencia (baseline `T-11`): AUC-PR 0.2658, recall 0.0 a umbral 0.5. El detalle de configuración y datos de cada experimento vive en `reports/experiments.json`.
