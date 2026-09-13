# Baseline (T-11)

Clasificador *dummy* con estrategia `prior` (predice la probabilidad previa de churn para todos los clientes), ajustado solo sobre `train` (4929 filas) y evaluado sobre `validation` (1057 filas).

| Métrica | Valor (clase `Yes`) |
|---|---|
| prevalence (train) | 0.2654 |
| AUC-ROC | 0.5000 |
| AUC-PR | 0.2658 |
| Recall | 0.0000 |
| Precisión | 0.0000 |
| F1 | 0.0000 |
| Accuracy (referencia) | 0.7342 |

Matriz de confusión (umbral 0.5):

- TP=0, FN=281, FP=0, TN=776.

Interpretación: al predecir siempre la clase previa, el umbral 0.5 no predice ningún `Yes` (recall 0) y AUC-ROC ≈ 0.5; AUC-PR coincide con la prevalencia. Es el valor de referencia que los candidatos de `T-13` deben superar. La partición `test` queda reservada para la selección final (`T-14`).
