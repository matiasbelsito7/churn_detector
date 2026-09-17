# Búsqueda de hiperparámetros (exploratoria)

Barrido de configuraciones por familia de modelo sobre la partición `validation` (mismo preprocessing y métricas que `T-13`; `test` queda reservado para `T-14`). Cada familia incluye su fila `default` (configuración oficial de `T-13`) como referencia. Ordenada por AUC-PR (desempate: recall).

| Modelo | Parámetros | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |
|---|---|---|---|---|---|---|---|
| logistic-regression | C=0.01 | 0.8474 | 0.6777 | 0.7758 | 0.5129 | 0.6176 | 0.7446 |
| logistic-regression | C=0.1 | 0.8431 | 0.6699 | 0.8043 | 0.5056 | 0.6209 | 0.7389 |
| logistic-regression | C=10.0 | 0.8415 | 0.6690 | 0.7972 | 0.5045 | 0.6179 | 0.7379 |
| logistic-regression | default | 0.8421 | 0.6686 | 0.8043 | 0.5056 | 0.6209 | 0.7389 |
| logistic-regression | C=1.0 | 0.8421 | 0.6686 | 0.8043 | 0.5056 | 0.6209 | 0.7389 |
| xgboost | learning_rate=0.05, max_depth=3, n_estimators=200, subsample=1.0 | 0.8425 | 0.6536 | 0.7972 | 0.5103 | 0.6222 | 0.7427 |
| random-forest | max_depth=10, min_samples_leaf=3, n_estimators=200 | 0.8443 | 0.6517 | 0.7651 | 0.5257 | 0.6232 | 0.7540 |
| xgboost | learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.8 | 0.8425 | 0.6507 | 0.7829 | 0.5069 | 0.6154 | 0.7398 |
| xgboost | learning_rate=0.05, max_depth=3, n_estimators=400, subsample=1.0 | 0.8406 | 0.6505 | 0.7900 | 0.5127 | 0.6218 | 0.7446 |
| random-forest | max_depth=10, min_samples_leaf=5, n_estimators=200 | 0.8433 | 0.6494 | 0.7794 | 0.5227 | 0.6257 | 0.7521 |
| random-forest | max_depth=10, min_samples_leaf=5, n_estimators=400 | 0.8431 | 0.6458 | 0.7829 | 0.5263 | 0.6295 | 0.7550 |
| xgboost | learning_rate=0.05, max_depth=3, n_estimators=400, subsample=0.8 | 0.8415 | 0.6458 | 0.7758 | 0.5154 | 0.6193 | 0.7465 |
| xgboost | learning_rate=0.1, max_depth=3, n_estimators=200, subsample=1.0 | 0.8395 | 0.6450 | 0.7900 | 0.5139 | 0.6227 | 0.7455 |
| random-forest | max_depth=10, min_samples_leaf=3, n_estimators=400 | 0.8429 | 0.6444 | 0.7651 | 0.5231 | 0.6214 | 0.7521 |
| xgboost | learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.8 | 0.8401 | 0.6419 | 0.7722 | 0.5154 | 0.6182 | 0.7465 |
| random-forest | max_depth=20, min_samples_leaf=3, n_estimators=200 | 0.8400 | 0.6394 | 0.7367 | 0.5294 | 0.6161 | 0.7559 |
| random-forest | max_depth=None, min_samples_leaf=3, n_estimators=200 | 0.8398 | 0.6393 | 0.7367 | 0.5267 | 0.6142 | 0.7540 |
| xgboost | learning_rate=0.1, max_depth=3, n_estimators=400, subsample=1.0 | 0.8338 | 0.6382 | 0.7687 | 0.5180 | 0.6189 | 0.7483 |
| xgboost | learning_rate=0.05, max_depth=6, n_estimators=200, subsample=1.0 | 0.8400 | 0.6373 | 0.7438 | 0.5305 | 0.6193 | 0.7569 |
| random-forest | max_depth=20, min_samples_leaf=3, n_estimators=400 | 0.8402 | 0.6363 | 0.7402 | 0.5266 | 0.6154 | 0.7540 |
| random-forest | default | 0.8415 | 0.6362 | 0.7687 | 0.5243 | 0.6234 | 0.7531 |
| random-forest | max_depth=None, min_samples_leaf=3, n_estimators=400 | 0.8401 | 0.6362 | 0.7402 | 0.5293 | 0.6172 | 0.7559 |
| xgboost | learning_rate=0.05, max_depth=6, n_estimators=200, subsample=0.8 | 0.8362 | 0.6359 | 0.7153 | 0.5317 | 0.6100 | 0.7569 |
| random-forest | max_depth=None, min_samples_leaf=5, n_estimators=400 | 0.8412 | 0.6358 | 0.7651 | 0.5218 | 0.6205 | 0.7512 |
| random-forest | max_depth=20, min_samples_leaf=5, n_estimators=400 | 0.8411 | 0.6358 | 0.7687 | 0.5243 | 0.6234 | 0.7531 |
| random-forest | max_depth=20, min_samples_leaf=5, n_estimators=200 | 0.8414 | 0.6354 | 0.7758 | 0.5317 | 0.6310 | 0.7588 |
| random-forest | max_depth=None, min_samples_leaf=5, n_estimators=200 | 0.8416 | 0.6353 | 0.7758 | 0.5317 | 0.6310 | 0.7588 |
| xgboost | learning_rate=0.1, max_depth=3, n_estimators=400, subsample=0.8 | 0.8353 | 0.6320 | 0.7438 | 0.5173 | 0.6102 | 0.7474 |
| xgboost | learning_rate=0.05, max_depth=6, n_estimators=400, subsample=0.8 | 0.8294 | 0.6240 | 0.6833 | 0.5486 | 0.6086 | 0.7663 |
| xgboost | default | 0.8334 | 0.6234 | 0.7260 | 0.5383 | 0.6182 | 0.7616 |
| xgboost | learning_rate=0.1, max_depth=6, n_estimators=200, subsample=1.0 | 0.8318 | 0.6135 | 0.7082 | 0.5408 | 0.6133 | 0.7625 |
| xgboost | learning_rate=0.05, max_depth=6, n_estimators=400, subsample=1.0 | 0.8308 | 0.6129 | 0.7082 | 0.5393 | 0.6123 | 0.7616 |
| xgboost | learning_rate=0.1, max_depth=6, n_estimators=400, subsample=1.0 | 0.8238 | 0.6077 | 0.6512 | 0.5596 | 0.6020 | 0.7711 |
| xgboost | learning_rate=0.1, max_depth=6, n_estimators=200, subsample=0.8 | 0.8251 | 0.6044 | 0.6868 | 0.5406 | 0.6050 | 0.7616 |
| xgboost | learning_rate=0.1, max_depth=6, n_estimators=400, subsample=0.8 | 0.8163 | 0.6024 | 0.6441 | 0.5656 | 0.6023 | 0.7739 |

Mejor configuración por familia:

- **logistic-regression**: C=0.01 → AUC-PR 0.6777, recall 0.7758.
- **xgboost**: learning_rate=0.05, max_depth=3, n_estimators=200, subsample=1.0 → AUC-PR 0.6536, recall 0.7972.
- **random-forest**: max_depth=10, min_samples_leaf=3, n_estimators=200 → AUC-PR 0.6517, recall 0.7651.

Candidato global: **logistic-regression** con C=0.01 → AUC-PR 0.6777, recall 0.7758. Referencia de producción: logistic-regression (default) AUC-PR 0.6686, recall 0.8043 (selección T-14).

Detalle completo (parámetros, métricas, semillas y hashes de datos) en `reports/grid_search.json`. La adopción de una configuración ganadora implicaría proponer la actualización de `docs/tasks.md` y reejecutar `T-17`/`T-14`.
