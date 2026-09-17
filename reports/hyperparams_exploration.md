# Análisis exploratorio de hiperparámetros

Método sobre `validation` (mismo preprocessing y métricas que `T-13`; `test` reservado para `T-14`): barrido univariado (un hiperparámetro por vez, resto en default) y búsqueda aleatoria conjunta con semilla fija `42`.

## logistic-regression

Default: AUC-PR 0.6686, recall 0.8043.

### Univariado (un parámetro a la vez)

- `C`:    0.001 -> APR 0.6817 / rec 0.7794;     0.01 -> APR 0.6777 / rec 0.7758;      0.1 -> APR 0.6699 / rec 0.8043;      1.0 -> APR 0.6686 / rec 0.8043;     10.0 -> APR 0.6690 / rec 0.7972;    100.0 -> APR 0.6678 / rec 0.8007
  Mejor: 0.001 (APR 0.6817, rec 0.7794).
- `l1_ratio`:      0.0 -> APR 0.6684 / rec 0.8043;      0.5 -> APR 0.6689 / rec 0.8078;      1.0 -> APR 0.6689 / rec 0.8078
  Mejor: 1.0 (APR 0.6689, rec 0.8078).
- `class_weight`: balanced -> APR 0.6686 / rec 0.8043;     None -> APR 0.6685 / rec 0.5409
  Mejor: balanced (APR 0.6686, rec 0.8043).

### Búsqueda aleatoria conjunta

| Parámetros | AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |
|---|---|---|---|---|---|
| C=0.01, class_weight=balanced, l1_ratio=1.0, solver=saga | 0.6838 | 0.8007 | 0.5022 | 0.6173 | 0.7360 |
| C=0.01, class_weight=None, l1_ratio=0.5, solver=saga | 0.6828 | 0.4875 | 0.6850 | 0.5696 | 0.8042 |
| C=0.01, class_weight=None, l1_ratio=1.0, solver=saga | 0.6821 | 0.4413 | 0.7006 | 0.5415 | 0.8013 |
| C=0.01, class_weight=balanced, l1_ratio=0.0, solver=saga | 0.6777 | 0.7758 | 0.5129 | 0.6176 | 0.7446 |
| C=0.1, class_weight=None, l1_ratio=1.0, solver=saga | 0.6760 | 0.5409 | 0.6756 | 0.6008 | 0.8089 |
| C=0.1, class_weight=balanced, l1_ratio=1.0, solver=saga | 0.6740 | 0.8043 | 0.5102 | 0.6243 | 0.7427 |
| C=0.1, class_weight=balanced, l1_ratio=0.5, solver=saga | 0.6724 | 0.8078 | 0.5078 | 0.6236 | 0.7408 |
| C=0.1, class_weight=balanced, l1_ratio=0.0, solver=saga | 0.6699 | 0.8043 | 0.5056 | 0.6209 | 0.7389 |
| C=10.0, class_weight=None, l1_ratio=0.5, solver=saga | 0.6690 | 0.5409 | 0.6726 | 0.5996 | 0.8079 |
| C=10.0, class_weight=None, l1_ratio=1.0, solver=saga | 0.6690 | 0.5409 | 0.6726 | 0.5996 | 0.8079 |
| C=1.0, class_weight=balanced, l1_ratio=0.5, solver=saga | 0.6689 | 0.8078 | 0.5067 | 0.6228 | 0.7398 |
| C=100.0, class_weight=None, l1_ratio=0.5, solver=saga | 0.6686 | 0.5302 | 0.6652 | 0.5901 | 0.8042 |
| C=100.0, class_weight=None, l1_ratio=1.0, solver=saga | 0.6686 | 0.5302 | 0.6652 | 0.5901 | 0.8042 |
| C=1.0, class_weight=balanced, l1_ratio=0.0, solver=saga | 0.6684 | 0.8043 | 0.5056 | 0.6209 | 0.7389 |
| C=10.0, class_weight=balanced, l1_ratio=0.0, solver=saga | 0.6683 | 0.7972 | 0.5045 | 0.6179 | 0.7379 |
| C=1.0, class_weight=None, l1_ratio=0.0, solver=saga | 0.6681 | 0.5409 | 0.6726 | 0.5996 | 0.8079 |
| C=1.0, class_weight=None, l1_ratio=0.5, solver=saga | 0.6680 | 0.5409 | 0.6667 | 0.5972 | 0.8061 |
| C=0.001, class_weight=balanced, l1_ratio=0.5, solver=saga | 0.5869 | 0.7367 | 0.4737 | 0.5766 | 0.7124 |
| C=0.001, class_weight=balanced, l1_ratio=1.0, solver=saga | 0.5016 | 0.6762 | 0.4774 | 0.5596 | 0.7171 |
| C=0.001, class_weight=None, l1_ratio=1.0, solver=saga | 0.2658 | 0.0000 | 0.0000 | 0.0000 | 0.7342 |

## random-forest

Default: AUC-PR 0.6362, recall 0.7687.

### Univariado (un parámetro a la vez)

- `n_estimators`:       50 -> APR 0.6391 / rec 0.7722;      100 -> APR 0.6416 / rec 0.7687;      200 -> APR 0.6353 / rec 0.7758;      300 -> APR 0.6362 / rec 0.7687;      400 -> APR 0.6358 / rec 0.7651;      600 -> APR 0.6362 / rec 0.7651
  Mejor: 100 (APR 0.6416, rec 0.7687).
- `max_depth`:     None -> APR 0.6362 / rec 0.7687;        5 -> APR 0.6605 / rec 0.8043;       10 -> APR 0.6489 / rec 0.7722;       15 -> APR 0.6404 / rec 0.7580;       20 -> APR 0.6360 / rec 0.7687;       30 -> APR 0.6362 / rec 0.7687
  Mejor: 5 (APR 0.6605, rec 0.8043).
- `min_samples_split`:        2 -> APR 0.6362 / rec 0.7687;        5 -> APR 0.6362 / rec 0.7687;       10 -> APR 0.6362 / rec 0.7687;       20 -> APR 0.6455 / rec 0.7687
  Mejor: 20 (APR 0.6455, rec 0.7687).
- `min_samples_leaf`:        1 -> APR 0.5979 / rec 0.6512;        2 -> APR 0.6243 / rec 0.7260;        5 -> APR 0.6362 / rec 0.7687;       10 -> APR 0.6478 / rec 0.7829
  Mejor: 10 (APR 0.6478, rec 0.7829).
- `max_features`:     sqrt -> APR 0.6362 / rec 0.7687;     log2 -> APR 0.6455 / rec 0.7722;     None -> APR 0.6199 / rec 0.7331
  Mejor: log2 (APR 0.6455, rec 0.7722).
- `class_weight`: balanced -> APR 0.6362 / rec 0.7687;     None -> APR 0.6445 / rec 0.5160
  Mejor: None (APR 0.6445, rec 0.5160).
- `criterion`:     gini -> APR 0.6362 / rec 0.7687;  entropy -> APR 0.6442 / rec 0.7651
  Mejor: entropy (APR 0.6442, rec 0.7651).
- `bootstrap`:     True -> APR 0.6362 / rec 0.7687;    False -> APR 0.6357 / rec 0.7224
  Mejor: True (APR 0.6362, rec 0.7687).

### Búsqueda aleatoria conjunta

| Parámetros | AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |
|---|---|---|---|---|---|
| bootstrap=True, class_weight=balanced, criterion=gini, max_depth=5, max_features=sqrt, min_samples_leaf=10, min_samples_split=2, n_estimators=400 | 0.6594 | 0.8007 | 0.5068 | 0.6207 | 0.7398 |
| bootstrap=False, class_weight=balanced, criterion=gini, max_depth=5, max_features=log2, min_samples_leaf=1, min_samples_split=5, n_estimators=300 | 0.6588 | 0.7900 | 0.5045 | 0.6158 | 0.7379 |
| bootstrap=True, class_weight=None, criterion=gini, max_depth=5, max_features=sqrt, min_samples_leaf=2, min_samples_split=10, n_estimators=400 | 0.6560 | 0.4911 | 0.6635 | 0.5644 | 0.7985 |
| bootstrap=True, class_weight=balanced, criterion=gini, max_depth=10, max_features=log2, min_samples_leaf=10, min_samples_split=20, n_estimators=400 | 0.6556 | 0.7900 | 0.5163 | 0.6245 | 0.7474 |
| bootstrap=True, class_weight=balanced, criterion=entropy, max_depth=5, max_features=log2, min_samples_leaf=10, min_samples_split=20, n_estimators=50 | 0.6548 | 0.7936 | 0.5150 | 0.6246 | 0.7465 |
| bootstrap=True, class_weight=None, criterion=entropy, max_depth=15, max_features=log2, min_samples_leaf=10, min_samples_split=2, n_estimators=100 | 0.6542 | 0.5053 | 0.6311 | 0.5613 | 0.7900 |
| bootstrap=True, class_weight=balanced, criterion=entropy, max_depth=None, max_features=sqrt, min_samples_leaf=10, min_samples_split=10, n_estimators=100 | 0.6534 | 0.7865 | 0.5225 | 0.6278 | 0.7521 |
| bootstrap=False, class_weight=None, criterion=gini, max_depth=10, max_features=sqrt, min_samples_leaf=10, min_samples_split=5, n_estimators=300 | 0.6507 | 0.5125 | 0.6316 | 0.5658 | 0.7909 |
| bootstrap=False, class_weight=None, criterion=gini, max_depth=15, max_features=log2, min_samples_leaf=10, min_samples_split=20, n_estimators=300 | 0.6506 | 0.5018 | 0.6351 | 0.5606 | 0.7909 |
| bootstrap=False, class_weight=None, criterion=entropy, max_depth=10, max_features=sqrt, min_samples_leaf=10, min_samples_split=10, n_estimators=200 | 0.6505 | 0.5196 | 0.6348 | 0.5714 | 0.7928 |
| bootstrap=False, class_weight=None, criterion=gini, max_depth=20, max_features=log2, min_samples_leaf=10, min_samples_split=2, n_estimators=300 | 0.6503 | 0.5018 | 0.6380 | 0.5618 | 0.7919 |
| bootstrap=True, class_weight=balanced, criterion=gini, max_depth=30, max_features=sqrt, min_samples_leaf=10, min_samples_split=2, n_estimators=200 | 0.6488 | 0.7794 | 0.5190 | 0.6230 | 0.7493 |
| bootstrap=True, class_weight=balanced, criterion=entropy, max_depth=30, max_features=log2, min_samples_leaf=5, min_samples_split=5, n_estimators=300 | 0.6478 | 0.7687 | 0.5217 | 0.6216 | 0.7512 |
| bootstrap=False, class_weight=balanced, criterion=entropy, max_depth=10, max_features=sqrt, min_samples_leaf=5, min_samples_split=20, n_estimators=300 | 0.6477 | 0.7544 | 0.5183 | 0.6145 | 0.7483 |
| bootstrap=True, class_weight=balanced, criterion=gini, max_depth=15, max_features=log2, min_samples_leaf=5, min_samples_split=2, n_estimators=100 | 0.6464 | 0.7651 | 0.5181 | 0.6178 | 0.7483 |
| bootstrap=False, class_weight=None, criterion=entropy, max_depth=10, max_features=log2, min_samples_leaf=1, min_samples_split=5, n_estimators=400 | 0.6455 | 0.5160 | 0.6332 | 0.5686 | 0.7919 |
| bootstrap=True, class_weight=None, criterion=gini, max_depth=None, max_features=log2, min_samples_leaf=5, min_samples_split=20, n_estimators=50 | 0.6454 | 0.5018 | 0.6380 | 0.5618 | 0.7919 |
| bootstrap=True, class_weight=balanced, criterion=entropy, max_depth=10, max_features=None, min_samples_leaf=10, min_samples_split=10, n_estimators=400 | 0.6434 | 0.7794 | 0.5341 | 0.6339 | 0.7606 |
| bootstrap=False, class_weight=balanced, criterion=gini, max_depth=10, max_features=sqrt, min_samples_leaf=2, min_samples_split=2, n_estimators=600 | 0.6430 | 0.7402 | 0.5306 | 0.6181 | 0.7569 |
| bootstrap=False, class_weight=balanced, criterion=gini, max_depth=20, max_features=sqrt, min_samples_leaf=10, min_samples_split=2, n_estimators=50 | 0.6409 | 0.7616 | 0.5271 | 0.6230 | 0.7550 |
| bootstrap=True, class_weight=None, criterion=gini, max_depth=15, max_features=sqrt, min_samples_leaf=5, min_samples_split=5, n_estimators=100 | 0.6396 | 0.5125 | 0.6400 | 0.5692 | 0.7938 |
| bootstrap=False, class_weight=balanced, criterion=entropy, max_depth=15, max_features=sqrt, min_samples_leaf=5, min_samples_split=20, n_estimators=200 | 0.6387 | 0.7473 | 0.5198 | 0.6131 | 0.7493 |
| bootstrap=False, class_weight=None, criterion=gini, max_depth=None, max_features=sqrt, min_samples_leaf=5, min_samples_split=2, n_estimators=400 | 0.6361 | 0.5053 | 0.6396 | 0.5646 | 0.7928 |
| bootstrap=False, class_weight=balanced, criterion=gini, max_depth=20, max_features=sqrt, min_samples_leaf=5, min_samples_split=10, n_estimators=300 | 0.6359 | 0.7224 | 0.5300 | 0.6114 | 0.7559 |
| bootstrap=True, class_weight=None, criterion=gini, max_depth=15, max_features=None, min_samples_leaf=10, min_samples_split=10, n_estimators=300 | 0.6336 | 0.5338 | 0.6637 | 0.5917 | 0.8042 |
| bootstrap=True, class_weight=balanced, criterion=gini, max_depth=10, max_features=None, min_samples_leaf=10, min_samples_split=2, n_estimators=200 | 0.6331 | 0.7829 | 0.5405 | 0.6395 | 0.7654 |
| bootstrap=True, class_weight=None, criterion=entropy, max_depth=15, max_features=log2, min_samples_leaf=2, min_samples_split=5, n_estimators=100 | 0.6293 | 0.5089 | 0.6300 | 0.5630 | 0.7900 |
| bootstrap=False, class_weight=balanced, criterion=gini, max_depth=15, max_features=log2, min_samples_leaf=2, min_samples_split=5, n_estimators=400 | 0.6246 | 0.6940 | 0.5478 | 0.6122 | 0.7663 |
| bootstrap=True, class_weight=balanced, criterion=gini, max_depth=30, max_features=None, min_samples_leaf=5, min_samples_split=2, n_estimators=300 | 0.6199 | 0.7331 | 0.5296 | 0.6149 | 0.7559 |
| bootstrap=False, class_weight=balanced, criterion=gini, max_depth=30, max_features=log2, min_samples_leaf=2, min_samples_split=5, n_estimators=600 | 0.6183 | 0.6726 | 0.5608 | 0.6117 | 0.7729 |
| bootstrap=False, class_weight=None, criterion=gini, max_depth=15, max_features=sqrt, min_samples_leaf=2, min_samples_split=2, n_estimators=600 | 0.6178 | 0.5409 | 0.6414 | 0.5869 | 0.7975 |
| bootstrap=False, class_weight=None, criterion=entropy, max_depth=30, max_features=log2, min_samples_leaf=2, min_samples_split=2, n_estimators=200 | 0.6155 | 0.5302 | 0.6422 | 0.5809 | 0.7966 |
| bootstrap=False, class_weight=balanced, criterion=entropy, max_depth=5, max_features=None, min_samples_leaf=10, min_samples_split=2, n_estimators=300 | 0.6110 | 0.7722 | 0.5082 | 0.6130 | 0.7408 |
| bootstrap=True, class_weight=balanced, criterion=gini, max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=10, n_estimators=300 | 0.6097 | 0.6904 | 0.5257 | 0.5969 | 0.7521 |
| bootstrap=False, class_weight=None, criterion=entropy, max_depth=30, max_features=log2, min_samples_leaf=1, min_samples_split=5, n_estimators=200 | 0.6057 | 0.5231 | 0.6309 | 0.5720 | 0.7919 |
| bootstrap=False, class_weight=balanced, criterion=gini, max_depth=15, max_features=None, min_samples_leaf=10, min_samples_split=2, n_estimators=400 | 0.5718 | 0.7331 | 0.4976 | 0.5928 | 0.7323 |
| bootstrap=False, class_weight=None, criterion=gini, max_depth=20, max_features=None, min_samples_leaf=10, min_samples_split=5, n_estimators=100 | 0.5700 | 0.5480 | 0.5347 | 0.5413 | 0.7531 |
| bootstrap=False, class_weight=None, criterion=entropy, max_depth=10, max_features=None, min_samples_leaf=5, min_samples_split=2, n_estimators=600 | 0.5399 | 0.5765 | 0.5436 | 0.5596 | 0.7588 |
| bootstrap=False, class_weight=None, criterion=entropy, max_depth=None, max_features=None, min_samples_leaf=5, min_samples_split=10, n_estimators=50 | 0.5198 | 0.5516 | 0.5099 | 0.5299 | 0.7398 |
| bootstrap=False, class_weight=balanced, criterion=gini, max_depth=20, max_features=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200 | 0.4826 | 0.6797 | 0.5026 | 0.5779 | 0.7360 |

## xgboost

Default: AUC-PR 0.6234, recall 0.7260.

### Univariado (un parámetro a la vez)

- `n_estimators`:       50 -> APR 0.6481 / rec 0.7794;      100 -> APR 0.6446 / rec 0.7758;      200 -> APR 0.6319 / rec 0.7438;      300 -> APR 0.6234 / rec 0.7260;      400 -> APR 0.6182 / rec 0.7189;      600 -> APR 0.6071 / rec 0.6940
  Mejor: 50 (APR 0.6481, rec 0.7794).
- `max_depth`:        2 -> APR 0.6481 / rec 0.8078;        3 -> APR 0.6540 / rec 0.7936;        4 -> APR 0.6433 / rec 0.7616;        6 -> APR 0.6179 / rec 0.7153;        8 -> APR 0.5990 / rec 0.6512;       10 -> APR 0.5904 / rec 0.6014
  Mejor: 3 (APR 0.6540, rec 0.7936).
- `learning_rate`:     0.01 -> APR 0.6529 / rec 0.7865;     0.05 -> APR 0.6415 / rec 0.7580;    0.075 -> APR 0.6234 / rec 0.7260;      0.1 -> APR 0.6261 / rec 0.7367;      0.2 -> APR 0.5964 / rec 0.6762
  Mejor: 0.01 (APR 0.6529, rec 0.7865).
- `subsample`:      0.7 -> APR 0.6257 / rec 0.7367;      0.8 -> APR 0.6266 / rec 0.7260;      0.9 -> APR 0.6268 / rec 0.7224;      1.0 -> APR 0.6234 / rec 0.7260
  Mejor: 0.9 (APR 0.6268, rec 0.7224).
- `colsample_bytree`:      0.7 -> APR 0.6239 / rec 0.7402;      0.8 -> APR 0.6307 / rec 0.7331;      0.9 -> APR 0.6234 / rec 0.7295;      1.0 -> APR 0.6234 / rec 0.7260
  Mejor: 0.8 (APR 0.6307, rec 0.7331).
- `min_child_weight`:        1 -> APR 0.6234 / rec 0.7260;        3 -> APR 0.6318 / rec 0.7438;        5 -> APR 0.6315 / rec 0.7367;        7 -> APR 0.6349 / rec 0.7402
  Mejor: 7 (APR 0.6349, rec 0.7402).
- `gamma`:      0.0 -> APR 0.6234 / rec 0.7260;      0.1 -> APR 0.6247 / rec 0.7367;      0.3 -> APR 0.6373 / rec 0.7509;      0.5 -> APR 0.6438 / rec 0.7687
  Mejor: 0.5 (APR 0.6438, rec 0.7687).
- `reg_lambda`:      0.1 -> APR 0.6407 / rec 0.7260;        1 -> APR 0.6234 / rec 0.7260;       10 -> APR 0.6359 / rec 0.7509
  Mejor: 0.1 (APR 0.6407, rec 0.7260).
- `scale_pos_weight`:      1.0 -> APR 0.6313 / rec 0.5089;      1.5 -> APR 0.6322 / rec 0.6121;      2.0 -> APR 0.6268 / rec 0.6762;      2.5 -> APR 0.6336 / rec 0.7189;     None -> APR 0.6234 / rec 0.7260
  Mejor: 2.5 (APR 0.6336, rec 0.7189).

### Búsqueda aleatoria conjunta

| Parámetros | AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |
|---|---|---|---|---|---|
| colsample_bytree=0.9, gamma=0.1, learning_rate=0.01, max_depth=2, min_child_weight=5, n_estimators=300, reg_lambda=10, scale_pos_weight=2.0, subsample=0.9 | 0.6629 | 0.7580 | 0.5406 | 0.6311 | 0.7644 |
| colsample_bytree=1.0, gamma=0.0, learning_rate=0.01, max_depth=3, min_child_weight=7, n_estimators=400, reg_lambda=0.1, scale_pos_weight=None, subsample=1.0 | 0.6615 | 0.8221 | 0.5066 | 0.6269 | 0.7398 |
| colsample_bytree=0.7, gamma=0.0, learning_rate=0.01, max_depth=2, min_child_weight=7, n_estimators=400, reg_lambda=10, scale_pos_weight=2.0, subsample=0.9 | 0.6610 | 0.7438 | 0.5457 | 0.6295 | 0.7673 |
| colsample_bytree=0.7, gamma=0.3, learning_rate=0.1, max_depth=2, min_child_weight=5, n_estimators=600, reg_lambda=1, scale_pos_weight=2.0, subsample=0.7 | 0.6600 | 0.7153 | 0.5447 | 0.6185 | 0.7654 |
| colsample_bytree=0.9, gamma=0.3, learning_rate=0.1, max_depth=2, min_child_weight=3, n_estimators=50, reg_lambda=1, scale_pos_weight=None, subsample=0.9 | 0.6600 | 0.8149 | 0.4989 | 0.6189 | 0.7332 |
| colsample_bytree=0.7, gamma=0.5, learning_rate=0.05, max_depth=2, min_child_weight=5, n_estimators=50, reg_lambda=0.1, scale_pos_weight=1.0, subsample=1.0 | 0.6568 | 0.4306 | 0.6954 | 0.5319 | 0.7985 |
| colsample_bytree=0.7, gamma=0.0, learning_rate=0.075, max_depth=2, min_child_weight=5, n_estimators=200, reg_lambda=1, scale_pos_weight=1.5, subsample=0.9 | 0.6565 | 0.6299 | 0.5803 | 0.6041 | 0.7805 |
| colsample_bytree=0.9, gamma=0.5, learning_rate=0.1, max_depth=3, min_child_weight=5, n_estimators=100, reg_lambda=10, scale_pos_weight=1.0, subsample=0.7 | 0.6556 | 0.5160 | 0.6532 | 0.5765 | 0.7985 |
| colsample_bytree=0.9, gamma=0.5, learning_rate=0.01, max_depth=2, min_child_weight=7, n_estimators=200, reg_lambda=10, scale_pos_weight=2.5, subsample=0.9 | 0.6543 | 0.7829 | 0.5116 | 0.6188 | 0.7436 |
| colsample_bytree=0.7, gamma=0.1, learning_rate=0.01, max_depth=6, min_child_weight=7, n_estimators=300, reg_lambda=0.1, scale_pos_weight=None, subsample=0.8 | 0.6515 | 0.7865 | 0.5225 | 0.6278 | 0.7521 |
| colsample_bytree=0.8, gamma=0.1, learning_rate=0.1, max_depth=2, min_child_weight=1, n_estimators=600, reg_lambda=1, scale_pos_weight=1.5, subsample=0.7 | 0.6514 | 0.6121 | 0.5677 | 0.5890 | 0.7729 |
| colsample_bytree=0.9, gamma=0.5, learning_rate=0.075, max_depth=3, min_child_weight=7, n_estimators=200, reg_lambda=10, scale_pos_weight=2.5, subsample=0.9 | 0.6507 | 0.7580 | 0.5108 | 0.6103 | 0.7427 |
| colsample_bytree=0.7, gamma=0.1, learning_rate=0.01, max_depth=4, min_child_weight=1, n_estimators=600, reg_lambda=1, scale_pos_weight=1.5, subsample=0.9 | 0.6492 | 0.6157 | 0.5864 | 0.6007 | 0.7824 |
| colsample_bytree=0.8, gamma=0.0, learning_rate=0.01, max_depth=6, min_child_weight=1, n_estimators=100, reg_lambda=0.1, scale_pos_weight=2.0, subsample=0.9 | 0.6491 | 0.6762 | 0.5621 | 0.6139 | 0.7739 |
| colsample_bytree=1.0, gamma=0.3, learning_rate=0.2, max_depth=2, min_child_weight=1, n_estimators=200, reg_lambda=0.1, scale_pos_weight=1.5, subsample=0.7 | 0.6486 | 0.6157 | 0.5825 | 0.5986 | 0.7805 |
| colsample_bytree=0.7, gamma=0.5, learning_rate=0.1, max_depth=2, min_child_weight=7, n_estimators=600, reg_lambda=1, scale_pos_weight=2.5, subsample=0.9 | 0.6485 | 0.7544 | 0.5183 | 0.6145 | 0.7483 |
| colsample_bytree=0.7, gamma=0.3, learning_rate=0.01, max_depth=6, min_child_weight=5, n_estimators=300, reg_lambda=0.1, scale_pos_weight=1.0, subsample=0.9 | 0.6475 | 0.4911 | 0.6603 | 0.5633 | 0.7975 |
| colsample_bytree=1.0, gamma=0.1, learning_rate=0.1, max_depth=2, min_child_weight=7, n_estimators=600, reg_lambda=1, scale_pos_weight=1.5, subsample=0.8 | 0.6463 | 0.6228 | 0.5833 | 0.6024 | 0.7815 |
| colsample_bytree=1.0, gamma=0.3, learning_rate=0.075, max_depth=6, min_child_weight=1, n_estimators=50, reg_lambda=10, scale_pos_weight=2.5, subsample=0.8 | 0.6460 | 0.7580 | 0.5448 | 0.6339 | 0.7673 |
| colsample_bytree=0.8, gamma=0.1, learning_rate=0.01, max_depth=6, min_child_weight=3, n_estimators=100, reg_lambda=0.1, scale_pos_weight=1.5, subsample=1.0 | 0.6458 | 0.5409 | 0.6360 | 0.5846 | 0.7956 |
| colsample_bytree=0.8, gamma=0.1, learning_rate=0.2, max_depth=2, min_child_weight=3, n_estimators=200, reg_lambda=10, scale_pos_weight=1.0, subsample=1.0 | 0.6453 | 0.5125 | 0.6372 | 0.5680 | 0.7928 |
| colsample_bytree=0.9, gamma=0.3, learning_rate=0.1, max_depth=3, min_child_weight=7, n_estimators=400, reg_lambda=0.1, scale_pos_weight=2.0, subsample=1.0 | 0.6433 | 0.7153 | 0.5537 | 0.6242 | 0.7711 |
| colsample_bytree=0.7, gamma=0.1, learning_rate=0.2, max_depth=3, min_child_weight=1, n_estimators=100, reg_lambda=1, scale_pos_weight=2.0, subsample=0.8 | 0.6423 | 0.6940 | 0.5357 | 0.6047 | 0.7588 |
| colsample_bytree=0.9, gamma=0.3, learning_rate=0.05, max_depth=8, min_child_weight=3, n_estimators=50, reg_lambda=10, scale_pos_weight=1.0, subsample=0.9 | 0.6391 | 0.4911 | 0.6571 | 0.5621 | 0.7966 |
| colsample_bytree=1.0, gamma=0.3, learning_rate=0.2, max_depth=4, min_child_weight=7, n_estimators=200, reg_lambda=1, scale_pos_weight=None, subsample=1.0 | 0.6386 | 0.7616 | 0.5169 | 0.6158 | 0.7474 |
| colsample_bytree=1.0, gamma=0.0, learning_rate=0.05, max_depth=6, min_child_weight=1, n_estimators=100, reg_lambda=1, scale_pos_weight=None, subsample=1.0 | 0.6384 | 0.7722 | 0.5229 | 0.6236 | 0.7521 |
| colsample_bytree=0.7, gamma=0.0, learning_rate=0.075, max_depth=6, min_child_weight=5, n_estimators=200, reg_lambda=10, scale_pos_weight=2.0, subsample=0.9 | 0.6342 | 0.6868 | 0.5530 | 0.6127 | 0.7692 |
| colsample_bytree=0.8, gamma=0.3, learning_rate=0.05, max_depth=6, min_child_weight=7, n_estimators=400, reg_lambda=10, scale_pos_weight=2.0, subsample=1.0 | 0.6337 | 0.6975 | 0.5414 | 0.6096 | 0.7625 |
| colsample_bytree=0.9, gamma=0.5, learning_rate=0.05, max_depth=10, min_child_weight=7, n_estimators=100, reg_lambda=1, scale_pos_weight=1.0, subsample=0.9 | 0.6321 | 0.5160 | 0.6444 | 0.5731 | 0.7956 |
| colsample_bytree=1.0, gamma=0.5, learning_rate=0.2, max_depth=10, min_child_weight=7, n_estimators=50, reg_lambda=10, scale_pos_weight=2.0, subsample=1.0 | 0.6312 | 0.6690 | 0.5579 | 0.6084 | 0.7711 |
| colsample_bytree=0.9, gamma=0.3, learning_rate=0.2, max_depth=6, min_child_weight=5, n_estimators=400, reg_lambda=1, scale_pos_weight=2.5, subsample=1.0 | 0.6268 | 0.7153 | 0.5262 | 0.6063 | 0.7531 |
| colsample_bytree=0.7, gamma=0.3, learning_rate=0.075, max_depth=8, min_child_weight=1, n_estimators=300, reg_lambda=1, scale_pos_weight=2.0, subsample=1.0 | 0.6257 | 0.6548 | 0.5542 | 0.6003 | 0.7682 |
| colsample_bytree=0.7, gamma=0.3, learning_rate=0.2, max_depth=6, min_child_weight=1, n_estimators=100, reg_lambda=1, scale_pos_weight=1.5, subsample=1.0 | 0.6214 | 0.6157 | 0.5710 | 0.5925 | 0.7748 |
| colsample_bytree=0.9, gamma=0.5, learning_rate=0.2, max_depth=10, min_child_weight=3, n_estimators=50, reg_lambda=10, scale_pos_weight=2.0, subsample=0.7 | 0.6117 | 0.6584 | 0.5781 | 0.6156 | 0.7815 |
| colsample_bytree=0.7, gamma=0.5, learning_rate=0.075, max_depth=10, min_child_weight=1, n_estimators=100, reg_lambda=0.1, scale_pos_weight=2.5, subsample=1.0 | 0.6097 | 0.6299 | 0.5463 | 0.5851 | 0.7625 |
| colsample_bytree=0.8, gamma=0.1, learning_rate=0.075, max_depth=10, min_child_weight=1, n_estimators=200, reg_lambda=0.1, scale_pos_weight=2.5, subsample=0.9 | 0.6059 | 0.6228 | 0.5912 | 0.6066 | 0.7852 |
| colsample_bytree=1.0, gamma=0.0, learning_rate=0.01, max_depth=6, min_child_weight=3, n_estimators=50, reg_lambda=10, scale_pos_weight=None, subsample=1.0 | 0.6011 | 0.7900 | 0.5034 | 0.6150 | 0.7370 |
| colsample_bytree=0.7, gamma=0.3, learning_rate=0.1, max_depth=8, min_child_weight=5, n_estimators=300, reg_lambda=0.1, scale_pos_weight=2.0, subsample=0.8 | 0.5967 | 0.6299 | 0.5747 | 0.6010 | 0.7777 |
| colsample_bytree=0.8, gamma=0.0, learning_rate=0.1, max_depth=6, min_child_weight=7, n_estimators=600, reg_lambda=0.1, scale_pos_weight=1.5, subsample=0.7 | 0.5918 | 0.5801 | 0.5621 | 0.5709 | 0.7682 |
| colsample_bytree=0.8, gamma=0.5, learning_rate=0.2, max_depth=10, min_child_weight=3, n_estimators=300, reg_lambda=1, scale_pos_weight=1.5, subsample=0.9 | 0.5858 | 0.5730 | 0.5750 | 0.5740 | 0.7739 |

## Resumen

| Familia | Default APR/rec | Mejor univariado | Mejor aleatorio |
|---|---|---|---|
| logistic-regression | 0.6686/0.8043 | 0.6817/0.7794 | 0.6838/0.8007 |
| random-forest | 0.6362/0.7687 | 0.6605/0.8043 | 0.6594/0.8007 |
| xgboost | 0.6234/0.7260 | 0.6540/0.7936 | 0.6629/0.7580 |

Mejor configuración global: **logistic-regression** [random] C=0.01, class_weight=balanced, l1_ratio=1.0, solver=saga -> AUC-PR 0.6838, recall 0.8007. Referencia de producción: logistic-regression (default) AUC-PR 0.6686, recall 0.8043 (selección T-14).

Detalle completo en `reports/hyperparams_exploration.json`. La adopción de una configuración ganadora implicaría proponer la actualización de `docs/tasks.md` y reejecutar `T-17`/`T-14`.
