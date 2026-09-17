# Adopción de configuración del modelo (T-30)

Decisión final: **DESCARTAR** (la adopción se revirtió).

Procedimiento: se entrenaron bajo condiciones idénticas (partición `train` de `T-11`, pipeline de `T-12`, semilla fija `42`) el champion actual (configuración por defecto de `T-13`/`T-14`) y el mejor candidato de `T-29`. La regla de selección de `T-14` se aplicó sobre `validation`; el candidato superó al champion y fue adoptado, pero la evaluación sobre `test` mostró que la mejora no generaliza y la adopción se revirtió.

Regla: Métrica primaria: AUC-PR sobre la clase `Yes`; desempate por recall sobre `Yes`. Métricas adecuadas al problema desbalanceado (specs.md §7).

Candidato: `logistic-regression` con {'C': 0.01, 'class_weight': 'balanced', 'l1_ratio': 1.0, 'solver': 'saga'} (`reports/hyperparams_exploration.md`).

## Comparación sobre validation

| Configuración | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Acc. (ref.) |
|---|---|---|---|---|---|---|
| Champion (default) | 0.8421 | 0.6686 | 0.8043 | 0.5056 | 0.6209 | 0.7389 |
| Candidato T-29 | 0.8445 | 0.6838 | 0.8007 | 0.5022 | 0.6173 | 0.7360 |

## Comparación sobre test (decisión final)

| Configuración | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Acc. (ref.) |
|---|---|---|---|---|---|---|
| Champion anterior (default, v3) | 0.8560 | 0.6698 | 0.8071 | 0.5033 | 0.6200 | 0.7379 |
| Candidato adoptado (v4) | 0.8386 | 0.6338 | 0.8036 | 0.4923 | 0.6106 | 0.7285 |

## Decisión

**DESCARTAR.** Sobre `test`, la configuración adoptada obtuvo una AUC-PR de 0.6338 frente a 0.6698 del champion anterior (Δ -0.0360) y un recall de 0.8036 frente a 0.8071 (Δ -0.0036). Sin mejora material, se mantiene la configuración por defecto.

## Acciones de reversión

- Alias `production` devuelto a `churn-logistic-regression` v3.
- Versión descartada `churn-logistic-regression` v4 retirada del registro (Archived).
- `reports/selection.md/.json` restaurados al champion de `T-14`.
- Artefacto standalone y predicciones regenerados con la configuración por defecto.

Auditabilidad: corridas en el experimento `churn-detector` (configuración, métricas y artefactos); registro en `reports/adoption.json`. La partición `test` ya fue evaluada contra la configuración descartada; el champion anterior conserva su evaluación original.
