# Plan de monitoring (T-22)

## 1. Propósito y alcance

Define qué se observa del sistema de predicción de churn en operación, qué
umbrales disparan una alerta y qué acciones se toman ante cada alerta.
Sigue `docs/specs.md` §9.5: observar la **distribución de entradas** (drift de
features) y de **predicciones**, y el **desempeño** cuando se dispone del
target.

El monitoreo opera sobre artefactos versionados y reproducibles, sin depender
de ejecución manual:

- **Referencia (baseline)**: distribución de features de la población de
  entrenamiento, distribución de predicciones del batch de despliegue
  (`T-15`) y desempeño del modelo seleccionado sobre `test` (`T-14`,
  `reports/selection.json`).
- **Corriente**: nuevo batch de features y predicciones que llega al sistema
  (por defecto, los mismos artefactos versionados; en producción, el batch
  acumulado del periodo).

## 2. Qué se monitorea

### 2.1 Drift de features

Se compara la distribución de cada feature del modelo (`FEATURE_NAMES`, ver
`src/modeling/preprocessing.py`) entre la población de entrenamiento
(referencia) y el batch corriente.

- **Numéricas** (`tenure`, `MonthlyCharges`, `TotalCharges`,
  `addon_missing_count`, `avg_monthly_charge_hist`): **PSI** (Population
  Stability Index) sobre la distribución de valores.
- **Categóricas**: desviación máxima de proporciones por categoría.

### 2.2 Distribución de predicciones

Se compara la distribución de la **probabilidad de churn** y la **tasa de
churn predicha** del batch de referencia con el batch corriente.

- Tasa predicha: diferencia de la proporción de `churn_class == "Yes"`.
- Probabilidad: PSI sobre la distribución de `churn_prob`.

### 2.3 Drift del target (cuando está disponible)

Si el batch corriente incluye el target real, se compara su prevalencia
(`Churn == "Yes"`) con la de la referencia. Complementa el drift de features.

### 2.4 Desempeño (cuando se dispone del target)

Si el batch corriente dispone del target real, se calculan las métricas del
problema desbalanceado sobre la clase `Yes` (`AUC-PR`, `recall`, `precision`,
`F1`) usando `src.modeling.baseline.evaluate_metrics` y se comparan con las
de la referencia (`reports/selection.json`, evaluación sobre `test`).

## 3. Umbrales de alerta

Cada check devuelve un estado `ok` / `warn` / `alert`. El estado general del
reporte es el peor de todos los checks.

| Check | `ok` | `warn` | `alert` |
|---|---|---|---|
| PSI (feature o probabilidad) | `< 0.10` | `0.10–0.25` | `> 0.25` |
| Shift categórico (Δ proporción máx.) | `< 0.05` | `0.05–0.10` | `> 0.10` |
| Tasa de churn predicha (Δ absoluto) | `< 0.03` | `0.03–0.08` | `> 0.08` |
| Prevalencia del target (Δ absoluto) | `< 0.03` | `0.03–0.08` | `> 0.08` |
| Caída de `recall_pos` vs referencia | `< 0.05` | `0.05–0.10` | `> 0.10` |
| Caída de `auc_pr` vs referencia | `< 0.05` | `0.05–0.10` | `> 0.10` |

Interpretación del PSI (convención estándar): `ok` sin señales; `warn` drift
moderado a vigilar; `alert` cambio significativo de población.

## 4. Acciones ante alertas

| Estado | Acción |
|---|---|
| `ok` | Sin acción. Se continúa con el siguiente periodo. |
| `warn` | Revisar el check afectado; si persiste dos periodos, escalar a alerta. Registrar observación en el reporte. |
| `alert` | **Detener decisiones automáticas** sobre la cohorte afectada. Investigar la causa (cambio de negocio, error de ingestión, cambio de esquema). Documentar y, si el drift persiste, **re-adquirir datos y re-entrenar** el modelo (pipeline `T-15`) y re-evaluar con `T-14` antes de volver a servir. |

Regla general: **nunca se re-entrena de forma automática**; toda acción ante
alerta implica revisión humana y evidencia documentada.

## 5. Implementación

Módulo: `src/serving/monitor.py`. Produce `reports/monitoring.md` y
`reports/monitoring.json`.

```console
python -m src.serving.monitor
```

Configuración por variables de entorno (por defecto usa los artefactos
versionados del repo):

| Variable | Defecto | Uso |
|---|---|---|
| `MONITOR_REFERENCE_FEATURES` | filas de `data/features/churn_features.csv` en `data/splits/train.csv` | features de referencia |
| `MONITOR_CURRENT_FEATURES` | `data/features/churn_features.csv` | features del batch corriente |
| `MONITOR_REFERENCE_PREDICTIONS` | `data/predictions/predictions.csv` | predicciones de referencia |
| `MONITOR_CURRENT_PREDICTIONS` | `data/predictions/predictions.csv` | predicciones del batch corriente |
| `MONITOR_REFERENCE_METRICS` | `reports/selection.json` (`test_metrics`) | desempeño de referencia |

Trazabilidad: el reporte registra rutas, filas y checksums de los artefactos
usados, de modo que el resultado del monitoreo es reproducible y auditable.

## 6. Verificación

- `tests/test_monitor.py`: cubre PSI, shift categórico, clasificación de
  umbrales y el reporte, con poblaciones sintéticas desplazadas para
  ejercitar `ok`/`warn`/`alert` sin depender de la base.
- La ejecución sobre los artefactos versionados produce estados `ok` (el batch
  corriente coincide con la referencia); el mecanismo de alerta se verifica
  con los tests sintéticos.
