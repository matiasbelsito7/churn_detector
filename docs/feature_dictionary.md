# Diccionario de features (T-10)

## Propósito

Documenta las features derivadas del EDA (`T-09`) e implementadas en
`src/analysis/features.py`. Cada feature nueva se introduce **únicamente** si
existe un hallazgo del EDA que la justifique (sección 5 de `reports/eda.md`,
hallazgos `F1`–`F9`).

Artefacto generado: `data/features/churn_features.csv` (trazabilidad del
proceso en `data/features/feature_log.json`), a partir del dataset procesado de
`T-08`.

## Consistencia entrenamiento / inferencia

`engineer_features(df)` es una transformación **pura y determinista**:
- No utiliza estado externo, semillas ni información del target de otras filas.
- Aplica la misma definición y agrega las mismas columnas **en el mismo orden**
  para cualquier entrada que respete el esquema procesado (`data_contract`).
- Es seguro aplicarla de forma idéntica sobre train, validación, prueba e
  inferencia: la misma función que genera el artefacto se usa en inferencia.

## Features derivadas

| Feature | Origen (hallazgo EDA) | Evidencia | Definición | Tipo / dominio |
|---|---|---|---|---|
| `tenure_bin` | **F2** – tenure central, churn decrece con antigüedad | Churn 56.21 % (tenure ≤ 3) vs 4.73 %–6.68 % (tenure ≥ 60) | Tramos ordinales de antigüedad alineados con la figura "Tasa de churn por antigüedad" | categoría ordinal: `0-5`, `6-11`, `12-23`, `24-35`, `36-47`, `48-59`, `60-72` |
| `is_new_client` | **F2** – segmento de cliente nuevo de alto riesgo | Churn 56.21 % en tenure ≤ 3 meses | `Yes` si `tenure <= 3`, `No` en caso contrario | binario `No`/`Yes` |
| `is_month_to_month` | **F3** – contrato es feature fuerte | Month-to-month 42.71 %, One year 11.27 %, Two year 2.83 % | `Yes` si `Contract == "Month-to-month"` | binario `No`/`Yes` |
| `is_electronic_check` | **F4** – método de pago separa perfiles | Electronic check 45.29 % vs automáticos 15–17 % | `Yes` si `PaymentMethod == "Electronic check"` | binario `No`/`Yes` |
| `is_fiber_optic` | **F5** – tipo de internet discrimina | Fiber optic 41.89 %, DSL 18.96 %, No 7.40 % | `Yes` si `InternetService == "Fiber optic"` | binario `No`/`Yes` |
| `addon_missing_count` | **F6** – servicios de valor añadido ausentes ↑ churn | OnlineSecurity No: 41.77 %, TechSupport No: 41.64 % | Conteo de servicios de valor añadido ausentes (`"No"`) entre `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`. Los clientes sin internet (`"No internet service"`) suman 0 (no aplica). | entero 0–4 |
| `avg_monthly_charge_hist` | **F9** – colinealidad tenure–TotalCharges (r = 0.83) | Alta correlación entre numéricas | `TotalCharges / tenure` si `tenure > 0`; `0.0` si `tenure == 0` (clientes nuevos, sin histórico) | flotante ≥ 0 |

## Features sin transformar (usadas tal cual)

- **`tenure`** (F2): se conserva la variable continua; las features `tenure_bin`
  e `is_new_client` son versiones discretas complementarias, no sustitutas.
- **`MonthlyCharges`** (F8): correlaciona positivamente con churn, pero
  confundada con `tenure` y contrato. Se conserva y la decisión de
  estandarizar/condicionar se resuelve en el preprocessing (`T-12`).
- **`TotalCharges`**: su contraparte derivada `avg_monthly_charge_hist` mitiga
  su colinealidad con `tenure` (F9); la selección final de columnas queda para
  el pipeline de modelado tras `T-11`.
- **`SeniorCitizen`** (F7): representación binaria ya normalizada en el
  procesado (`No`/`Yes`); se usa directamente.
- **`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`,
  `StreamingTV`, `StreamingMovies`, `PaperlessBilling`**: no presentan un
  hallazgo que motive transformación derivada; se conservan para evaluación del
  pipeline (F1–F9 no las justifican como features nuevas).

## Reproducción

```console
python -m src.analysis.features
```

El comando verifica el esquema de entrada, genera el CSV de features y escribe
el log de trazabilidad (checksums de entrada/salida, semilla, features, fecha).
