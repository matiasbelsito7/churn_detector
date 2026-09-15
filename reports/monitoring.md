# Monitoreo del sistema de predicción de churn (T-22)

Estado general: **OK**.

Umbrales y acciones: `docs/monitoring.md`. Chequeos por dominio:

### Drift de features (referencia = `train`, corriente = batch)

| Chequeo | Métrica | Valor | Estado |
|---|---|---|---|
| tenure | psi | 0.0006 | ok |
| MonthlyCharges | psi | 0.0013 | ok |
| TotalCharges | psi | 0.0004 | ok |
| addon_missing_count | psi | 0.0001 | ok |
| avg_monthly_charge_hist | psi | 0.0008 | ok |
| gender | categorical_max_shift | 0.0016 | ok |
| SeniorCitizen | categorical_max_shift | 0.0016 | ok |
| Partner | categorical_max_shift | 0.0006 | ok |
| Dependents | categorical_max_shift | 0.0054 | ok |
| PhoneService | categorical_max_shift | 0.0048 | ok |
| MultipleLines | categorical_max_shift | 0.0076 | ok |
| InternetService | categorical_max_shift | 0.0048 | ok |
| OnlineSecurity | categorical_max_shift | 0.0035 | ok |
| OnlineBackup | categorical_max_shift | 0.0041 | ok |
| DeviceProtection | categorical_max_shift | 0.0012 | ok |
| TechSupport | categorical_max_shift | 0.0038 | ok |
| StreamingTV | categorical_max_shift | 0.0044 | ok |
| StreamingMovies | categorical_max_shift | 0.0049 | ok |
| Contract | categorical_max_shift | 0.0038 | ok |
| PaperlessBilling | categorical_max_shift | 0.0034 | ok |
| PaymentMethod | categorical_max_shift | 0.0053 | ok |
| tenure_bin | categorical_max_shift | 0.0033 | ok |
| is_new_client | categorical_max_shift | 0.0017 | ok |
| is_month_to_month | categorical_max_shift | 0.0008 | ok |
| is_electronic_check | categorical_max_shift | 0.0010 | ok |
| is_fiber_optic | categorical_max_shift | 0.0048 | ok |

### Distribución de predicciones

| Chequeo | Métrica | Valor | Estado |
|---|---|---|---|
| churn_rate | abs_delta | 0.0000 | ok |
| churn_prob | psi | 0.0000 | ok |

### Drift del target

| Chequeo | Métrica | Valor | Estado |
|---|---|---|---|
| target_prevalence | abs_delta | 0.0000 | ok |

### Desempeño (target disponible)

| Chequeo | Métrica | Valor | Estado |
|---|---|---|---|
| recall_pos | drop_vs_reference | -0.0018 | ok |
| auc_pr | drop_vs_reference | 0.0015 | ok |

### Trazabilidad

| Artefacto | Origen | Filas | SHA256 |
|---|---|---|---|
| features_referencia | data\features\churn_features.csv | 4929 | `ec2921872616…` |
| features_corriente | data\features\churn_features.csv | 7043 | `ec2921872616…` |
| predicciones_referencia | data\predictions\predictions.csv | 7043 | `d10be60bb80b…` |
| predicciones_corriente | data\predictions\predictions.csv | 7043 | `d10be60bb80b…` |
