# Explicabilidad del modelo (T-28)

Método: valores SHAP con `shap.LinearExplainer` sobre el pipeline del modelo seleccionado (features preprocesadas con OHE y escalado, `T-12`). Las contribuciones de las columnas codificadas se agrupan en su feature original para un reporte interpretable. La explicación es descriptiva y no reentrena ni altera la selección de `T-14`.

- Modelo explicado: `churn-logistic-regression@production` (specs.md §7).
- Partición: `data/splits/validation.csv` (1057 filas, sha256 `5011d2f9916daf6f9a106757b07e7fb71ecc8b4a00b91587d1176079a59466d7`). `test` queda reservado a la evaluación única de `T-14`.
- Fondo de referencia: primeras 500 filas de la partición (features preprocesadas).
- Base value (log-odds medio del modelo): -0.7951.

## Importancia global de features

| Feature | mean \|SHAP\| | % contribución | media (SHAP firmado) |
|---|---|---|---|
| MonthlyCharges | 0.6934 | 13.9% | -0.0308 |
| tenure | 0.4904 | 9.8% | +0.0103 |
| is_fiber_optic | 0.4376 | 8.8% | +0.0334 |
| Contract | 0.3810 | 7.6% | +0.0794 |
| is_month_to_month | 0.3583 | 7.2% | +0.0376 |
| InternetService | 0.3225 | 6.5% | +0.0210 |
| StreamingMovies | 0.2931 | 5.9% | -0.0303 |
| StreamingTV | 0.2368 | 4.7% | -0.0249 |
| MultipleLines | 0.2212 | 4.4% | -0.0234 |
| is_new_client | 0.1626 | 3.3% | +0.0092 |
| tenure_bin | 0.1623 | 3.3% | -0.0062 |
| addon_missing_count | 0.1535 | 3.1% | +0.0422 |
| DeviceProtection | 0.1480 | 3.0% | +0.0044 |
| PaperlessBilling | 0.1375 | 2.8% | -0.0055 |
| TechSupport | 0.1159 | 2.3% | +0.0261 |
| OnlineSecurity | 0.1096 | 2.2% | +0.0217 |
| OnlineBackup | 0.1042 | 2.1% | +0.0175 |
| is_electronic_check | 0.1040 | 2.1% | -0.0015 |
| TotalCharges | 0.0731 | 1.5% | +0.0007 |
| Dependents | 0.0702 | 1.4% | +0.0109 |
| PaymentMethod | 0.0690 | 1.4% | +0.0004 |
| SeniorCitizen | 0.0529 | 1.1% | +0.0107 |
| avg_monthly_charge_hist | 0.0529 | 1.1% | +0.0008 |
| PhoneService | 0.0150 | 0.3% | +0.0003 |
| Partner | 0.0144 | 0.3% | -0.0015 |
| gender | 0.0066 | 0.1% | +0.0006 |

Las features con mayor contribución absoluta son **MonthlyCharges, tenure, is_fiber_optic**, coherentes con los hallazgos del EDA de `T-09` sobre la relación entre antigüedad, contrato y modalidad de cobro con el churn. El signo de la media SHAP indica la dirección media de cada feature sobre la probabilidad de churn.

## Figuras

- Importancia (global): reports/figures/shap_importance_bar.png
- Summary plot: reports/figures/shap_summary.png
- Dependence de `MonthlyCharges`: reports/figures/shap_dependence_MonthlyCharges.png
- Dependence de `tenure`: reports/figures/shap_dependence_tenure.png
- Dependence de `is_fiber_optic`: reports/figures/shap_dependence_is_fiber_optic.png

## Trazabilidad

- Partición `validation` (T-11) con hashes en `data/splits/partition_log.json`.
- Modelo cargado desde el Model Registry con alias `production` (T-14).
- Artefactos estructurados en `reports/explainability.json`.
