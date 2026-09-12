# Audit del dataset crudo (T-05)

Informe generado automáticamente desde el raw versionado.

- **Archivo:** `data\raw\Telco-Customer-Churn.csv`
- **Dimensiones:** 7043 filas × 21 columnas
- **Filas duplicadas (completas):** 0
- **Identificadores repetidos (`customerID`):** 0

## 1. Esquema: tipos reales vs esperados

| Campo | Tipo esperado | Tipo real (raw) | Desviación |
|---|---|---|---|
| customerID | identificador | `object` | No |
| gender | categórica | `object` | No |
| SeniorCitizen | entera | `int64` | No |
| Partner | categórica | `object` | No |
| Dependents | categórica | `object` | No |
| tenure | entera | `int64` | No |
| PhoneService | categórica | `object` | No |
| MultipleLines | categórica | `object` | No |
| InternetService | categórica | `object` | No |
| OnlineSecurity | categórica | `object` | No |
| OnlineBackup | categórica | `object` | No |
| DeviceProtection | categórica | `object` | No |
| TechSupport | categórica | `object` | No |
| StreamingTV | categórica | `object` | No |
| StreamingMovies | categórica | `object` | No |
| Contract | categórica | `object` | No |
| PaperlessBilling | categórica | `object` | No |
| PaymentMethod | categórica | `object` | No |
| MonthlyCharges | flotante | `float64` | No |
| TotalCharges | flotante | `object` | **SÍ — 11 valores no parseables como flotante (antes: object)** (nulos esperados si `tenure == 0`) |
| Churn | target (categórica binaria) | `object` | No |

## 2. Valores nulos y ausencias

| Campo | Nulos |
|---|---|
| customerID | 0 |
| gender | 0 |
| SeniorCitizen | 0 |
| Partner | 0 |
| Dependents | 0 |
| tenure | 0 |
| PhoneService | 0 |
| MultipleLines | 0 |
| InternetService | 0 |
| OnlineSecurity | 0 |
| OnlineBackup | 0 |
| DeviceProtection | 0 |
| TechSupport | 0 |
| StreamingTV | 0 |
| StreamingMovies | 0 |
| Contract | 0 |
| PaperlessBilling | 0 |
| PaymentMethod | 0 |
| MonthlyCharges | 0 |
| TotalCharges | 11 |
| Churn | 0 |

## 3. Dominios y rangos

| Campo | Dominio/rango esperado | Valores observados | Cumple |
|---|---|---|---|
| customerID | único (patrón `^\d{4}-[A-Z]{5}$`) | 7043 valores únicos | Sí |
| gender | Female, Male | Female, Male | Sí |
| SeniorCitizen | [0, 1] | [0, 1] | Sí |
| Partner | No, Yes | No, Yes | Sí |
| Dependents | No, Yes | No, Yes | Sí |
| tenure | [0, 72] | [0, 72] | Sí |
| PhoneService | No, Yes | No, Yes | Sí |
| MultipleLines | No, No phone service, Yes | No, No phone service, Yes | Sí |
| InternetService | DSL, Fiber optic, No | DSL, Fiber optic, No | Sí |
| OnlineSecurity | No, No internet service, Yes | No, No internet service, Yes | Sí |
| OnlineBackup | No, No internet service, Yes | No, No internet service, Yes | Sí |
| DeviceProtection | No, No internet service, Yes | No, No internet service, Yes | Sí |
| TechSupport | No, No internet service, Yes | No, No internet service, Yes | Sí |
| StreamingTV | No, No internet service, Yes | No, No internet service, Yes | Sí |
| StreamingMovies | No, No internet service, Yes | No, No internet service, Yes | Sí |
| Contract | Month-to-month, One year, Two year | Month-to-month, One year, Two year | Sí |
| PaperlessBilling | No, Yes | No, Yes | Sí |
| PaymentMethod | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check | Sí |
| MonthlyCharges | [18.25, 118.75] | [18.25, 118.75] | Sí |
| TotalCharges | >= 0.0 | [18.8, 8684.8] | Sí |
| Churn | No, Yes | No, Yes | Sí |

## 4. Coherencia entre campos

- Sin incoherencias entre campos detectadas.

## 5. Resumen de hallazgos

- `TotalCharges` se almacena como texto con 11 celdas no numéricas (solo espacios), todas con `tenure == 0` (ausencia estructural, no aleatoria). Debe convertirse a flotante en `T-07`.
- `SeniorCitizen` usa codificación numérica `0/1` mientras el resto de los binarios usa `No/Yes`; normalizar en `T-07` para un esquema consistente.

## 6. Desbalanceo del target

- `No`: 5174 (73.46%)
- `Yes`: 1869 (26.54%)
- Clase de interés (positiva): `Yes` (1869).