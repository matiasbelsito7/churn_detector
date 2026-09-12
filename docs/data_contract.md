# Contrato y diccionario de datos

Este documento define el **contrato de datos** y el **diccionario de datos** del
dataset IBM Telco Customer Churn. Es la referencia para las validaciones de
calidad (`T-06`), las transformaciones de limpieza y el resto de las capas del
sistema, conforme a `docs/specs.md` (sección 5.3).

- Fuente y procedencia del archivo crudo: `data/raw/PROVENANCE.yaml`.
- Artefacto crudo: `data/raw/Telco-Customer-Churn.csv` (7043 filas, 21 columnas).
- Las definiciones de tipos corresponden a los **tipos esperados**; las
  desviaciones observadas en el raw se documentan en el informe de audit
  (`reports/audit_raw.md`, tarea `T-05`).

## 1. Definición del target

- **Campo target:** `Churn`.
- **Tipo:** binario (`No` / `Yes`).
- **Clase de interés (positiva):** `Churn == "Yes"` — el cliente abandonó el
  servicio.
- **Clase negativa:** `Churn == "No"` — el cliente permanece.
- **Problema:** clasificación binaria **desbalanceada**. La clase positiva es
  minoritaria, por lo que la evaluación prioriza métricas sobre la clase de
  interés (recall, precisión, F1, AUC-PR) y no la accuracy global.

## 2. Esquema esperado

Tipos lógicos: `string` (categórica o identificador), `int`, `float`.
`nullable` indica si se admite ausencia de valor en el raw.

| # | Campo | Tipo | Dominio / rango | Nullable | Descripción |
|---|---|---|---|---|---|
| 1 | `customerID` | string (id) | patrón `^\d{4}-[A-Z]{5}$`, único | no | Identificador del cliente. No es feature. |
| 2 | `gender` | string (cat) | `{Female, Male}` | no | Género del cliente. |
| 3 | `SeniorCitizen` | int (binario) | `{0, 1}` | no | 1 si el cliente es adulto mayor; 0 en caso contrario. |
| 4 | `Partner` | string (cat) | `{No, Yes}` | no | Tiene pareja. |
| 5 | `Dependents` | string (cat) | `{No, Yes}` | no | Tiene personas a cargo. |
| 6 | `tenure` | int | `[0, 72]` | no | Meses de antigüedad como cliente. |
| 7 | `PhoneService` | string (cat) | `{No, Yes}` | no | Tiene servicio telefónico. |
| 8 | `MultipleLines` | string (cat) | `{No, No phone service, Yes}` | no | Tiene múltiples líneas. |
| 9 | `InternetService` | string (cat) | `{DSL, Fiber optic, No}` | no | Tipo de servicio de internet. |
| 10 | `OnlineSecurity` | string (cat) | `{No, No internet service, Yes}` | no | Contrata seguridad online. |
| 11 | `OnlineBackup` | string (cat) | `{No, No internet service, Yes}` | no | Contrata respaldo online. |
| 12 | `DeviceProtection` | string (cat) | `{No, No internet service, Yes}` | no | Contrata protección de dispositivo. |
| 13 | `TechSupport` | string (cat) | `{No, No internet service, Yes}` | no | Contrata soporte técnico. |
| 14 | `StreamingTV` | string (cat) | `{No, No internet service, Yes}` | no | Contrata streaming de TV. |
| 15 | `StreamingMovies` | string (cat) | `{No, No internet service, Yes}` | no | Contrata streaming de películas. |
| 16 | `Contract` | string (cat) | `{Month-to-month, One year, Two year}` | no | Tipo de contrato. |
| 17 | `PaperlessBilling` | string (cat) | `{No, Yes}` | no | Facturación sin papel. |
| 18 | `PaymentMethod` | string (cat) | `{Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check}` | no | Método de pago. |
| 19 | `MonthlyCharges` | float | `[18.25, 118.75]` | no | Cargo mensual (misma moneda, sin unidad explícita). |
| 20 | `TotalCharges` | float | `>= 0` (tope no acotado) | **sí** | Cargo total acumulado. Ausente cuando `tenure == 0`. |
| 21 | `Churn` | string (target) | `{No, Yes}` | no | Target: abandonó el servicio. |

### 2.1 Diccionario de categorías codificadas

- `SeniorCitizen` usa codificación numérica `0/1` en lugar de `No/Yes`, a
  diferencia del resto de los campos binarios. Se debe normalizar durante la
  limpieza (`T-07`) para mantener coherencia de esquema.
- Los valores `"No phone service"` y `"No internet service"` expresan
  **no aplicabilidad** (el servicio base no está contratado), no una negación
  simple. Deben distinguirse de `"No"` en el análisis y el preprocessing.

## 3. Reglas de coherencia entre campos

Reglas derivadas de la lógica del negocio del dataset; serán verificadas por
los checks de `T-06`:

1. `MultipleLines == "No phone service"` si y solo si `PhoneService == "No"`.
2. `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
   `StreamingTV` y `StreamingMovies` toman el valor `"No internet service"` si y
   solo si `InternetService == "No"`.
3. Si `tenure == 0`, entonces `TotalCharges` está ausente (sin cargos aún).
4. `customerID` es único: no hay identificadores repetidos.

## 4. Reglas de calidad exigidas

Estas reglas son la base de los checks automatizados de `T-06`. Un dataset que
no las satisface **no avanza** a la fase de limpieza/modelado
(`docs/constitution.md`, sección 3; `docs/specs.md`, sección 5.2).

| Regla | Descripción |
|---|---|
| R1 | El esquema (nombres y orden de columnas) coincide con el de la sección 2. |
| R2 | Cada campo categórico contiene solo valores dentro de su dominio. |
| R3 | Los campos numéricos son parseables y caen dentro de su rango. |
| R4 | `customerID` es único. |
| R5 | No existen filas duplicadas. |
| R6 | Se cumplen las reglas de coherencia entre campos (sección 3). |
| R7 | Las ausencias de `TotalCharges` ocurren solo cuando `tenure == 0`. |

## 5. Estado de las ausencias

El raw no contiene valores nulos textuales en ninguna columna, pero
`TotalCharges` está almacenado como texto y presenta **11 cadenas vacías**
equivalentes a ausencia, todas con `tenure == 0`. Estas ausencias son
**estructurales** (dependen de la lógica del negocio), no aleatorias. Su
tratamiento se define en `T-08` con justificación basada en datos. Este
hallazgo se documenta formalmente en el audit (`T-05`).
