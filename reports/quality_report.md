# Informe de calidad de datos (T-06)

- **Fuente validada:** `data\raw\Telco-Customer-Churn.csv`
- **Estado del dataset:** **APROBADO** — puede avanzar a la siguiente fase.

| Regla | Descripción | Resultado | Detalle |
|---|---|---|---|
| R1 | Esquema de columnas | Pasa | El esquema coincide con el contrato. |
| R2 | Dominios categóricos | Pasa | Todos los dominios se respetan. |
| R3 | Números: parseo y rango | Pasa | Números válidos dentro de rango. |
| R4 | Unicidad y formato de customerID | Pasa | customerID único y con formato válido. |
| R5 | Filas duplicadas | Pasa | No hay filas duplicadas. |
| R6 | Coherencia entre campos | Pasa | Sin incoherencias entre campos. |
| R7 | Ausencia de TotalCharges solo con tenure == 0 | Pasa | 11 ausencias de TotalCharges. |

Reglas cumplidas: 7/7.
