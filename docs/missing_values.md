# Tratamiento de valores faltantes (T-08)

## 1. Alcance

Análisis de presencia de valores faltantes sobre el dataset procesado
(`T-07`), decisión de tratamiento y justificación basada en datos. Debe
respetar `docs/constitution.md` (sección 3): cada decisión se documenta con su
motivo y evidencia; no se aplica un método por defecto.

## 2. Presencia de valores faltantes

Tras la limpieza (`T-07`), único campo con valores faltantes:

- **`TotalCharges`**: 11 valores ausentes sobre 7043 filas (0.16 %).

Resto de campos: 0 ausencias.

## 3. Evidencia del patrón

| Criterio | Resultado |
|---|---|
| Filas con `TotalCharges` ausente | 11 |
| Filas con `tenure == 0` | 11 |
| Intersección (`tenure == 0` y ausente) | 11 (100 %) |
| `tenure` en filas ausentes | siempre `0` |
| `Churn` en filas ausentes | siempre `No` (11/11) |
| `MonthlyCharges` en filas ausentes | rango normal [19.70, 80.85], media 41.4 |

Conclusión: la ausencia es **determinística y estructural**, no aleatoria. Un
cliente con `tenure == 0` aún no tiene cargos acumulados; la ausencia de
`TotalCharges` está perfectamente determinada por `tenure == 0`. No hay otros
campos afectados ni patrones combinados de ausencia.

## 4. Decisión de tratamiento

**Imputar `TotalCharges = 0` para todo caso faltante.**

### 4.1 Justificación

1. **Definición de negocio:** `TotalCharges` es el monto acumulado facturado al
   cliente. Con cero meses de antigüedad el acumulado es exactamente 0.
2. **Coherencia con los datos:** los 11 casos son exactamente los de
   `tenure == 0`, por lo que la imputación con 0 es exacta, no aproximada.
3. **La ausencia no informa sobre el target:** los 11 casos son `Churn = No`
   porque el cliente es nuevo; la presencia/ausencia no aporta señal extra por
   sí misma.
4. **Inferencia reproducible:** la regla es una constante (0), aplicada de forma
   idéntica en entrenamiento e inferencia sin estado aprendido.

### 4.2 Alternativas descartadas

| Alternativa | Motivo de rechazo |
|---|---|
| Descartar las filas | Elimina 0.16 % de datos sin necesidad; la lógica de negocio permite un valor exacto. |
| Marcar con bandera de ausencia | Redundante: `tenure == 0` ya codifica el caso de forma completa. |
| Imputar por media/mediana | Erróneo conceptualmente: atribuiría un acumulado de un cliente antiguo a uno nuevo. |
| Imputación por modelo/regresión | Sobre-ingeniería para una ausencia determinística (constitution §4). |

## 5. Implementación

Regla implementada como paso del pipeline de limpieza en
`src/data/clean.py` (`clean_raw`): tras normalizar `TotalCharges` a flotante,
todo valor ausente se rellena con `0.0`.

Resultado: el dataset procesado (`data/processed/churn_cleaned.csv`) queda **sin
valores faltantes** y es el artefacto consumido por la fase de EDA (`T-09`).
