# Persistencia en PostgreSQL (T-16)

Esquema cargado en el contenedor PostgreSQL local (`docker-compose.yml`): tabla `predictions` con la salida de inferencia por cliente y tabla `source_files` para la trazabilidad al origen versionado (sha256 del dataset de features).

Predicciones cargadas: **7043** (fuentes: 1).

Verificación por SQL:

- Predicciones totales: 7043.
- Clientes distintos: 7043.
- Clases fuera de {Yes, No}: 0.
- Predicciones con origen trazable: 7043.
- Distribución por clase: Yes=2907, No=4136.
- Probabilidad en rango [0,1]: [0.0131, 0.9705].
