"""Pipeline de preprocessing reproducible y sin data leakage (T-12).

Construye una unidad reutilizable de transformación que se ajusta
**únicamente** con datos de entrenamiento y se aplica de forma idéntica a
train, validación, prueba e inferencia:

- **Selección**: whitelist de features `FEATURE_NAMES` (excluye el
  identificador `customerID` y el target `Churn`); `remainder="drop"` descarta
  cualquier columna no seleccionada.
- **Imputación**: numéricos con mediana y categóricos con moda, ajustadas con
  train (inferencia puede traer valores nulos, p. ej. `TotalCharges`).
- **Escalado**: `StandardScaler` sobre numéricos (media 0/desviación 1 de
  train).
- **Codificación**: `OneHotEncoder` sobre categóricos con
  `handle_unknown="ignore"` (categoría nueva en inferencia se codifica a
  ceros, no rompe el pipeline).

Las funciones `fit_preprocessing` y `transform_features` separan el ajuste de
la aplicación: `transform_features` no ajusta nada, lo que hace imposible
filtrar información de validación/prueba/inferencia hacia el transformador.

Uso:

    pipeline = fit_preprocessing(X_train)
    X_train_t = transform_features(pipeline, X_train)
    X_val_t = transform_features(pipeline, X_val)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.analysis.features import FEATURE_COLUMNS
from src.data.contract import COLUMNS, TARGET

NUMERIC_FEATURE_NAMES = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "addon_missing_count",
    "avg_monthly_charge_hist",
)

# Orden canónico de features para modelado: esquema procesado (sin
# identificador ni target) + features derivadas, conservando el orden original.
FEATURE_NAMES = tuple(
    name for name in (*COLUMNS, *FEATURE_COLUMNS) if name not in ("customerID", TARGET)
)

CATEGORIC_FEATURE_NAMES = tuple(
    name for name in FEATURE_NAMES if name not in NUMERIC_FEATURE_NAMES
)


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Devuelve el pipeline de preprocessing sin ajustar."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(NUMERIC_FEATURE_NAMES)),
            ("cat", categorical_transformer, list(CATEGORIC_FEATURE_NAMES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessing(df: pd.DataFrame) -> ColumnTransformer:
    """Ajusta el pipeline con datos de entrenamiento (único ajuste permitido).

    ``df`` debe contener las columnas de ``FEATURE_NAMES``. Devuelve el
    pipeline ajustado; la aplicación posterior se hace con
    ``transform_features``.
    """
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(df[list(FEATURE_NAMES)])
    return pipeline


def transform_features(pipeline: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline ajustado sin reajustar nada.

    Solo transforma las columnas seleccionadas (``FEATURE_NAMES``) y devuelve
    un DataFrame con los nombres de features resultantes.
    """
    X = pipeline.transform(df[list(FEATURE_NAMES)])
    return pd.DataFrame(
        np.asarray(X),
        columns=list(pipeline.get_feature_names_out()),
    )
