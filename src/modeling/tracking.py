"""Integración de MLflow para tracking de experimentos (T-17).

Registra cada experimento de `T-13` (logistic-regression y random-forest) en
el tracking: parámetros, métricas del problema y artefactos (métricas en JSON,
nombres de features y el pipeline completo ajustado). El pipeline completo
(preprocessing `T-12` + modelo) se publica como artefacto de modelo, de modo
que quede recuperable y aplicable para inferencia.

El backend de tracking es una base SQLite local (`mlflow.db`, no versionada);
los artefactos se guardan en `mlruns/` (no versionado). Ambos se regeneran
ejecutando el módulo.

Uso local:

    python -m src.modeling.tracking
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Detalle:
- Cada corrida se entrena de nuevo de forma reproducible (partición y semilla
  fijas de `T-11`/`T-13`), de modo que las métricas registradas coinciden con
  las del registro en `reports/experiments.json`.
- Los modelos se publican con la serialización `cloudpickle` (es la que soporta
  el gráfico de objetos de nuestro pipeline, incluyendo metadatos `numpy`);
  permite empaquetar las dependencias de código y cargar el pipeline completo
  para inferencia sin entrenar de nuevo.
- Los modelos se registran en el *Model Registry* local como
  `churn-logistic-regression` y `churn-random-forest` (versión `latest`), a la
  espera de que la selección de `T-14` promueva el modelo elegido.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.contract import TARGET
from src.modeling.preprocessing import FEATURE_NAMES
from src.modeling.train import (
    CANDIDATE_NAMES,
    train_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_FILE = PROJECT_ROOT / "data" / "splits" / "validation.csv"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

EXPERIMENT_NAME = "churn-detector"
MODEL_PREFIX = "churn-"

METRIC_KEYS = (
    "auc_roc",
    "auc_pr",
    "recall_pos",
    "precision_pos",
    "f1_pos",
    "accuracy",
)


def _as_float(value: object) -> float:
    """Convierte un valor de métrica (real o numpy escalar) a float."""
    return float(np.asarray(value, dtype=float))


def log_candidate_run(
    name: str,
    pipeline: Pipeline,
    params: dict[str, Any],
    metrics: dict[str, object],
    feature_names: tuple[str, ...],
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
) -> None:
    """Crea una corrida MLflow con configuración, métricas y artefactos."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({key: _as_float(metrics[key]) for key in METRIC_KEYS})
        mlflow.log_text(
            json.dumps({**metrics, "feature_names": list(feature_names)}, indent=2),
            "metrics.json",
        )
        mlflow.log_text(json.dumps(list(feature_names), indent=2), "features.json")
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=f"{MODEL_PREFIX}{name}",
            serialization_format="cloudpickle",
        )


def verify_recovered_model(
    name: str,
    X: pd.DataFrame,
    y_bin: np.ndarray,
    tracking_uri: str = TRACKING_URI,
) -> float:
    """Carga el modelo registrado y devuelve AUC-PR sobre ``y_bin`` (0/1)."""
    mlflow.set_tracking_uri(tracking_uri)
    loaded = mlflow.sklearn.load_model(f"models:/churn-{name}/latest")
    prob = np.asarray(loaded.predict_proba(X))[:, 1]
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_bin, prob))


def main() -> int:
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)
    X_val = df_val[list(FEATURE_NAMES)]
    y_bin = df_val[TARGET].map({"Yes": 1, "No": 0}).astype(int).to_numpy()

    for name in CANDIDATE_NAMES:
        result = train_candidate(name, df_train, df_val)
        log_candidate_run(
            name=result.name,
            pipeline=result.pipeline,
            params=result.params,
            metrics=result.metrics,
            feature_names=FEATURE_NAMES,
        )
        print(
            f"[{datetime.now(UTC):%H:%M:%S}] {name}: "
            f"AUC-PR={result.metrics['auc_pr']:.4f}"
        )

    for name in CANDIDATE_NAMES:
        auc_pr = verify_recovered_model(name, X_val, y_bin)
        print(f"Modelo registrado {name} recuperado y evaluado: AUC-PR={auc_pr:.4f}")

    print(
        f"Tracking en {MLFLOW_DB.relative_to(PROJECT_ROOT)}/ - "
        "ver con: mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
