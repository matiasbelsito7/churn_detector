"""Predicción de churn con el modelo seleccionado (T-15).

Produce, para un conjunto de clientes sin target conocido, la salida definida
en `docs/specs.md` §8.1: identificador del cliente, probabilidad de churn y
clase predicha (umbral 0.5). Aplica exactamente el mismo procesamiento que en
entrenamiento: el modelo cargado desde el Model Registry de MLflow es el
pipeline completo (preprocessing `T-12` + modelo seleccionado en `T-14`).

La transformación de features (T-10) se asume ya aplicada sobre la entrada
(dataset de features), igual que en entrenamiento; `predict_clients` es pura y
determinista: misma entrada → misma salida.

Ejecución:
    python -m src.serving.predict
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.hashing import sha256_file
from src.modeling.preprocessing import FEATURE_NAMES
from src.modeling.select import load_selected_model
from src.modeling.train import prob_yes
from src.paths import (
    FEATURES_FILE,
    PROJECT_ROOT,
)
from src.paths import (
    PREDICTION_LOG as LOG_FILE,
)
from src.paths import (
    PREDICTIONS_DIR as OUTPUT_DIR,
)
from src.paths import (
    PREDICTIONS_FILE as OUTPUT_FILE,
)

PREDICTION_THRESHOLD = 0.5
MODEL_OUTPUT_COLUMNS = ("customerID", "churn_prob", "churn_class")


def predict_clients(
    model: Pipeline,
    df: pd.DataFrame,
    threshold: float = PREDICTION_THRESHOLD,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> pd.DataFrame:
    """Devuelve customerID, probabilidad de churn y clase predicha.

    ``df`` debe contener ``customerID`` y las columnas ``feature_names`` (las
    features del modelo) y puede incluir el target; el target se descarta antes
    de predecir.
    """
    if "customerID" not in df.columns:
        raise ValueError("La entrada debe incluir la columna customerID")
    X = df[list(feature_names)]
    prob = np.asarray(prob_yes(model, X))
    out = pd.DataFrame(
        {
            "customerID": df["customerID"].to_numpy(),
            "churn_prob": prob,
            "churn_class": np.where(prob >= threshold, "Yes", "No"),
        }
    )
    return out


def main() -> int:
    df = pd.read_csv(FEATURES_FILE)
    model = load_selected_model("logistic-regression")
    predictions = predict_clients(model, df)
    predictions_sorted = predictions.sort_values("customerID").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_sorted.to_csv(OUTPUT_FILE, index=False)

    features_sha = sha256_file(FEATURES_FILE)
    LOG_FILE.write_text(
        json.dumps(
            {
                "task": "T-15",
                "model": "churn-logistic-regression@production",
                "model_version": "production",
                "threshold": PREDICTION_THRESHOLD,
                "input_features": {
                    "path": str(FEATURES_FILE.relative_to(PROJECT_ROOT)),
                    "sha256": features_sha,
                    "rows": int(len(df)),
                },
                "output": {
                    "path": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
                    "rows": int(len(predictions)),
                    "sha256": sha256_file(OUTPUT_FILE),
                },
                "churn_rate_pred": float(
                    (predictions_sorted["churn_class"] == "Yes").mean()
                ),
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Predicciones: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
