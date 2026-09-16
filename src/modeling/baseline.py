"""Baseline simple con las métricas del problema (T-11).

Entrena un clasificador *dummy* sobre la partición de entrenamiento (estrategia
`prior`: predice la probabilidad previa de churn para todos los clientes) y
evalúa sobre la partición de validación con las métricas definidas en la
sección 7 de `docs/specs.md` (recall, precisión, F1, AUC-ROC, AUC-PR sobre la
clase de interés; accuracy solo como referencia).

El ajuste se realiza únicamente con datos de entrenamiento (sin data leakage).
La partición de prueba queda reservada para la selección final (`T-14`).

Ejecución:
    python -m src.modeling.baseline
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.contract import TARGET
from src.paths import (
    BASELINE_REPORT_JSON as REPORT_JSON,
)
from src.paths import (
    BASELINE_REPORT_MD as REPORT_MD,
)
from src.paths import (
    PROJECT_ROOT,
    TRAIN_FILE,
    VAL_FILE,
)
from src.seeds import RANDOM_SEED

POSITIVE = "Yes"
ABSTAIN_THRESHOLD = 0.5


def fit_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    model = DummyClassifier(strategy="prior")
    model.fit(X_train, y_train)
    return model


def prob_positive(model: DummyClassifier, X: pd.DataFrame) -> np.ndarray:
    classes = model.classes_
    idx = list(classes).index(POSITIVE)
    prob = model.predict_proba(X)
    return np.asarray(prob[:, idx])


def evaluate_metrics(y_true: pd.Series, y_score_pos: np.ndarray) -> dict[str, object]:
    y_bin = y_true.map({POSITIVE: 1}).fillna(0).astype(int).to_numpy()
    y_pred = np.where(y_score_pos >= ABSTAIN_THRESHOLD, POSITIVE, "No")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["No", POSITIVE]).ravel()
    return {
        "auc_roc": float(roc_auc_score(y_bin, y_score_pos)),
        "auc_pr": float(average_precision_score(y_bin, y_score_pos)),
        "recall_pos": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)
        ),
        "precision_pos": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)
        ),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def build_report(
    metrics: dict[str, object],
    train_size: int,
    val_size: int,
    prevalence: float,
) -> str:
    return "\n".join(
        [
            "# Baseline (T-11)",
            "",
            "Clasificador *dummy* con estrategia `prior` (predice la "
            f"probabilidad previa de churn para todos los clientes), ajustado "
            f"solo sobre `train` ({train_size} filas) y evaluado sobre "
            f"`validation` ({val_size} filas).",
            "",
            "| Métrica | Valor (clase `Yes`) |",
            "|---|---|",
            f"| prevalence (train) | {prevalence:.4f} |",
            f"| AUC-ROC | {metrics['auc_roc']:.4f} |",
            f"| AUC-PR | {metrics['auc_pr']:.4f} |",
            f"| Recall | {metrics['recall_pos']:.4f} |",
            f"| Precisión | {metrics['precision_pos']:.4f} |",
            f"| F1 | {metrics['f1_pos']:.4f} |",
            f"| Accuracy (referencia) | {metrics['accuracy']:.4f} |",
            "",
            "Matriz de confusión (umbral 0.5):",
            "",
            f"- TP={metrics['true_positive']}, FN={metrics['false_negative']}, "
            f"FP={metrics['false_positive']}, TN={metrics['true_negative']}.",
            "",
            "Interpretación: al predecir siempre la clase previa, el umbral "
            "0.5 no predice ningún `Yes` (recall 0) y AUC-ROC ≈ 0.5; AUC-PR "
            "coincide con la prevalencia. Es el valor de referencia que los "
            "candidatos de `T-13` deben superar. La partición `test` queda "
            "reservada para la selección final (`T-14`).",
            "",
        ]
    )


def main() -> int:
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    y_train = train[TARGET]
    X_train = train.drop(columns=[TARGET])

    model = fit_baseline(X_train, y_train)
    y_val = val[TARGET]
    prob_yes = prob_positive(model, val.drop(columns=[TARGET]))

    metrics = evaluate_metrics(y_val, prob_yes)
    prevalence = float((y_train == POSITIVE).mean())

    report_json = {
        "task": "T-11",
        "model": type(model).__name__,
        "strategy": "prior",
        "train_file": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
        "validation_file": str(VAL_FILE.relative_to(PROJECT_ROOT)),
        "train_size": int(len(train)),
        "validation_size": int(len(val)),
        "prevalence_train": prevalence,
        "seed": RANDOM_SEED,
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    REPORT_JSON.write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(
        build_report(metrics, len(train), len(val), prevalence),
        encoding="utf-8",
    )
    print(f"Baseline evaluado: {REPORT_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
