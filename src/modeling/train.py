"""Entrenamiento de candidatos de modelo bajo condiciones idénticas (T-13).

Entrena al menos tres familias de modelo distintas (regresión logística,
random forest y xgboost) sobre la partición de `train` (T-11) y el pipeline
de preprocessing de `T-12`, evaluándolas sobre la misma partición de
`validation` y con las mismas métricas definidas en `docs/specs.md` (§7) y el
baseline de `T-11`.

Garantías:
- Condiciones idénticas: misma partición, mismas métricas, misma semilla.
- Sin data leakage: el pipeline de preprocessing (T-12) y cada modelo se
  ajustan exclusivamente con `train`; `validation` solo se evalua y `test`
  queda reservado para `T-14`.
- Reproducible: semilla fija (`RANDOM_SEED`) y configuraciones explícitas que
  se registran junto a cada experimento.

Cada experimento queda registrado en `reports/experiments.json` (datos,
configuración y métricas) y resumido en `reports/training.md`. La integración
con MLflow ocurre en `T-17`.

Ejecución:
    python -m src.modeling.train
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data.contract import POSITIVE_CLASS, TARGET
from src.hashing import sha256_file
from src.modeling.baseline import evaluate_metrics
from src.modeling.preprocessing import (
    FEATURE_NAMES,
    build_preprocessing_pipeline,
)
from src.paths import (
    EXPERIMENTS_JSON as EXPERIMENTS_FILE,
)
from src.paths import (
    PROJECT_ROOT,
    TEST_FILE,
    TRAIN_FILE,
    TRAINING_REPORT,
    VAL_FILE,
)
from src.seeds import RANDOM_SEED


def _logistic_builder(seed: int, **overrides: Any) -> LogisticRegression:
    params: dict[str, Any] = {
        "class_weight": "balanced",
        "max_iter": 2000,
    }
    params.update(overrides)
    return LogisticRegression(random_state=seed, **params)


def _forest_builder(seed: int, **overrides: Any) -> RandomForestClassifier:
    params: dict[str, Any] = {
        "n_estimators": 300,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    params.update(overrides)
    return RandomForestClassifier(random_state=seed, **params)


class _LabelEncodedXGBClassifier(XGBClassifier):
    """XGBClassifier (xgboost>=3) que acepta el target `No`/`Yes` del proyecto.

    `xgboost>=3` no codifica etiquetas de clase automáticamente y espera un
    target numérico. Este wrapper codifica `No`/`Yes` con `LabelEncoder` al
    ajustar y expone `classes_` con las etiquetas originales después del
    ajuste, de modo que `prob_yes` y las métricas interactúan con él igual que
    con el resto de candidatos. Internamente (`_fitted_ == False`) mantiene la
    convención numérica que xgboost usa durante `fit`.

    `xgboost>=3` ya no consume `class_weight`; este wrapper lo respeta
    traduciéndolo a `scale_pos_weight` (parámetro nativo) y lo retira de los
    parámetros que se envían al booster.
    """

    def fit(self, X: Any, y: Any, **kwargs: Any) -> _LabelEncodedXGBClassifier:
        class_weight = getattr(self, "kwargs", {}).get("class_weight")
        if class_weight == "balanced" and self.scale_pos_weight is None:
            positive = int((np.asarray(y) == POSITIVE_CLASS).sum())
            negative = int(len(y) - positive)
            self.scale_pos_weight = negative / max(positive, 1)
        if hasattr(self, "kwargs") and "class_weight" in self.kwargs:
            self.kwargs.pop("class_weight")
        self._encoder = LabelEncoder().fit(y)
        super().fit(X, self._encoder.transform(y), **kwargs)
        self._fitted_ = True
        return self

    @property
    def classes_(self) -> np.ndarray:
        if getattr(self, "_fitted_", False):
            return self._encoder.classes_
        return super().classes_


def _xgboost_builder(seed: int, **overrides: Any) -> _LabelEncodedXGBClassifier:
    params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.075,
        "subsample": 1.0,
        "tree_method": "hist",
        "class_weight": "balanced",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    params.update(overrides)
    return _LabelEncodedXGBClassifier(random_state=seed, **params)


CANDIDATES: dict[str, Callable[..., Any]] = {
    "logistic-regression": _logistic_builder,
    "random-forest": _forest_builder,
    "xgboost": _xgboost_builder,
}


def build_candidate(name: str, seed: int, **overrides: Any) -> Any:
    """Devuelve el estimador del candidato ``name`` con semilla ``seed``.

    ``overrides`` permite ajustar parámetros (p. ej. en tests) sin cambiar la
    configuración por defecto del experimento. Añadir un candidato nuevo solo
    requiere registrar su constructor en ``CANDIDATES`` (OCP).
    """
    try:
        builder = CANDIDATES[name]
    except KeyError as exc:
        raise ValueError(f"Candidato desconocido: {name}") from exc
    return builder(seed, **overrides)


CANDIDATE_NAMES = tuple(CANDIDATES)


def prob_yes(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de la clase positiva a partir de un pipeline ajustado."""
    model = pipeline.named_steps["model"]
    classes = model.classes_
    idx = list(classes).index(POSITIVE_CLASS)
    return np.asarray(pipeline.predict_proba(X)[:, idx])


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    model: Any
    pipeline: Any
    params: dict[str, Any]
    metrics: dict[str, object]
    train_size: int
    validation_size: int
    seed: int


def train_candidate(
    name: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    seed: int = RANDOM_SEED,
    **model_overrides: Any,
) -> ExperimentResult:
    """Entrena ``name`` con preprocessing ajustado solo en train y lo evalua.

    Devuelve el resultado con el modelo, su configuración y las métricas sobre
    la partición de validación. ``test`` nunca participa aquí.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", build_candidate(name, seed, **model_overrides)),
        ]
    )
    pipeline.fit(df_train[list(FEATURE_NAMES)], df_train[TARGET])

    y_val = df_val[TARGET]
    prob = prob_yes(pipeline, df_val[list(FEATURE_NAMES)])
    metrics = evaluate_metrics(y_val, prob)

    model = pipeline.named_steps["model"]
    return ExperimentResult(
        name=name,
        model=model,
        pipeline=pipeline,
        params=model.get_params(deep=True),
        metrics=metrics,
        train_size=int(len(df_train)),
        validation_size=int(len(df_val)),
        seed=seed,
    )


def experiment_record(
    result: ExperimentResult, train_sha: str, val_sha: str
) -> dict[str, object]:
    return {
        "name": result.name,
        "model": type(result.model).__name__,
        "params": result.params,
        "metrics": result.metrics,
        "train_size": result.train_size,
        "validation_size": result.validation_size,
        "seed": result.seed,
        "input_files": {
            "train": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
            "train_sha256": train_sha,
            "validation": str(VAL_FILE.relative_to(PROJECT_ROOT)),
            "validation_sha256": val_sha,
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def build_training_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Entrenamiento de candidatos (T-13)",
        "",
        "Candidatos entrenados bajo condiciones idénticas: partición `train` "
        "de `T-11`, pipeline de preprocessing de `T-12` (ajustado solo en "
        "train) y evaluación sobre `validation` con las métricas del problema. "
        "La partición `test` queda reservada para `T-14`.",
        "",
        "| Modelo | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        m = rec["metrics"]
        lines.append(
            f"| {rec['name']} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} | "
            f"{m['recall_pos']:.4f} | {m['precision_pos']:.4f} | "
            f"{m['f1_pos']:.4f} | {m['accuracy']:.4f} |"
        )
    lines += [
        "",
        "Referencia (baseline `T-11`): AUC-PR 0.2658, recall 0.0 a umbral 0.5. "
        "El detalle de configuración y datos de cada experimento vive en "
        "`reports/experiments.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)

    records: list[dict[str, object]] = []
    for name in CANDIDATE_NAMES:
        result = train_candidate(name, df_train, df_val)
        records.append(
            experiment_record(result, sha256_file(TRAIN_FILE), sha256_file(VAL_FILE))
        )

    payload = {
        "task": "T-13",
        "seed": RANDOM_SEED,
        "test_file": {
            "path": str(TEST_FILE.relative_to(PROJECT_ROOT)),
            "reserved": "Se usa solo en la selección final (T-14).",
        },
        "leakage_note": (
            "Preprocessing (T-12) y modelos ajustados únicamente con train; "
            "validation solo se evalua; test no participa en T-13."
        ),
        "experiments": records,
    }
    EXPERIMENTS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    TRAINING_REPORT.write_text(build_training_report(records), encoding="utf-8")
    print(f"Experimentos registrados: {EXPERIMENTS_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
