"""Pruebas de la integración de MLflow (T-17) con URI de tracking aislada."""

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from src.modeling.tracking import (
    EXPERIMENT_NAME,
    METRIC_KEYS,
    log_candidate_run,
    verify_recovered_model,
)


def small_pipeline() -> Pipeline:
    rng = np.random.RandomState(0)
    X = rng.randn(80, 5)
    y = (rng.rand(80) > 0.5).astype(int)
    pipeline = Pipeline([("model", LogisticRegression(max_iter=500))])
    pipeline.fit(X, y)
    return pipeline


def tracking_uri(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def small_metrics() -> dict[str, object]:
    return {
        "auc_roc": 0.9,
        "auc_pr": 0.7,
        "recall_pos": 0.8,
        "precision_pos": 0.6,
        "f1_pos": 0.6,
        "accuracy": 0.75,
        "true_positive": 5,
    }


def test_log_candidate_run_creates_run(tmp_path):
    uri = tracking_uri(tmp_path)
    pipeline = small_pipeline()
    log_candidate_run(
        "test-model",
        pipeline,
        {"max_iter": 500},
        small_metrics(),
        ("f1", "f2"),
        tracking_uri=uri,
        experiment_name=EXPERIMENT_NAME,
    )
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    assert len(runs) == 1
    assert float(runs.iloc[0]["metrics.auc_pr"]) == 0.7
    for key in METRIC_KEYS:
        assert f"metrics.{key}" in runs.columns


def test_verify_recovered_model_matches_registered(tmp_path):
    uri = tracking_uri(tmp_path)
    rng = np.random.RandomState(1)
    X = rng.randn(40, 5)
    y = (rng.rand(40) > 0.5).astype(int)
    log_candidate_run(
        "test-model",
        small_pipeline(),
        {"max_iter": 500},
        small_metrics(),
        ("f1", "f2"),
        tracking_uri=uri,
        experiment_name=EXPERIMENT_NAME,
    )
    mlflow.set_tracking_uri(uri)
    loaded = mlflow.sklearn.load_model("models:/churn-test-model/latest")
    expected = average_precision_score(y, np.asarray(loaded.predict_proba(X))[:, 1])
    got = verify_recovered_model("test-model", pd.DataFrame(X), y, tracking_uri=uri)
    assert got == expected
