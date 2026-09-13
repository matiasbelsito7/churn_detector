"""Pruebas de evaluación y selección del modelo (T-14)."""

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.modeling.select import (
    SELECTION_RULE,
    build_selection_report,
    load_selected_model,
    log_selection_run,
    promote_model,
    select_candidate,
)
from src.modeling.tracking import (
    EXPERIMENT_NAME,
    METRIC_KEYS,
    log_candidate_run,
)


def record(name: str, auc_pr: float, recall: float) -> dict:
    return {
        "name": name,
        "metrics": {"auc_pr": auc_pr, "recall_pos": recall},
    }


def test_select_candidate_prizes_auc_pr():
    records = [
        record("random-forest", 0.63, 0.80),
        record("logistic-regression", 0.67, 0.77),
    ]
    assert select_candidate(records) == "logistic-regression"


def test_select_candidate_ties_break_by_recall():
    records = [
        record("random-forest", 0.66, 0.80),
        record("logistic-regression", 0.66, 0.77),
    ]
    assert select_candidate(records) == "random-forest"


def small_pipeline() -> Pipeline:
    rng = np.random.RandomState(0)
    X = rng.randn(80, 5)
    y = (rng.rand(80) > 0.5).astype(int)
    pipeline = Pipeline([("model", LogisticRegression(max_iter=500))])
    pipeline.fit(X, y)
    return pipeline


def small_metrics() -> dict[str, object]:
    return {
        "auc_roc": 0.9,
        "auc_pr": 0.7,
        "recall_pos": 0.8,
        "precision_pos": 0.6,
        "f1_pos": 0.6,
        "accuracy": 0.75,
    }


def tracking_uri(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_promote_and_load_selected_model(tmp_path):
    uri = tracking_uri(tmp_path)
    log_candidate_run(
        "test-model",
        small_pipeline(),
        {"max_iter": 500},
        small_metrics(),
        ("f1", "f2"),
        tracking_uri=uri,
        experiment_name=EXPERIMENT_NAME,
    )
    version = promote_model("test-model", tracking_uri=uri)
    assert version == 1
    loaded = load_selected_model("test-model", tracking_uri=uri)
    rng = np.random.RandomState(1)
    X = rng.randn(10, 5)
    prob = loaded.predict_proba(X)
    assert prob.shape == (10, 2)


def test_log_selection_run_registers_params_and_metrics(tmp_path):
    uri = tracking_uri(tmp_path)
    run_id = log_selection_run(
        "test-model",
        1,
        small_metrics(),
        tracking_uri=uri,
        experiment_name=EXPERIMENT_NAME,
    )
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == run_id
    assert runs.iloc[0]["tags.mlflow.runName"] == "seleccion-test-model"
    assert runs.iloc[0]["params.selected_candidate"] == "test-model"
    for key in METRIC_KEYS:
        assert f"metrics.{key}" in runs.columns


def test_build_selection_report_is_complete():
    records = [
        {
            "name": "random-forest",
            "metrics": {
                "auc_roc": 0.8415,
                "auc_pr": 0.6362,
                "recall_pos": 0.77,
                "precision_pos": 0.52,
                "f1_pos": 0.62,
                "accuracy": 0.75,
            },
        },
        {
            "name": "logistic-regression",
            "metrics": {
                "auc_roc": 0.8421,
                "auc_pr": 0.6686,
                "recall_pos": 0.80,
                "precision_pos": 0.51,
                "f1_pos": 0.62,
                "accuracy": 0.74,
            },
        },
    ]
    test_metrics = {
        "auc_roc": 0.85,
        "auc_pr": 0.67,
        "recall_pos": 0.81,
        "precision_pos": 0.52,
        "f1_pos": 0.63,
        "accuracy": 0.75,
        "true_positive": 10,
        "false_negative": 5,
        "false_positive": 3,
        "true_negative": 20,
    }
    report = build_selection_report(
        records, "logistic-regression", test_metrics, 38, "abc123"
    )
    assert "# Selección del modelo (T-14)" in report
    assert "logistic-regression" in report
    assert "churn-logistic-regression/production" in report
    assert "sha256 `abc123`" in report
    assert "0.6686" in report
    assert SELECTION_RULE in report
    assert "TP=10" in report
