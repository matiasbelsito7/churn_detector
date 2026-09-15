"""Pruebas del baseline simple con las métricas del problema (T-11)."""

import json

import numpy as np
import pandas as pd
import pytest
from src.modeling import baseline
from src.modeling.baseline import (
    POSITIVE,
    build_report,
    evaluate_metrics,
    fit_baseline,
    prob_positive,
)


def make_train_val() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(3)
    n_train, n_val = 400, 100
    X_tr = pd.DataFrame({"a": rng.rand(n_train), "b": rng.randint(0, 5, n_train)})
    y_tr = pd.Series(["Yes" if rng.rand() < 0.27 else "No" for _ in range(n_train)])
    X_va = pd.DataFrame({"a": rng.rand(n_val), "b": rng.randint(0, 5, n_val)})
    y_va = pd.Series(["Yes" if rng.rand() < 0.27 else "No" for _ in range(n_val)])
    return X_tr, y_tr, X_va, y_va


def test_fit_baseline_and_prior_probabilities():
    X_tr, y_tr, X_va, _ = make_train_val()
    model = fit_baseline(X_tr, y_tr)
    prob = prob_positive(model, X_va)
    prior = float((y_tr == POSITIVE).mean())
    assert np.allclose(prob, prior)


def test_evaluate_metrics_dummy_prior_baseline_values():
    X_tr, y_tr, X_va, y_va = make_train_val()
    model = fit_baseline(X_tr, y_tr)
    prob = prob_positive(model, X_va)
    metrics = evaluate_metrics(y_va, prob)
    assert metrics["auc_roc"] == 0.5
    assert metrics["auc_pr"] == pytest.approx(
        float((y_va == POSITIVE).mean()), abs=1e-6
    )


def test_recall_zero_under_prior_threshold():
    X_tr, y_tr, X_va, y_va = make_train_val()
    model = fit_baseline(X_tr, y_tr)
    prob = prob_positive(model, X_va)
    assert (prob < 0.5).all()
    metrics = evaluate_metrics(y_va, prob)
    assert metrics["recall_pos"] == 0.0
    assert metrics["true_positive"] == 0


def test_evaluate_metrics_perfect_scores():
    y_true = pd.Series(["No", "No", "Yes", "Yes"])
    y_score = np.array([0.1, 0.2, 0.9, 0.8])
    metrics = evaluate_metrics(y_true, y_score)
    assert metrics["auc_roc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["true_positive"] == 2
    assert metrics["false_negative"] == 0


def test_evaluate_metrics_inverted_scores():
    y_true = pd.Series(["No", "No", "Yes", "Yes"])
    y_score = np.array([0.9, 0.8, 0.1, 0.2])
    metrics = evaluate_metrics(y_true, y_score)
    assert metrics["auc_roc"] == 0.0
    assert metrics["recall_pos"] == 0.0
    assert metrics["true_positive"] == 0


def test_build_report_contains_sizes_and_metrics():
    metrics = evaluate_metrics(
        pd.Series(["No", "No", "Yes", "Yes"]), np.array([0.1, 0.2, 0.9, 0.8])
    )
    report = build_report(metrics, train_size=400, val_size=100, prevalence=0.265)
    assert "# Baseline (T-11)" in report
    assert "400 filas" in report
    assert "100 filas" in report
    assert "0.2650" in report
    assert "AUC-ROC" in report
    assert "TP=2" in report


def test_main_writes_reports(tmp_path, monkeypatch):
    rng = np.random.RandomState(3)
    n_tr, n_va = 100, 40

    def frame(n):
        return pd.DataFrame(
            {
                "customerID": [f"{i:04d}-AAAAA" for i in range(n)],
                "a": rng.rand(n),
                "b": rng.randint(0, 5, n),
                "Churn": ["Yes" if rng.rand() < 0.27 else "No" for _ in range(n)],
            }
        )

    train = tmp_path / "train.csv"
    val = tmp_path / "validation.csv"
    frame(n_tr).to_csv(train, index=False)
    frame(n_va).to_csv(val, index=False)
    report_md = tmp_path / "baseline.md"
    report_json = tmp_path / "baseline_metrics.json"
    monkeypatch.setattr(baseline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(baseline, "TRAIN_FILE", train)
    monkeypatch.setattr(baseline, "VAL_FILE", val)
    monkeypatch.setattr(baseline, "REPORT_MD", report_md)
    monkeypatch.setattr(baseline, "REPORT_JSON", report_json)
    assert baseline.main() == 0
    assert "Baseline (T-11)" in report_md.read_text(encoding="utf-8")
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["task"] == "T-11"
    assert payload["train_size"] == n_tr
    assert payload["validation_size"] == n_va
    assert "auc_roc" in payload["metrics"]
