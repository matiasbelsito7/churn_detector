"""Pruebas del entrenamiento de candidatos bajo condiciones idénticas (T-13)."""

import json

import numpy as np
import pandas as pd
import pytest
from src.data.contract import TARGET
from src.modeling import train
from src.modeling.train import (
    CANDIDATE_NAMES,
    build_candidate,
    build_training_report,
    experiment_record,
    train_candidate,
)


def make_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice(["No", "Yes"], n, p=[0.8, 0.2]),
        "Partner": rng.choice(["No", "Yes"], n),
        "Dependents": rng.choice(["No", "Yes"], n),
        "tenure": rng.randint(0, 73, n),
        "PhoneService": rng.choice(["No", "Yes"], n, p=[0.1, 0.9]),
        "MultipleLines": rng.choice(["No", "Yes", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["No", "Yes", "No internet service"], n),
        "OnlineBackup": rng.choice(["No", "Yes", "No internet service"], n),
        "DeviceProtection": rng.choice(["No", "Yes", "No internet service"], n),
        "TechSupport": rng.choice(["No", "Yes", "No internet service"], n),
        "StreamingTV": rng.choice(["No", "Yes", "No internet service"], n),
        "StreamingMovies": rng.choice(["No", "Yes", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["No", "Yes"], n),
        "PaymentMethod": rng.choice(
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            n,
        ),
        "MonthlyCharges": rng.uniform(18.25, 118.75, n),
        "TotalCharges": rng.uniform(0.0, 8684.8, n),
        "tenure_bin": rng.choice(
            ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59", "60-72"], n
        ),
        "is_new_client": rng.choice(["No", "Yes"], n, p=[0.8, 0.2]),
        "is_month_to_month": rng.choice(["No", "Yes"], n, p=[0.55, 0.45]),
        "is_electronic_check": rng.choice(["No", "Yes"], n, p=[0.66, 0.34]),
        "is_fiber_optic": rng.choice(["No", "Yes"], n, p=[0.56, 0.44]),
        "addon_missing_count": rng.randint(0, 5, n),
        "avg_monthly_charge_hist": rng.uniform(0.0, 121.4, n),
    }
    df = pd.DataFrame(data)
    churn_strength = (
        (df["tenure"] <= 3).astype(int) * 3.0
        + (df["is_month_to_month"] == "Yes").astype(int) * 2.0
        + (df["is_electronic_check"] == "Yes").astype(int) * 1.0
    )
    scores = churn_strength + rng.normal(0, 1.2, n)
    df[TARGET] = np.where(scores > np.median(scores), "Yes", "No")
    return df


def test_two_distinct_candidates_defined():
    assert len(CANDIDATE_NAMES) >= 2
    assert len(set(CANDIDATE_NAMES)) == len(CANDIDATE_NAMES)


def test_build_candidate_known_names():
    logistic = build_candidate("logistic-regression", seed=1)
    forest = build_candidate("random-forest", seed=1)
    assert type(logistic).__name__ == "LogisticRegression"
    assert type(forest).__name__ == "RandomForestClassifier"


def test_build_candidate_unknown_name_raises():
    with pytest.raises(ValueError):
        build_candidate("no-existe", seed=1)


def test_train_candidate_returns_metrics_in_range():
    df_train = make_df(600, seed=1)
    df_val = make_df(150, seed=2)
    result = train_candidate(CANDIDATE_NAMES[0], df_train, df_val)
    assert result.train_size == len(df_train)
    assert result.validation_size == len(df_val)
    for key in (
        "auc_roc",
        "auc_pr",
        "recall_pos",
        "precision_pos",
        "f1_pos",
        "accuracy",
    ):
        assert 0.0 <= result.metrics[key] <= 1.0


def test_validation_does_not_influence_model():
    df_train = make_df(600, seed=3)
    result_a = train_candidate("logistic-regression", df_train, make_df(150, seed=4))
    result_b = train_candidate("logistic-regression", df_train, make_df(150, seed=5))
    np.testing.assert_allclose(result_a.model.coef_, result_b.model.coef_)
    assert result_a.metrics != result_b.metrics


def test_train_candidate_is_deterministic():
    df_train = make_df(600, seed=6)
    df_val = make_df(150, seed=7)
    first = train_candidate("random-forest", df_train, df_val)
    second = train_candidate("random-forest", df_train, df_val)
    assert first.metrics == second.metrics
    assert first.params == second.params


def test_candidates_differ_in_metrics():
    df_train = make_df(600, seed=8)
    df_val = make_df(150, seed=9)
    logistic = train_candidate("logistic-regression", df_train, df_val)
    forest = train_candidate("random-forest", df_train, df_val)
    assert logistic.metrics != forest.metrics


def test_candidate_model_fitted_on_train():
    df_train = make_df(200, seed=10)
    df_val = make_df(50, seed=11)
    result = train_candidate("logistic-regression", df_train, df_val)
    assert set(result.model.classes_) == {"No", "Yes"}


def test_experiment_record_captures_config_and_data():
    df_train = make_df(200, seed=12)
    df_val = make_df(50, seed=13)
    result = train_candidate("logistic-regression", df_train, df_val)
    record = experiment_record(result, "sha-train", "sha-val")
    assert record["name"] == "logistic-regression"
    assert record["model"] == "LogisticRegression"
    assert record["train_size"] == 200
    assert record["validation_size"] == 50
    assert record["seed"] == result.seed
    assert record["input_files"]["train_sha256"] == "sha-train"
    assert record["input_files"]["validation_sha256"] == "sha-val"
    assert "auc_pr" in record["metrics"]


def test_build_training_report_lists_all_candidates():
    df_train = make_df(200, seed=14)
    df_val = make_df(50, seed=15)
    records = [
        experiment_record(train_candidate(name, df_train, df_val), "s1", "s2")
        for name in CANDIDATE_NAMES
    ]
    report = build_training_report(records)
    assert "# Entrenamiento de candidatos (T-13)" in report
    for name in CANDIDATE_NAMES:
        assert name in report
    assert "baseline `T-11`" in report


def test_main_writes_experiments_and_report(tmp_path, monkeypatch):
    df_train = make_df(300, seed=16)
    df_val = make_df(80, seed=17)
    train_file = tmp_path / "train.csv"
    val_file = tmp_path / "validation.csv"
    test_file = tmp_path / "test.csv"
    df_train.to_csv(train_file, index=False)
    df_val.to_csv(val_file, index=False)
    df_val.to_csv(test_file, index=False)
    experiments = tmp_path / "experiments.json"
    report = tmp_path / "training.md"
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train, "TRAIN_FILE", train_file)
    monkeypatch.setattr(train, "VAL_FILE", val_file)
    monkeypatch.setattr(train, "TEST_FILE", test_file)
    monkeypatch.setattr(train, "EXPERIMENTS_FILE", experiments)
    monkeypatch.setattr(train, "TRAINING_REPORT", report)
    assert train.main() == 0
    payload = json.loads(experiments.read_text(encoding="utf-8"))
    assert payload["task"] == "T-13"
    assert [rec["name"] for rec in payload["experiments"]] == list(CANDIDATE_NAMES)
    assert "test" in report.read_text(encoding="utf-8")
