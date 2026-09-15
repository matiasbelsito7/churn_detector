"""Pruebas de las particiones reproducibles (T-11)."""

import json

import numpy as np
import pandas as pd
from src.analysis.features import FEATURE_COLUMNS
from src.data.contract import COLUMNS
from src.modeling import split
from src.modeling.split import build_log, create_partitions


def sample_df(n: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-AAAAA" for i in range(n)],
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
            "Churn": rng.choice(["No", "Yes"], n, p=[0.735, 0.265]),
        },
        columns=list(COLUMNS),
    )
    df["tenure_bin"] = "0-5"
    df["is_new_client"] = "No"
    df["is_month_to_month"] = "No"
    df["is_electronic_check"] = "No"
    df["is_fiber_optic"] = "No"
    df["addon_missing_count"] = 0
    df["avg_monthly_charge_hist"] = 0.0
    return df


def ids(df: pd.DataFrame) -> set[str]:
    return set(df["customerID"])


def test_split_covers_all_rows_once():
    df = sample_df()
    split = create_partitions(df)
    union = ids(split.train) | ids(split.validation) | ids(split.test)
    assert ids(split.train).isdisjoint(ids(split.validation))
    assert ids(split.train).isdisjoint(ids(split.test))
    assert ids(split.validation).isdisjoint(ids(split.test))
    assert union == ids(df) and len(union) == len(df)


def test_split_preserves_schema():
    df = sample_df()
    split = create_partitions(df)
    expected = list(COLUMNS) + list(FEATURE_COLUMNS)
    for part in (split.train, split.validation, split.test):
        assert list(part.columns) == expected


def test_split_is_deterministic():
    df = sample_df()
    first = create_partitions(df)
    second = create_partitions(df)
    assert ids(first.train) == ids(second.train)
    assert ids(first.validation) == ids(second.validation)
    assert ids(first.test) == ids(second.test)


def test_split_respects_requested_sizes():
    df = sample_df(n=2000)
    split = create_partitions(df)
    total = len(df)
    assert abs(len(split.train) / total - 0.70) <= 0.01
    assert abs(len(split.validation) / total - 0.15) <= 0.01
    assert abs(len(split.test) / total - 0.15) <= 0.01


def test_split_is_stratified_by_target():
    df = sample_df(n=2000)
    overall = (df["Churn"] == "Yes").mean()
    split = create_partitions(df)
    for part in (split.train, split.validation, split.test):
        rate = (part["Churn"] == "Yes").mean()
        assert abs(rate - overall) <= 0.02


def test_split_with_different_seed_differs():
    df = sample_df()
    first = create_partitions(df, seed=1)
    second = create_partitions(df, seed=2)
    assert ids(first.test) != ids(second.test)


def test_build_log_traces_partitions():
    parts = create_partitions(sample_df(n=100))
    log = build_log(
        "sha-in", {"train": "sha-tr", "validation": "sha-va", "test": "sha-te"}, parts
    )
    assert log["task"] == "T-11"
    assert log["input_sha256"] == "sha-in"
    assert log["rows"]["train"] == len(parts.train)
    assert log["rows"]["validation"] == len(parts.validation)
    assert log["rows"]["test"] == len(parts.test)
    assert log["output_sha256"]["train"] == "sha-tr"
    assert log["fractions"] == {
        "train": split.TRAIN_FRAC,
        "validation": split.VAL_FRAC,
        "test": split.TEST_FRAC,
    }


def test_main_writes_partitions_and_log(tmp_path, monkeypatch):
    df = sample_df(n=100)
    in_file = tmp_path / "churn_features.csv"
    df.to_csv(in_file, index=False)
    splits_dir = tmp_path / "splits"
    monkeypatch.setattr(split, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(split, "INPUT_FILE", in_file)
    monkeypatch.setattr(split, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(split, "LOG_FILE", splits_dir / "partition_log.json")
    monkeypatch.setattr(split, "TRAIN_FRAC", 0.70)
    monkeypatch.setattr(split, "VAL_FRAC", 0.15)
    monkeypatch.setattr(split, "TEST_FRAC", 0.15)
    assert split.main() == 0
    for name in ("train", "validation", "test"):
        part = pd.read_csv(splits_dir / f"{name}.csv")
        assert list(part.columns) == list(COLUMNS) + list(FEATURE_COLUMNS)
    log = json.loads((splits_dir / "partition_log.json").read_text(encoding="utf-8"))
    assert log["task"] == "T-11"
    assert log["rows"]["train"] + log["rows"]["validation"] + log["rows"]["test"] == 100


def test_main_rejects_bad_schema(tmp_path, monkeypatch):
    in_file = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(in_file, index=False)
    monkeypatch.setattr(split, "INPUT_FILE", in_file)
    assert split.main() == 1
