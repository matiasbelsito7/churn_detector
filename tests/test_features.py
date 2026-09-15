"""Pruebas del feature engineering derivado del EDA (T-10)."""

import json

import pandas as pd
import pytest
from src.analysis import features
from src.analysis.features import (
    FEATURE_COLUMNS,
    build_log,
    engineer_features,
)
from src.data.contract import COLUMNS
from src.seeds import RANDOM_SEED


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001-AAAAA", "0002-BBBBB", "0003-CCCCC", "0004-DDDDD"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": ["No", "No", "Yes", "No"],
            "Partner": ["No", "Yes", "No", "Yes"],
            "Dependents": ["Yes", "No", "No", "Yes"],
            "tenure": [2, 3, 60, 72],
            "PhoneService": ["Yes", "No", "Yes", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes", "No"],
            "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic"],
            "OnlineSecurity": ["No", "No", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No", "No internet service", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "No"],
            "TechSupport": ["No", "No", "No internet service", "Yes"],
            "StreamingTV": ["No", "No", "No internet service", "Yes"],
            "StreamingMovies": ["Yes", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [60.0, 29.85, 19.0, 118.75],
            "TotalCharges": [120.0, 89.55, 1140.0, 5904.0],
            "Churn": ["No", "Yes", "No", "Yes"],
        },
        columns=list(COLUMNS),
    )


def test_adds_features_in_fixed_order():
    out = engineer_features(make_df())
    assert list(out.columns) == list(COLUMNS) + list(FEATURE_COLUMNS)


def test_does_not_modify_input():
    df = make_df()
    engineer_features(df)
    pd.testing.assert_frame_equal(df, make_df())


def test_output_is_deterministic():
    first = engineer_features(make_df())
    second = engineer_features(make_df())
    pd.testing.assert_frame_equal(first, second)


def test_tenure_bin_edges():
    out = engineer_features(make_df())
    assert out["tenure_bin"].tolist() == ["0-5", "0-5", "60-72", "60-72"]


def test_is_new_client_threshold():
    out = engineer_features(make_df())
    assert out["is_new_client"].tolist() == ["Yes", "Yes", "No", "No"]


def test_is_month_to_month():
    out = engineer_features(make_df())
    assert out["is_month_to_month"].tolist() == ["Yes", "No", "No", "Yes"]


def test_is_electronic_check():
    out = engineer_features(make_df())
    assert out["is_electronic_check"].tolist() == ["Yes", "No", "No", "No"]


def test_is_fiber_optic():
    out = engineer_features(make_df())
    assert out["is_fiber_optic"].tolist() == ["Yes", "No", "No", "Yes"]


def test_addon_missing_count_partial():
    out = engineer_features(make_df())
    assert out["addon_missing_count"].tolist() == [3, 3, 0, 2]


def test_avg_monthly_charge_hist():
    out = engineer_features(make_df())
    expected = [60.0, 29.85, 19.0, 82.0]
    for got, want in zip(
        out["avg_monthly_charge_hist"].tolist(), expected, strict=True
    ):
        assert got == pytest.approx(want, abs=1e-9)


def test_avg_monthly_charge_hist_zero_for_new_client():
    df = make_df()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = 0.0
    out = engineer_features(df)
    assert out.loc[0, "avg_monthly_charge_hist"] == 0.0


def test_same_schema_between_full_and_subset():
    full = engineer_features(make_df())
    subset = engineer_features(make_df().head(2))
    assert list(subset.columns) == list(full.columns)
    assert list(subset.dtypes) == list(full.dtypes)


def test_binary_indicator_domain():
    out = engineer_features(make_df())
    for col in (
        "is_new_client",
        "is_month_to_month",
        "is_electronic_check",
        "is_fiber_optic",
    ):
        assert set(out[col].unique()) <= {"No", "Yes"}


def test_addon_missing_count_range():
    out = engineer_features(make_df())
    assert out["addon_missing_count"].between(0, 4).all()


def test_avg_monthly_charge_hist_non_negative():
    out = engineer_features(make_df())
    assert (out["avg_monthly_charge_hist"] >= 0).all()


def test_output_bool_free_columns():
    out = engineer_features(make_df())
    assert not out.apply(lambda s: pd.api.types.is_bool_dtype(s)).any()


def test_build_log_schema():
    log = build_log("aaa", "bbb", engineer_features(make_df()))
    assert log["task"] == "T-10"
    assert log["input_sha256"] == "aaa"
    assert log["output_sha256"] == "bbb"
    assert log["rows"] == 4
    assert log["features"] == list(FEATURE_COLUMNS)
    assert log["seed"] == RANDOM_SEED


def test_main_writes_features(tmp_path, monkeypatch):
    input_file = tmp_path / "churn_cleaned.csv"
    out_file = tmp_path / "churn_features.csv"
    log_file = tmp_path / "feature_log.json"
    make_df().to_csv(input_file, index=False)
    monkeypatch.setattr(features, "INPUT_FILE", input_file)
    monkeypatch.setattr(features, "OUTPUT_FILE", out_file)
    monkeypatch.setattr(features, "LOG_FILE", log_file)
    monkeypatch.setattr(features, "PROJECT_ROOT", tmp_path)
    assert features.main() == 0
    out = pd.read_csv(out_file)
    assert "tenure_bin" in out.columns
    assert "addon_missing_count" in out.columns
    log = json.loads(log_file.read_text(encoding="utf-8"))
    assert log["task"] == "T-10"


def test_main_rejects_bad_schema(tmp_path, monkeypatch):
    input_file = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(input_file, index=False)
    monkeypatch.setattr(features, "INPUT_FILE", input_file)
    assert features.main() == 1
