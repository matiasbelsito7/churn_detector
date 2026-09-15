"""Pruebas unitarias de la limpieza y tratamiento de faltantes (T-07/T-08)."""

import json

import pandas as pd
from src.data import clean
from src.data.clean import build_log, clean_raw, sha256_file
from src.data.contract import COLUMNS
from src.seeds import RANDOM_SEED


def sample_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001-AAAAA", "0002-BBBBB"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["No", "Yes"],
            "Dependents": ["Yes", "No"],
            "tenure": [0, 5],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["Fiber optic", "No"],
            "OnlineSecurity": ["No", "No internet service"],
            "OnlineBackup": ["Yes", "No internet service"],
            "DeviceProtection": ["No", "No internet service"],
            "TechSupport": ["No", "No internet service"],
            "StreamingTV": ["No", "No internet service"],
            "StreamingMovies": ["Yes", "No internet service"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 50.0],
            "TotalCharges": ["", 250.0],
            "Churn": ["No", "Yes"],
        },
        columns=list(COLUMNS),
    )


def test_clean_does_not_modify_input():
    df = sample_raw()
    clean_raw(df)
    pd.testing.assert_frame_equal(df, sample_raw())


def test_clean_preserves_rows_and_columns():
    df_clean = clean_raw(sample_raw())
    assert df_clean.shape == (2, len(COLUMNS))
    assert list(df_clean.columns) == list(COLUMNS)


def test_totalcharges_present_value_unchanged():
    df_clean = clean_raw(sample_raw())
    assert pd.api.types.is_float_dtype(
        df_clean["TotalCharges"]
    ), "TotalCharges debe ser float."
    assert df_clean.loc[1, "TotalCharges"] == 250.0


def test_totalcharges_missing_imputed_zero():
    df_clean = clean_raw(sample_raw())
    assert df_clean.loc[0, "TotalCharges"] == 0.0


def test_no_missing_values_after_cleaning():
    df_clean = clean_raw(sample_raw())
    assert df_clean.isna().sum().sum() == 0


def test_imputation_is_identical_to_inference_rule():
    for value in ("", "   "):
        df = sample_raw()
        df.loc[1, "TotalCharges"] = value
        df_clean = clean_raw(df)
        assert df_clean.loc[1, "TotalCharges"] == 0.0


def test_clean_rejects_nonnumeric_totalcharges():
    df = sample_raw()
    df.loc[1, "TotalCharges"] = "abc"
    df_clean = clean_raw(df)
    assert df_clean.loc[1, "TotalCharges"] == 0.0


def test_senior_citizen_normalized_to_yes_no():
    df_clean = clean_raw(sample_raw())
    assert df_clean["SeniorCitizen"].tolist() == ["No", "Yes"]


def test_clean_output_is_deterministic():
    first = clean_raw(sample_raw())
    second = clean_raw(sample_raw())
    pd.testing.assert_frame_equal(first, second)


def test_sha256_file_deterministic_and_sensitive(tmp_path):
    path = tmp_path / "archivo.txt"
    other = tmp_path / "archivo2.txt"
    path.write_text("contenido fijo", encoding="utf-8")
    other.write_text("contenido fijo X", encoding="utf-8")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64
    assert sha256_file(path) != sha256_file(other)


def test_build_log_traces_provenance():
    log = build_log("abc123", "def456", clean_raw(sample_raw()))
    assert log["task"] == "T-07"
    assert log["input_sha256"] == "abc123"
    assert log["output_sha256"] == "def456"
    assert log["seed"] == RANDOM_SEED
    assert log["rows"] == 2
    assert log["columns"] == len(COLUMNS)
    assert isinstance(log["steps"], list) and len(log["steps"]) == 4


def test_main_blocks_cleaning_when_quality_fails(tmp_path, monkeypatch, capsys):
    bad = sample_raw()
    bad.loc[0, "gender"] = "Other"
    raw_path = tmp_path / "raw.csv"
    bad.to_csv(raw_path, index=False)
    out_dir = tmp_path / "processed"
    monkeypatch.setattr(clean, "RAW_FILE", raw_path)
    monkeypatch.setattr(clean, "OUTPUT_FILE", out_dir / "churn_cleaned.csv")
    monkeypatch.setattr(clean, "LOG_FILE", out_dir / "processing_log.json")
    assert clean.main() == 1
    assert not (out_dir / "churn_cleaned.csv").exists()
    assert "bloquea la limpieza" in capsys.readouterr().out


def test_main_writes_artifacts_when_quality_passes(tmp_path, monkeypatch):
    raw_path = tmp_path / "data" / "raw" / "raw.csv"
    raw_path.parent.mkdir(parents=True)
    sample_raw().to_csv(raw_path, index=False)
    out_dir = tmp_path / "data" / "processed"
    output = out_dir / "churn_cleaned.csv"
    log_file = out_dir / "processing_log.json"
    monkeypatch.setattr(clean, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(clean, "RAW_FILE", raw_path)
    monkeypatch.setattr(clean, "OUTPUT_FILE", output)
    monkeypatch.setattr(clean, "LOG_FILE", log_file)
    assert clean.main() == 0
    df_clean = pd.read_csv(output)
    assert list(df_clean.columns) == list(COLUMNS)
    assert df_clean["SeniorCitizen"].isin(["No", "Yes"]).all()
    log = json.loads(log_file.read_text(encoding="utf-8"))
    assert log["task"] == "T-07"
    assert log["rows"] == 2
