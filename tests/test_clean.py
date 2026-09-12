"""Pruebas unitarias de la limpieza de datos (T-07)."""

import numpy as np
import pandas as pd
import pytest
from src.data.clean import clean_raw
from src.data.contract import COLUMNS


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


def test_totalcharges_blank_becomes_missing_and_float():
    df_clean = clean_raw(sample_raw())
    assert pd.api.types.is_float_dtype(
        df_clean["TotalCharges"]
    ), "TotalCharges debe ser float."
    assert np.isnan(
        df_clean.loc[0, "TotalCharges"]
    ), "Celda de solo espacios debe quedar ausente."
    assert df_clean.loc[1, "TotalCharges"] == 250.0


def test_senior_citizen_normalized_to_yes_no():
    df_clean = clean_raw(sample_raw())
    assert df_clean["SeniorCitizen"].tolist() == ["No", "Yes"]


def test_clean_output_is_deterministic():
    first = clean_raw(sample_raw())
    second = clean_raw(sample_raw())
    pd.testing.assert_frame_equal(first, second)


def test_clean_accepts_whitespace_only_totalcharges():
    df = sample_raw()
    df.loc[1, "TotalCharges"] = "   "
    df_clean = clean_raw(df)
    assert np.isnan(df_clean.loc[1, "TotalCharges"])


def test_clean_rejects_nonnumeric_totalcharges():
    df = sample_raw()
    df.loc[1, "TotalCharges"] = "abc"
    df_clean = clean_raw(df)
    assert np.isnan(df_clean.loc[1, "TotalCharges"])


def test_clean_missing_reader_direct():
    pytest.importorskip("pandas")
    # smoke: clean_raw carga bien con el esquema del contrato
    assert "Churn" in clean_raw(sample_raw()).columns
