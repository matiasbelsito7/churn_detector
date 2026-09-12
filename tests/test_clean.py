"""Pruebas unitarias de la limpieza y tratamiento de faltantes (T-07/T-08)."""

import pandas as pd
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
