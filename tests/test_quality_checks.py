"""Pruebas unitarias de los checks de calidad de datos (T-06)."""

import pandas as pd
from src.data.contract import COLUMNS
from src.data.quality_checks import run_all


def valid_df() -> pd.DataFrame:
    data = {
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
    }
    return pd.DataFrame(data, columns=list(COLUMNS))


def by_rule(df: pd.DataFrame) -> dict[str, bool]:
    return {r.rule_id: r.passed for r in run_all(df)}


def test_all_checks_pass_on_valid_dataset():
    assert all(by_rule(valid_df()).values())


def test_schema_fails_when_column_order_changes():
    df = valid_df()
    df = df[df.columns[::-1]]
    assert not by_rule(df)["R1"]


def test_domains_fail_on_out_of_domain_value():
    df = valid_df()
    df.loc[0, "gender"] = "Other"
    assert not by_rule(df)["R2"]


def test_domains_fail_on_missing_category_value():
    df = valid_df()
    df.loc[0, "gender"] = None
    assert not by_rule(df)["R2"]


def test_numeric_fails_on_unparsable_value():
    df = valid_df()
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(object)
    df.loc[0, "MonthlyCharges"] = "abc"
    assert not by_rule(df)["R3"]


def test_numeric_fails_on_out_of_range_value():
    df = valid_df()
    df.loc[0, "MonthlyCharges"] = 200.0
    assert not by_rule(df)["R3"]


def test_unique_ids_fail_on_duplicate():
    df = valid_df()
    df.loc[1, "customerID"] = "0001-AAAAA"
    assert not by_rule(df)["R4"]


def test_unique_ids_fail_on_bad_format():
    df = valid_df()
    df.loc[0, "customerID"] = "malformat"
    assert not by_rule(df)["R4"]


def test_duplicates_fail_on_repeated_row():
    df = valid_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not by_rule(df)["R5"]


def test_coherence_fails_on_phone_and_multiples():
    df = valid_df()
    df.loc[0, "MultipleLines"] = "No phone service"
    assert not by_rule(df)["R6"]


def test_coherence_fails_on_internet_dependent_field():
    df = valid_df()
    df.loc[0, "OnlineSecurity"] = "No internet service"
    assert not by_rule(df)["R6"]


def test_missing_totalcharges_fails_with_tenure_gt_zero():
    df = valid_df()
    df.loc[1, "TotalCharges"] = ""
    assert not by_rule(df)["R7"]


def test_missing_totalcharges_passes_with_tenure_zero():
    assert by_rule(valid_df())["R7"]
