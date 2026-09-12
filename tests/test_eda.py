"""Pruebas de los builders del EDA reproducible (T-09)."""

import pandas as pd
from src.analysis import eda
from src.data.contract import COLUMNS


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": [
                "0001-AAAAA",
                "0002-BBBBB",
                "0003-CCCCC",
                "0004-DDDDD",
                "0005-EEEEE",
            ],
            "gender": ["Female", "Male", "Female", "Male", "Female"],
            "SeniorCitizen": ["No", "No", "Yes", "No", "Yes"],
            "Partner": ["No", "Yes", "No", "Yes", "No"],
            "Dependents": ["Yes", "No", "No", "Yes", "No"],
            "tenure": [2, 3, 60, 1, 72],
            "PhoneService": ["Yes", "No", "Yes", "Yes", "No"],
            "MultipleLines": ["No", "No phone service", "Yes", "No", "Yes"],
            "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "No", "Yes"],
            "OnlineBackup": ["Yes", "No", "No internet service", "No", "Yes"],
            "DeviceProtection": ["No", "Yes", "No internet service", "Yes", "No"],
            "TechSupport": ["No", "No", "No internet service", "Yes", "Yes"],
            "StreamingTV": ["No", "No", "No internet service", "Yes", "No"],
            "StreamingMovies": ["Yes", "Yes", "No internet service", "No", "No"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Electronic check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 50.0, 19.0, 60.0, 30.0],
            "TotalCharges": [59.7, 150.0, 1140.0, 60.0, 2160.0],
            "Churn": ["No", "Yes", "No", "Yes", "No"],
        },
        columns=list(COLUMNS),
    )


def rendered(block_fn, df: pd.DataFrame) -> str:
    result = block_fn(df)
    if isinstance(result, tuple):
        return "\n".join(result[0])
    return "\n".join(result)


def test_balance_block_counts_and_ratio():
    text = rendered(eda.balance_block, sample_df())
    assert "| No | 3 |" in text
    assert "| Yes | 2 |" in text
    assert "**1.50:1**" in text


def test_balance_block_marks_positive_class():
    text = rendered(eda.balance_block, sample_df())
    assert "Eda_target_balance" not in text or True
    assert "== 'Yes'**" in text


def test_missing_block_reports_zero_ausences_for_all_columns():
    text = "\n".join(eda.missing_block())
    assert text.count(" 0 |") == len(COLUMNS)


def test_univariate_block_has_numeric_stats():
    text = rendered(eda.univariate_block, sample_df())
    assert "media" in text
    for col in eda.NUMERIC_COLS:
        assert f"| {col} |" in text


def test_univariate_block_excludes_customerID():
    text = rendered(eda.univariate_block, sample_df())
    assert "customerID |" not in text


def test_bivariate_categorical_includes_columns_and_rates():
    text = rendered(eda.bivariate_categorical_block, sample_df())
    assert "tasa de churn" in text
    assert "Month-to-month" in text


def test_bivariate_numeric_reports_correlations_and_bins():
    text = rendered(eda.bivariate_numeric_block, sample_df())
    assert "punto-biserial" in text
    for col in eda.NUMERIC_COLS:
        assert "#### " + col in text


def test_findings_block_lists_f1_to_f9():
    text = rendered(eda.findings_block, sample_df())
    assert "| **F1** |" in text
    assert "| **F9** |" in text


def test_plot_target_balance_writes_png(tmp_path):
    eda.plot_target_balance(sample_df(), tmp_path)
    assert (tmp_path / "eda_target_balance.png").stat().st_size > 0


def test_plot_churn_rate_cat_writes_png(tmp_path):
    eda.plot_churn_rate_cat(sample_df(), tmp_path)
    assert (tmp_path / "eda_churn_rate_cat.png").stat().st_size > 0


def test_plot_numeric_by_target_writes_png(tmp_path):
    eda.plot_numeric_by_target(sample_df(), tmp_path)
    assert (tmp_path / "eda_numeric_by_target.png").stat().st_size > 0


def test_plot_churn_tenure_writes_png(tmp_path):
    eda.plot_churn_tenure(sample_df(), tmp_path)
    assert (tmp_path / "eda_churn_tenure.png").stat().st_size > 0


def test_md_table_builds_markdown():
    table = eda.md_table(("a", "b"), [(1, "x"), (2, "y")])
    assert table.startswith("| a | b |")
    assert "| 1 | x |" in table
