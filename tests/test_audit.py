"""Pruebas unitarias del data audit del raw (T-05)."""

import pandas as pd
from src.data.audit import audit, multivariate_checks, numeric_column
from src.data.contract import COLUMNS


def sample_raw() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "customerID": ["0001-AAAAA", "0002-BBBBB", "0003-CCCCC"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["No", "Yes", "No"],
            "Dependents": ["Yes", "No", "No"],
            "tenure": [0, 5, 60],
            "PhoneService": ["Yes", "No", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes"],
            "InternetService": ["Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["No", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "No internet service", "Yes"],
            "TechSupport": ["No", "No internet service", "No"],
            "StreamingTV": ["No", "No internet service", "Yes"],
            "StreamingMovies": ["Yes", "No internet service", "Yes"],
            "Contract": ["Month-to-month", "Two year", "One year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [60.0, 29.85, 19.0],
            "TotalCharges": ["", "149.25", "1140.0"],
            "Churn": ["No", "Yes", "No"],
        },
        columns=list(COLUMNS),
        dtype=object,
    )
    return df


def test_audit_reports_dimensions_and_sections():
    text = audit(sample_raw())
    assert "3 filas × 21 columnas" in text
    assert "## 1. Esquema: tipos reales vs esperados" in text
    assert "## 2. Valores nulos y ausencias" in text
    assert "## 3. Dominios y rangos" in text
    assert "## 4. Coherencia entre campos" in text
    assert "## 5. Resumen de hallazgos" in text
    assert "## 6. Desbalanceo del target" in text


def test_audit_flags_totalcharges_as_text_with_unparsable():
    text = audit(sample_raw())
    assert "`TotalCharges` se almacena como texto" in text
    assert "1 celdas no numéricas" in text


def test_audit_flags_seniorcitizen_mixed_coding():
    text = audit(sample_raw())
    assert "`SeniorCitizen` usa codificación numérica" in text


def test_audit_reports_target_balance():
    text = audit(sample_raw())
    assert "- `No`: 2 (66.67%)" in text
    assert "- `Yes`: 1 (33.33%)" in text
    assert "- Clase de interés (positiva): `Yes` (1)." in text


def test_audit_does_not_modify_input():
    df = sample_raw()
    audit(df)
    pd.testing.assert_frame_equal(df, sample_raw())


def test_multivariate_checks_detects_phone_incoherence():
    df = sample_raw()
    df.loc[2, "PhoneService"] = "No"
    df.loc[2, "MultipleLines"] = "Yes"
    findings = multivariate_checks(df)
    assert any("MultipleLines/PhoneService" in f for f in findings)


def test_multivariate_checks_detects_internet_incoherence():
    df = sample_raw()
    df.loc[2, "InternetService"] = "No"
    df.loc[2, "OnlineSecurity"] = "Yes"
    findings = multivariate_checks(df)
    assert any("OnlineSecurity/InternetService" in f for f in findings)


def test_multivariate_checks_detects_totalcharges_missing_with_tenure_gt_zero():
    df = sample_raw()
    df.loc[1, "TotalCharges"] = None
    findings = multivariate_checks(df)
    assert any("ausente" in f and "tenure > 0" in f for f in findings)


def test_multivariate_checks_reports_no_incoherences_on_clean():
    findings = multivariate_checks(sample_raw())
    assert findings == ["- Sin incoherencias entre campos detectadas."]


def test_numeric_column_coerces_text_to_float():
    series = pd.Series(["1.5", "2", ""])
    out = numeric_column(series)
    assert out.tolist()[:2] == [1.5, 2.0]
    assert out.isna().tolist()[2]
