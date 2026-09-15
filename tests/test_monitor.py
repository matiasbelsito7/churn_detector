"""Pruebas del monitoreo del sistema (T-22).

Cubren las funciones puras (PSI, shift categórico, clasificación de umbrales,
desempeño) y el reporte; no dependen de la base de datos ni de la red.
"""

import json

import numpy as np
import pandas as pd
import pytest
from src.data.contract import TARGET
from src.modeling.baseline import evaluate_metrics
from src.serving import monitor

SAMPLE_CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "tenure_bin": ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59", "60-72"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "is_new_client": ["Yes", "No"],
    "is_month_to_month": ["Yes", "No"],
    "is_electronic_check": ["Yes", "No"],
    "is_fiber_optic": ["Yes", "No"],
}


def sample_features(n: int = 100, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data: dict[str, object] = {
        "customerID": [f"ID-{i:04d}" for i in range(n)],
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": np.round(rng.uniform(20, 120, n), 2),
        "TotalCharges": np.round(rng.uniform(0, 8000, n), 2),
        "addon_missing_count": rng.integers(0, 4, n),
        "avg_monthly_charge_hist": np.round(rng.uniform(10, 130, n), 2),
        TARGET: rng.choice(["Yes", "No"], n),
    }
    for col in SAMPLE_CATEGORIES:
        data[col] = rng.choice(SAMPLE_CATEGORIES[col], n)
    df = pd.DataFrame(data)

    # Desplazamiento aplicado sobre las features numéricas para simular drift.
    for col in (
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "addon_missing_count",
        "avg_monthly_charge_hist",
    ):
        df[col] = df[col] + shift
    return df


def sample_predictions(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    prob = rng.uniform(0.01, 0.95, n)
    return pd.DataFrame(
        {
            "customerID": [f"ID-{i:04d}" for i in range(n)],
            "churn_prob": prob,
            "churn_class": np.where(prob >= 0.5, "Yes", "No"),
        }
    )


def test_classify_thresholds():
    assert monitor.classify(0.0, 0.1, 0.25) == "ok"
    assert monitor.classify(0.15, 0.1, 0.25) == "warn"
    assert monitor.classify(0.3, 0.1, 0.25) == "alert"
    assert monitor.classify(0.1, 0.1, 0.25) == "warn"
    assert monitor.classify(0.25, 0.1, 0.25) == "alert"


def test_psi_constant_distributions_gives_zero():
    ref = pd.Series(np.ones(50) * 5.0)
    cur = pd.Series(np.ones(50) * 5.0)
    assert monitor.psi(ref, cur) == pytest.approx(0.0)


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 500))
    cur = pd.Series(rng.normal(3, 1, 500))
    assert monitor.psi(ref, cur) > monitor.PSI_ALERT


def test_categorical_max_shift():
    ref = pd.Series(["a", "a", "a", "b"])
    same = pd.Series(["a", "a", "a", "b"])
    different = pd.Series(["b", "b", "b", "a"])
    assert monitor.categorical_max_shift(ref, same) == pytest.approx(0.0)
    assert monitor.categorical_max_shift(ref, different) == pytest.approx(
        abs(0.75 - 0.25)
    )


def test_feature_checks_classification():
    ref = sample_features(200)
    cur_far = sample_features(200, shift=30)
    checks = monitor.feature_checks(ref, cur_far)
    numeric = [c for c in checks if c["metric"] == "psi"]
    categorical = [c for c in checks if c["metric"] == "categorical_max_shift"]
    assert len(numeric) == len(monitor.NUMERIC_FEATURE_NAMES)
    assert len(categorical) == len(monitor.CATEGORIC_FEATURE_NAMES)
    by_name = {c["name"]: c for c in numeric}
    assert by_name["tenure"]["status"] == "alert"
    assert by_name["addon_missing_count"]["status"] == "alert"
    assert all(c["status"] == "ok" for c in categorical)


def test_prediction_checks_detect_rate_shift():
    rng = np.random.default_rng(3)
    ref = sample_predictions(200)
    current = sample_predictions(200)
    # Fuerza un aumento de la tasa de churn predicha.
    current["churn_class"] = np.where(rng.random(200) < 0.9, "Yes", "No")
    checks = monitor.prediction_checks(ref, current)
    rate = next(c for c in checks if c["name"] == "churn_rate")
    assert rate["status"] == "alert"


def test_target_checks_only_when_target_present():
    ref = pd.DataFrame({TARGET: ["Yes", "No", "Yes"]})
    cur = pd.DataFrame({TARGET: ["Yes", "No", "Yes"]})

    checks = monitor.target_checks(ref, cur)
    assert checks[0]["value"] == pytest.approx(0.0)
    assert checks[0]["status"] == "ok"

    cur_drift = pd.DataFrame({TARGET: ["Yes", "Yes", "Yes"]})
    checks = monitor.target_checks(ref, cur_drift)
    assert checks[0]["value"] > 0
    assert checks[0]["status"] == "warn" or checks[0]["status"] == "alert"


def test_performance_checks_reference_metrics():
    # Solo se computan cuando el batch corriente tiene target.
    df = sample_features(100)
    pred = sample_predictions(100)
    pred["churn_prob"] = np.where(df[TARGET] == "Yes", 0.9, 0.1)
    reference = {"recall_pos": 0.9, "auc_pr": 0.9}

    checks = monitor.performance_checks(df, pred, reference)
    assert len(checks) == 2
    assert all(c["status"] == "ok" for c in checks)

    # Batch sin target: no hay chequeos de desempeño.
    no_target = df.drop(columns=[TARGET])
    assert monitor.performance_checks(no_target, pred, reference) == []


def test_evaluate_metrics_is_reused_correctly():
    # evaluate_metrics (T-11) es la fuente de métricas del monitoreo.
    y = pd.Series(["Yes", "No", "Yes", "Yes", "No"])
    prob = np.array([0.8, 0.2, 0.6, 0.9, 0.1])
    metrics = evaluate_metrics(y, prob)
    assert set(("recall_pos", "auc_pr", "f1_pos")).issubset(metrics)
    assert 0 <= metrics["auc_pr"] <= 1


def test_overall_status_takes_worst():
    checks = [
        {"status": "ok"},
        {"status": "warn"},
        {"status": "ok"},
        {"status": "alert"},
    ]
    assert monitor.overall_status(checks) == "alert"
    assert monitor.overall_status(checks[:2]) == "warn"
    assert monitor.overall_status([{"status": "ok"}]) == "ok"


def test_zero_drift_reference_equals_current(monkeypatch, tmp_path):
    """Con referencia == corriente, todos los chequeos deben ser ok o vacíos."""
    df = sample_features(200)
    pred = sample_predictions(200)

    ref_path = tmp_path / "ref.csv"
    pred_path = tmp_path / "pred.csv"
    ref_path.write_text(df.to_csv(index=False), encoding="utf-8")
    pred_path.write_text(pred.to_csv(index=False), encoding="utf-8")

    # Referencia = corriente via mismo archivo de features (sin dividir por train).
    monkeypatch.setattr(monitor, "MONITOR_REFERENCE_FEATURES", str(ref_path))
    monkeypatch.setattr(monitor, "MONITOR_CURRENT_FEATURES", str(ref_path))
    monkeypatch.setattr(monitor, "MONITOR_REFERENCE_PREDICTIONS", str(pred_path))
    monkeypatch.setattr(monitor, "MONITOR_CURRENT_PREDICTIONS", str(pred_path))

    ref_features = pd.read_csv(ref_path)
    cur_features = pd.read_csv(ref_path)
    ref_pred = pd.read_csv(pred_path)
    cur_pred = pd.read_csv(pred_path)

    merged = cur_features[["customerID", TARGET]].merge(
        cur_pred[["customerID", "churn_prob"]], on="customerID"
    )
    perf_metrics = evaluate_metrics(
        merged[TARGET], merged["churn_prob"].to_numpy("float")
    )
    reference = {
        "recall_pos": perf_metrics["recall_pos"],
        "auc_pr": perf_metrics["auc_pr"],
    }

    checks = {
        "features": monitor.feature_checks(ref_features, cur_features),
        "predictions": monitor.prediction_checks(ref_pred, cur_pred),
        "target": monitor.target_checks(ref_features, cur_features),
        "performance": monitor.performance_checks(cur_features, cur_pred, reference),
    }
    all_checks = [c for group in checks.values() for c in group]
    assert monitor.overall_status(all_checks) == "ok"


def test_build_report_reflects_status(monkeypatch, tmp_path):
    df = sample_features(10)
    pred = sample_predictions(10)
    ref_path = tmp_path / "ref.csv"
    pred_path = tmp_path / "pred.csv"
    ref_path.write_text(df.to_csv(index=False), encoding="utf-8")
    pred_path.write_text(pred.to_csv(index=False), encoding="utf-8")

    checks = {
        "features": [],  # simplicidad: sin chequeos de features
        "predictions": monitor.prediction_checks(pred, pred),
        "target": monitor.target_checks(df, df),
        "performance": [],
    }
    all_checks = [c for group in checks.values() for c in group]
    status = monitor.overall_status(all_checks)
    provenance = {"features": {"path": "x", "rows": 10, "sha256": "a" * 64}}
    report = monitor.build_report(status, checks, provenance)

    assert "# Monitoreo del sistema de predicción de churn (T-22)" in report
    assert "**OK**" in report
    assert "predicciones" in report
    assert "Trazabilidad" in report


def test_main_writes_reports(monkeypatch, tmp_path):
    """`main()` genera los reportes a partir de artefactos versionados."""
    df = sample_features(20)
    pred = sample_predictions(20)
    merged = df[["customerID", TARGET]].merge(
        pred[["customerID", "churn_prob"]], on="customerID"
    )
    perf_metrics = evaluate_metrics(
        merged[TARGET], merged["churn_prob"].to_numpy("float")
    )
    reference_metrics = {
        "recall_pos": float(perf_metrics["recall_pos"]),
        "auc_pr": float(perf_metrics["auc_pr"]),
    }
    ref_features = tmp_path / "ref.csv"
    cur_features = tmp_path / "cur.csv"
    ref_pred = tmp_path / "pred.csv"
    ref_metrics = tmp_path / "selection.json"
    ref_features.write_text(df.to_csv(index=False), encoding="utf-8")
    cur_features.write_text(df.to_csv(index=False), encoding="utf-8")
    ref_pred.write_text(pred.to_csv(index=False), encoding="utf-8")
    ref_metrics.write_text(
        json.dumps({"test_metrics": reference_metrics}),
        encoding="utf-8",
    )

    monkeypatch.setattr(monitor, "MONITOR_REFERENCE_FEATURES", str(ref_features))
    monkeypatch.setattr(monitor, "MONITOR_CURRENT_FEATURES", str(cur_features))
    monkeypatch.setattr(monitor, "MONITOR_REFERENCE_PREDICTIONS", str(ref_pred))
    monkeypatch.setattr(monitor, "MONITOR_CURRENT_PREDICTIONS", str(ref_pred))
    monkeypatch.setattr(monitor, "MONITOR_REFERENCE_METRICS", str(ref_metrics))

    report_json = tmp_path / "monitoring.json"
    report_md = tmp_path / "monitoring.md"
    monkeypatch.setattr(monitor, "REPORT_JSON", report_json)
    monkeypatch.setattr(monitor, "REPORT_MD", report_md)

    assert monitor.main() == 0

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["task"] == "T-22"
    assert payload["status"] == "ok"
    assert set(payload["checks"]) == {
        "features",
        "predictions",
        "target",
        "performance",
    }
    md = report_md.read_text(encoding="utf-8")
    assert "# Monitoreo del sistema de predicción de churn (T-22)" in md
