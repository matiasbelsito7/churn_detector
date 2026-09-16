"""Pruebas de explicabilidad del modelo con SHAP (T-28)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.modeling.explain import (
    DEFAULT_FIGURES,
    aggregate_shap,
    build_report,
    feature_importance,
    generate_shap,
    map_column_to_original,
    write_artifacts,
    write_figures,
)
from src.modeling.preprocessing import (
    FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    build_preprocessing_pipeline,
)


def toy_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {"tenure": rng.randint(0, 73, n).astype(float)}
    data["MonthlyCharges"] = rng.uniform(18.0, 120.0, n)
    data["TotalCharges"] = rng.uniform(0.0, 8000.0, n)
    data["addon_missing_count"] = rng.randint(0, 5, n).astype(float)
    data["avg_monthly_charge_hist"] = rng.uniform(0.0, 160.0, n)
    categoric = [name for name in FEATURE_NAMES if name not in NUMERIC_FEATURE_NAMES]
    for name in categoric:
        data[name] = "No"
    data["gender"] = "Female"
    data["InternetService"] = "DSL"
    data["PaymentMethod"] = "Electronic check"
    data["Contract"] = "Month-to-month"
    return pd.DataFrame(data)[list(FEATURE_NAMES)]


def toy_pipeline(df: pd.DataFrame) -> Pipeline:
    y = pd.Series(
        np.where(np.arange(len(df)) % 2 == 0, "Yes", "No"),
        name="Churn",
    )
    preprocessing = build_preprocessing_pipeline()
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(df, y)
    return pipeline


def test_map_column_to_original_resolves_numeric_and_onehot():
    assert map_column_to_original("tenure") == "tenure"
    assert map_column_to_original("gender_Female") == "gender"
    assert map_column_to_original("PaymentMethod_Electronic check") == "PaymentMethod"
    assert map_column_to_original("tenure_bin_0-5") == "tenure_bin"


def test_map_column_to_original_raises_on_unknown():
    with pytest.raises(ValueError, match="desconocida"):
        map_column_to_original("Fantasma")


def test_aggregate_shap_sums_onehot_columns_per_original_feature():
    names = [
        "tenure",
        "gender_Female",
        "gender_Male",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
    ]
    shap_df = pd.DataFrame([[1.0, 10.0, -4.0, 3.0, 1.0]], columns=names)
    agg = aggregate_shap(shap_df)
    assert list(agg.columns) == ["tenure", "gender", "PaymentMethod"]
    assert agg.loc[0, "tenure"] == pytest.approx(1.0)
    assert agg.loc[0, "gender"] == pytest.approx(6.0)
    assert agg.loc[0, "PaymentMethod"] == pytest.approx(4.0)


def test_generate_shap_aggregates_to_original_features():
    df = toy_frame(80)
    pipeline = toy_pipeline(df)
    result = generate_shap(pipeline, df.head(20))
    assert list(result.shap_df.columns) == list(FEATURE_NAMES)
    assert result.shap_df.shape == (20, len(FEATURE_NAMES))
    assert np.isfinite(result.shap_df.to_numpy()).all()
    assert np.isfinite(result.base_value)


def test_feature_importance_orders_by_abs_mean_desc():
    shap_df = pd.DataFrame(
        [[1.0, -0.5, 0.1], [-1.0, 0.5, 0.2]],
        columns=["f1", "f2", "f3"],
    )
    importance = feature_importance(shap_df)
    assert list(importance.index) == ["f1", "f2", "f3"]
    assert float(importance["f1"]) == pytest.approx(1.0)


def test_write_figures_creates_graphics(tmp_path):
    df = toy_frame(30)
    pipeline = toy_pipeline(df)
    result = generate_shap(pipeline, df)
    write_figures(result.shap_df, df, tmp_path)
    top_names = list(feature_importance(result.shap_df).index[:3])
    for name in (
        *DEFAULT_FIGURES,
        *(f"shap_dependence_{feature}.png" for feature in top_names),
    ):
        assert (tmp_path / name).is_file()
        assert (tmp_path / name).stat().st_size > 0


def test_write_artifacts_writes_report_json_and_figures(tmp_path):
    df = toy_frame(30)
    pipeline = toy_pipeline(df)
    result = generate_shap(pipeline, df)
    md_path = tmp_path / "explainability.md"
    json_path = tmp_path / "explainability.json"
    figures_dir = tmp_path / "figures"
    write_artifacts(result, df, md_path, json_path, figures_dir, "abc123")

    assert md_path.is_file()
    text = md_path.read_text(encoding="utf-8")
    assert "# Explicabilidad del modelo (T-28)" in text
    assert "base value" in text.lower() or "Base value" in text
    assert "abc123" in text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["task"] == "T-28"
    assert payload["model"] == "churn-logistic-regression"
    assert payload["base_value"] == pytest.approx(result.base_value)
    assert len(payload["feature_importance"]) == len(FEATURE_NAMES)
    top = payload["feature_importance"][0]
    assert set(top) == {"feature", "mean_abs_shap", "share_pct", "mean_shap"}
    assert (figures_dir / "shap_importance_bar.png").is_file()


def test_build_report_includes_top_features_and_figures_paths():
    df = toy_frame(40)
    pipeline = toy_pipeline(df)
    result = generate_shap(pipeline, df)
    importance = feature_importance(result.shap_df)
    report = build_report(result, df, importance, "abc123", "logistic-regression")
    top = importance.index[0]
    assert "# Explicabilidad del modelo (T-28)" in report
    assert "`churn-logistic-regression@production`" in report
    assert f"**{top}" in report
    assert "shap_importance_bar.png" in report
