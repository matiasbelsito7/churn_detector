"""Pruebas del pipeline de preprocessing sin data leakage (T-12)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from src.analysis.features import FEATURE_COLUMNS
from src.modeling.preprocessing import (
    CATEGORIC_FEATURE_NAMES,
    FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    build_preprocessing_pipeline,
    fit_preprocessing,
    transform_features,
)


def make_features_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {
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
        "tenure_bin": rng.choice(
            ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59", "60-72"], n
        ),
        "is_new_client": rng.choice(["No", "Yes"], n),
        "is_month_to_month": rng.choice(["No", "Yes"], n),
        "is_electronic_check": rng.choice(["No", "Yes"], n),
        "is_fiber_optic": rng.choice(["No", "Yes"], n),
        "addon_missing_count": rng.randint(0, 5, n),
        "avg_monthly_charge_hist": rng.uniform(0.0, 121.4, n),
    }
    df = pd.DataFrame(data)
    return df[list(FEATURE_NAMES)]


def evaluate_output(pipeline, frames: dict[str, pd.DataFrame]):
    out = {name: transform_features(pipeline, frame) for name, frame in frames.items()}
    ref = out["train"]
    assert ref["tenure"].notna().all()
    for name, frame_out in out.items():
        assert list(frame_out.columns) == list(ref.columns)
        assert frame_out.shape[0] == frames[name].shape[0]
        assert all(pd.api.types.is_float_dtype(dtype) for dtype in frame_out.dtypes)
    return out


def test_feature_names_exclude_id_and_target():
    assert "customerID" not in FEATURE_NAMES
    assert "Churn" not in FEATURE_NAMES
    assert set(FEATURE_COLUMNS) <= set(FEATURE_NAMES)
    numeric = {
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "addon_missing_count",
        "avg_monthly_charge_hist",
    }
    assert set(NUMERIC_FEATURE_NAMES) == numeric
    assert set(CATEGORIC_FEATURE_NAMES) == set(FEATURE_NAMES) - numeric


def test_pipeline_requires_fit_before_transform():
    df = make_features_df(50, seed=1)
    pipeline = build_preprocessing_pipeline()
    with pytest.raises(NotFittedError):
        transform_features(pipeline, df)


def test_transform_same_schema_train_val_test():
    train = make_features_df(2000, seed=1)
    val = make_features_df(400, seed=2)
    test = make_features_df(400, seed=3)
    pipeline = fit_preprocessing(train)
    out = evaluate_output(pipeline, {"train": train, "validation": val, "test": test})
    assert len(out["train"].columns) > len(NUMERIC_FEATURE_NAMES) + len(
        CATEGORIC_FEATURE_NAMES
    )


def test_identical_row_encoded_identically_from_any_partition():
    train = make_features_df(2000, seed=5)
    val = make_features_df(400, seed=6)
    pipeline = fit_preprocessing(train)
    probe = val.iloc[[3]].reset_index(drop=True)
    frame = pd.concat([probe, probe.copy()], ignore_index=True)
    out = transform_features(pipeline, frame)
    np.testing.assert_allclose(out.iloc[0].to_numpy(), out.iloc[1].to_numpy())
    same_val = transform_features(pipeline, probe)
    np.testing.assert_allclose(out.iloc[0].to_numpy(), same_val.iloc[0].to_numpy())


def test_numeric_scaling_uses_train_statistics():
    train = make_features_df(2000, seed=10)
    val = make_features_df(40, seed=11)
    val["tenure"] = 72
    pipeline = fit_preprocessing(train)
    scaler = pipeline.named_transformers_["num"].named_steps["scaler"]
    mean, scale = scaler.mean_[0], scaler.scale_[0]
    out = transform_features(pipeline, val)
    assert np.allclose(out["tenure"].to_numpy(), (72.0 - mean) / scale)


def test_numeric_imputation_follows_train_median():
    train = make_features_df(2000, seed=20)
    val = make_features_df(1, seed=21)
    val["TotalCharges"] = np.nan
    pipeline = fit_preprocessing(train)
    output = transform_features(pipeline, val)
    assert output["TotalCharges"].notna().all()
    imputer = pipeline.named_transformers_["num"].named_steps["imputer"]
    scaler = pipeline.named_transformers_["num"].named_steps["scaler"]
    idx = NUMERIC_FEATURE_NAMES.index("TotalCharges")
    median = imputer.statistics_[idx]
    expected = (median - scaler.mean_[idx]) / scaler.scale_[idx]
    assert output["TotalCharges"].iloc[0] == pytest.approx(expected)


def test_unknown_category_encoded_as_zeros():
    train = make_features_df(2000, seed=30)
    val = make_features_df(1, seed=31)
    val["Contract"] = "Ultra flexible"
    pipeline = fit_preprocessing(train)
    output = transform_features(pipeline, val)
    assert output.select_dtypes(include=[np.number]).shape[1] == output.shape[1]
    encoded_cols = [c for c in output.columns if c.startswith("Contract")]
    assert all(output[c].iloc[0] == 0.0 for c in encoded_cols)


def test_remainder_drop_ignores_extra_columns():
    train = make_features_df(200, seed=40)
    with_id = train.copy()
    with_id["customerID"] = "0001-AAAAA"
    pipeline = fit_preprocessing(train)
    out_clean = transform_features(pipeline, train)
    out_with_id = transform_features(pipeline, with_id)
    np.testing.assert_array_equal(out_clean.to_numpy(), out_with_id.to_numpy())


def test_transform_does_not_refit():
    train = make_features_df(2000, seed=50)
    val = make_features_df(200, seed=51)
    pipeline = fit_preprocessing(train)
    scaler = pipeline.named_transformers_["num"].named_steps["scaler"]
    mean_before = scaler.mean_.copy()
    transform_features(pipeline, val)
    np.testing.assert_array_equal(mean_before, scaler.mean_)


def test_deterministic_given_same_train():
    train = make_features_df(2000, seed=60)
    val = make_features_df(200, seed=61)
    first = transform_features(fit_preprocessing(train), val)
    second = transform_features(fit_preprocessing(train), val)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
