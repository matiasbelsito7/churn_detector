"""Pruebas del módulo de predicción (T-15)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.serving.predict import (
    MODEL_OUTPUT_COLUMNS,
    predict_clients,
)

FEATURE_COLS = [f"f{i}" for i in range(5)]


def model_for_training() -> tuple[Pipeline, pd.DataFrame]:
    rng = np.random.RandomState(0)
    X = rng.randn(120, 5)
    y = (rng.rand(120) > 0.5).astype(int).astype(str)
    y = np.where(y == "1", "Yes", "No")
    pipeline = Pipeline([("model", LogisticRegression(max_iter=500))])
    pipeline.fit(pd.DataFrame(X, columns=FEATURE_COLS), y)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df.insert(0, "customerID", np.arange(len(X)).astype(str))
    return pipeline, df


def test_predict_clients_output_columns_and_types():
    model, df = model_for_training()
    out = predict_clients(model, df, feature_names=FEATURE_COLS)
    assert list(out.columns) == list(MODEL_OUTPUT_COLUMNS)
    assert len(out) == len(df)
    assert out["customerID"].tolist() == df["customerID"].tolist()
    assert out["churn_prob"].between(0, 1).all()
    assert set(out["churn_class"].unique()).issubset({"Yes", "No"})


def test_predict_clients_deterministic():
    model, df = model_for_training()
    first = predict_clients(model, df, feature_names=FEATURE_COLS)
    second = predict_clients(model, df, feature_names=FEATURE_COLS)
    pd.testing.assert_frame_equal(first, second)


def test_predict_clients_threshold_zero_one():
    model, df = model_for_training()
    out = predict_clients(model, df, threshold=0.0, feature_names=FEATURE_COLS)
    assert (out["churn_class"] == "Yes").all()
    out = predict_clients(model, df, threshold=1.0, feature_names=FEATURE_COLS)
    assert (out["churn_class"] == "No").all()


def test_predict_clients_requires_customerid():
    model, df = model_for_training()
    with pytest.raises(ValueError):
        predict_clients(
            model, df.drop(columns=["customerID"]), feature_names=FEATURE_COLS
        )


def test_predict_clients_drops_target_before_predicting():
    model, df = model_for_training()
    df_with_target = df.copy()
    df_with_target["Churn"] = "No"
    out = predict_clients(model, df_with_target, feature_names=FEATURE_COLS)
    expected = predict_clients(model, df, feature_names=FEATURE_COLS)
    pd.testing.assert_frame_equal(out, expected)
