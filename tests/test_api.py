"""Pruebas de integración de la API de predicción (T-18)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.modeling.preprocessing import (
    FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    build_preprocessing_pipeline,
)
from src.serving.api import HEALTH_ENDPOINT, PREDICT_ENDPOINT, create_app


def fake_model() -> Pipeline:
    rng = np.random.RandomState(0)
    n = 200
    data = {}
    for name in FEATURE_NAMES:
        if name in NUMERIC_FEATURE_NAMES:
            data[name] = rng.rand(n) * 50
        else:
            data[name] = rng.choice(["No", "Yes"], size=n)
    X = pd.DataFrame(data, columns=list(FEATURE_NAMES))
    labels = np.where(rng.rand(n) > 0.5, "Yes", "No")
    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


VALID_PAYLOAD = {
    "customerID": "0001-ABCD",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 74.5,
    "TotalCharges": 1788.0,
}


@pytest.fixture()
def client() -> TestClient:
    app = create_app(load_pipeline=fake_model)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client_without_model() -> TestClient:
    def broken_loader() -> Pipeline:
        raise RuntimeError("credenciales secretas del backend")

    return TestClient(
        create_app(load_pipeline=broken_loader), raise_server_exceptions=False
    )


def payload(**overrides: object) -> dict[str, object]:
    body = dict(VALID_PAYLOAD)
    body.update(overrides)
    return body


def test_health_check(client: TestClient) -> None:
    resp = client.get(HEALTH_ENDPOINT)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_payload(client: TestClient) -> None:
    resp = client.post(PREDICT_ENDPOINT, json=payload())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"customerID", "churn_prob", "churn_class"}
    assert body["customerID"] == VALID_PAYLOAD["customerID"]
    assert isinstance(body["churn_prob"], float)
    assert 0.0 <= body["churn_prob"] <= 1.0
    assert body["churn_class"] in {"Yes", "No"}


def test_predict_missing_field_explicit_error(client: TestClient) -> None:
    resp = client.post(PREDICT_ENDPOINT, json=payload(gender=None))
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "invalid_request"
    assert any(err["field"] == "gender" for err in body["errors"])


def test_predict_invalid_category(client: TestClient) -> None:
    resp = client.post(PREDICT_ENDPOINT, json=payload(InternetService="Satellite"))
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"


def test_predict_out_of_range(client: TestClient) -> None:
    resp = client.post(PREDICT_ENDPOINT, json=payload(tenure=-1))
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"


def test_predict_unknown_field_rejected(client: TestClient) -> None:
    resp = client.post(PREDICT_ENDPOINT, json=payload(malware="x"))
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"


def test_predict_charges_consistency(client: TestClient) -> None:
    resp = client.post(PREDICT_ENDPOINT, json=payload(TotalCharges=None))
    assert resp.status_code == 422
    body = resp.json()
    assert any("TotalCharges" in err["message"] for err in body["errors"])


def test_predict_model_unavailable(client_without_model: TestClient) -> None:
    resp = client_without_model.post(PREDICT_ENDPOINT, json=payload())
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "model_unavailable"
    assert "credenciales" not in body["message"]


def test_health_available_without_model(client_without_model: TestClient) -> None:
    resp = client_without_model.get(HEALTH_ENDPOINT)
    assert resp.status_code == 200


def test_predict_internal_error_hides_details(client: TestClient, monkeypatch) -> None:
    import src.serving.api as api_module

    def boom(model, df, *args, **kwargs):
        raise ValueError("no se debe exponer este detalle")

    monkeypatch.setattr(api_module, "predict_clients", boom)
    resp = client.post(PREDICT_ENDPOINT, json=payload())
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    assert "no se debe exponer" not in body["message"]
