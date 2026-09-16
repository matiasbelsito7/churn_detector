"""Pruebas de integración de la API de predicción (T-18)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.modeling.preprocessing import (
    FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    build_preprocessing_pipeline,
)
from src.serving.api import (
    HEALTH_ENDPOINT,
    PREDICT_ENDPOINT,
    PredictionRequest,
    _payload_to_frame,
    create_app,
)
from src.serving.prediction_service import PredictionService


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
    import src.serving.prediction_service as service_module

    def boom(model, df, *args, **kwargs):
        raise ValueError("no se debe exponer este detalle")

    monkeypatch.setattr(service_module, "predict_clients", boom)
    resp = client.post(PREDICT_ENDPOINT, json=payload())
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    assert "no se debe exponer" not in body["message"]


def test_default_load_pipeline_reads_model_file(monkeypatch, tmp_path) -> None:
    import src.serving.api as api_module

    model_file = tmp_path / "model.joblib"
    dump(fake_model(), model_file)
    monkeypatch.setenv("MODEL_FILE", str(model_file))
    monkeypatch.setattr(api_module, "MODEL_FILE", str(model_file))

    model = api_module._default_load_pipeline()
    assert isinstance(model, Pipeline)


def test_default_load_pipeline_falls_back_to_registry(monkeypatch) -> None:
    import src.serving.api as api_module

    monkeypatch.delenv("MODEL_FILE", raising=False)
    monkeypatch.setattr(api_module, "MODEL_FILE", None)
    fake = object()
    monkeypatch.setattr(api_module, "load_selected_model", lambda *a, **kw: fake)

    assert api_module._default_load_pipeline() is fake


def test_default_load_pipeline_missing_file_raises(monkeypatch, tmp_path) -> None:
    import src.serving.api as api_module

    missing = tmp_path / "no-existe.joblib"
    monkeypatch.setenv("MODEL_FILE", str(missing))
    monkeypatch.setattr(api_module, "MODEL_FILE", str(missing))

    with pytest.raises(FileNotFoundError):
        api_module._default_load_pipeline()


def test_prediction_service_loads_model_once(tmp_path) -> None:
    import joblib

    model_file = tmp_path / "model.joblib"
    dump(fake_model(), model_file)
    calls = []

    def load_pipeline():
        calls.append(True)
        return joblib.load(model_file)

    service = PredictionService(load_pipeline=load_pipeline)
    frame = _payload_to_frame(PredictionRequest(**VALID_PAYLOAD))
    service.predict(frame)
    service.predict(frame)
    assert len(calls) == 1


def test_prediction_service_output_columns():
    service = PredictionService(load_pipeline=fake_model)
    frame = _payload_to_frame(PredictionRequest(**VALID_PAYLOAD))
    out = service.predict(frame)
    assert list(out.columns) == ["customerID", "churn_prob", "churn_class"]
    assert out.iloc[0]["customerID"] == VALID_PAYLOAD["customerID"]
