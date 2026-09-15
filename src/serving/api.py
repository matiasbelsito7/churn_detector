"""API de predicción de churn con FastAPI (T-18).

Cumple `docs/specs.md` §8.2:

- `GET /health`: disponibilidad del servicio.
- `POST /predict`: predicción por cliente. Recibe los campos del esquema
  procesado (`docs/data_contract.md`), aplica el mismo feature engineering que
  en entrenamiento (`T-10`) y devuelve identificador, probabilidad de churn y
  clase predicha usando el modelo seleccionado (`T-14`) con su pipeline,
  cargado desde el Model Registry de MLflow (`T-17`).

Los errores son explícitos y no exponen detalles internos: entrada inválida
devuelve un código `invalid_request` con mención de los campos, y una
indisponibilidad del modelo devuelve `model_unavailable`.

Ejecución:
    python -m src.serving.api
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, cast

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.pipeline import Pipeline

from src.analysis.features import engineer_features
from src.modeling.select import load_selected_model
from src.serving.predict import PREDICTION_THRESHOLD, predict_clients

logger = logging.getLogger(__name__)

SELECTED_MODEL_NAME = "logistic-regression"
HEALTH_ENDPOINT = "/health"
PREDICT_ENDPOINT = "/predict"


class ApiError(Exception):
    """Error deliberado de la API con código y mensaje públicos."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class PredictionRequest(BaseModel):
    """Contrato de entrada: esquema procesado sin target (specs.md §8.1)."""

    model_config = ConfigDict(extra="forbid")

    customerID: str = Field(..., min_length=1, description="Identificador del cliente.")
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(..., ge=18.25, le=118.75)
    TotalCharges: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _charges_consistent(self) -> PredictionRequest:
        if self.tenure > 0 and self.TotalCharges is None:
            raise ValueError("TotalCharges es obligatorio cuando tenure > 0")
        return self


class PredictionResponse(BaseModel):
    """Contrato de salida: identificador, probabilidad y clase (specs.md §8.1)."""

    customerID: str
    churn_prob: float
    churn_class: Literal["Yes", "No"]


def _payload_to_frame(payload: PredictionRequest) -> pd.DataFrame:
    """Convierte el payload en un DataFrame con el esquema procesado."""
    data = payload.model_dump()
    customer_id = data.pop("customerID")
    frame = pd.DataFrame([data])
    frame.insert(0, "customerID", customer_id)
    return frame


def _error_body(
    code: str, message: str, errors: list[dict[str, str]] | None = None
) -> dict[str, object]:
    body: dict[str, object] = {"code": code, "message": message}
    if errors:
        body["errors"] = errors
    return body


def _validation_errors(exc: RequestValidationError) -> list[dict[str, str]]:
    errors = []
    for err in exc.errors():
        loc = [str(part) for part in err.get("loc", [])]
        field = loc[-1] if loc else "body"
        errors.append(
            {"field": field, "message": str(err.get("msg", "valor inválido"))}
        )
    return errors


def _load_model(state: Any) -> Pipeline:
    """Carga el modelo desde el Model Registry (una única vez por proceso)."""
    if state.model is None:
        try:
            state.model = state.load_pipeline()
        except Exception:
            logger.exception("No se pudo cargar el modelo seleccionado")
            raise ApiError(
                503,
                "model_unavailable",
                "El modelo de predicción no está disponible en este momento.",
            ) from None
    return state.model


def create_app(
    load_pipeline: Callable[[], Pipeline] | None = None,
    threshold: float = PREDICTION_THRESHOLD,
) -> FastAPI:
    """Construye la app. Permite inyectar el cargador del modelo en pruebas."""
    app = FastAPI(
        title="Churn Detector API",
        description=(
            "Predicción de churn por cliente (probabilidad, clase e "
            "identificador) con el pipeline del modelo seleccionado."
        ),
        version="0.1.0",
    )
    app.state.load_pipeline = load_pipeline or (
        lambda: load_selected_model(SELECTED_MODEL_NAME)
    )
    app.state.model = None
    app.state.threshold = threshold

    @app.get(HEALTH_ENDPOINT)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(PREDICT_ENDPOINT, response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        df = _payload_to_frame(payload)
        df = engineer_features(df)
        model = _load_model(request.app.state)
        out = predict_clients(model, df, threshold=request.app.state.threshold)
        row = out.iloc[0]
        return PredictionResponse(
            customerID=str(row["customerID"]),
            churn_prob=float(row["churn_prob"]),
            churn_class=cast(Literal["Yes", "No"], str(row["churn_class"])),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_body(
                "invalid_request",
                "Entrada inválida: revise los campos indicados.",
                _validation_errors(exc),
            ),
        )

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Error no controlado en la API")
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "No se pudo procesar la predicción."),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serving.api:app", host="127.0.0.1", port=8000)
