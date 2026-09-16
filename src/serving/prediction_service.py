"""Servicio de predicción de churn por cliente (T-27).

Encapsula la lógica de inferencia (carga del pipeline del modelo y predicción)
en un servicio inyectable, de modo que la API (`src/serving/api.py`) se
limita a validar el contrato HTTP y delegar en él.

Garantiza el requisito de `docs/specs.md` §8.1: la misma transformación de
features (`T-10`) y el mismo pipeline (preprocessing `T-12` + modelo
seleccionado `T-14`) que en entrenamiento, sobre datos sin target conocido.

El modelo se carga una única vez por instancia (política lazy) y se reutiliza
en todas las predicciones; la carga fallida se expone como error explícito
para que la capa HTTP la traduzca en `model_unavailable`.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn.pipeline import Pipeline

from src.analysis.features import engineer_features
from src.serving.predict import PREDICTION_THRESHOLD, predict_clients


class PredictionService:
    """Predice churn por cliente con el pipeline del modelo seleccionado."""

    def __init__(
        self,
        load_pipeline: Callable[[], Pipeline] | None = None,
        threshold: float = PREDICTION_THRESHOLD,
    ) -> None:
        self._load_pipeline = load_pipeline
        self._threshold = threshold
        self._model: Pipeline | None = None

    @property
    def model(self) -> Pipeline:
        """Pipeline del modelo, cargado una única vez (lazy).

        Si la carga falla, la excepción original sigue en cadena; la capa
        HTTP la traduce a `model_unavailable` sin exponer detalles internos.
        """
        if self._model is None:
            if self._load_pipeline is None:
                raise RuntimeError(
                    "No hay cargador de modelo configurado en el servicio."
                )
            self._model = self._load_pipeline()
        return self._model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve la salida estándar para el/los clientes de ``df``.

        ``df`` debe seguir el esquema procesado (con ``customerID``); aplica
        el feature engineering (`T-10`) y el pipeline del modelo, y devuelve
        customerID, probabilidad de churn y clase predicha (specs.md §8.1).
        """
        features = engineer_features(df)
        return predict_clients(self.model, features, threshold=self._threshold)
