"""Pruebas de la orquestación del pipeline reproducible (T-15)."""

import pytest
from src.serving.pipeline import STAGE_NAMES, run_pipeline


def test_stage_names_order_es_end_to_end():
    assert STAGE_NAMES == (
        "datos",
        "features",
        "particiones",
        "entrenamiento",
        "tracking",
        "seleccion",
        "prediccion",
    )


def test_run_pipeline_unknown_stage_raises():
    with pytest.raises(ValueError):
        run_pipeline(["no-existe"])


def test_run_pipeline_selects_subset_in_canonical_order():
    from unittest import mock

    calls: list[str] = []
    runners = {
        "datos": lambda: calls.append("datos"),
        "features": lambda: calls.append("features"),
        "particiones": lambda: calls.append("particiones"),
        "entrenamiento": lambda: calls.append("entrenamiento"),
        "tracking": lambda: calls.append("tracking"),
        "seleccion": lambda: calls.append("seleccion"),
        "prediccion": lambda: calls.append("prediccion"),
    }
    with mock.patch(
        "src.serving.pipeline.PIPELINE_STAGES",
        tuple((name, fn) for name, fn in runners.items()),
    ):
        run_pipeline(["prediccion", "features"])
    assert calls == ["prediccion", "features"]
