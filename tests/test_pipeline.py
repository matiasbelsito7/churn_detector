"""Pruebas de la orquestación del pipeline reproducible (T-15)."""

from unittest import mock

import pytest
from src.serving import pipeline
from src.serving.pipeline import STAGE_NAMES, run_pipeline


def test_stage_names_order_es_end_to_end():
    assert STAGE_NAMES == (
        "datos",
        "features",
        "particiones",
        "entrenamiento",
        "tracking",
        "seleccion",
        "explicabilidad",
        "prediccion",
    )


def test_run_pipeline_unknown_stage_raises():
    with pytest.raises(ValueError):
        run_pipeline(["no-existe"])


def test_run_pipeline_selects_subset_in_canonical_order():
    calls: list[str] = []
    runners = {
        "datos": lambda: calls.append("datos"),
        "features": lambda: calls.append("features"),
        "particiones": lambda: calls.append("particiones"),
        "entrenamiento": lambda: calls.append("entrenamiento"),
        "tracking": lambda: calls.append("tracking"),
        "seleccion": lambda: calls.append("seleccion"),
        "explicabilidad": lambda: calls.append("explicabilidad"),
        "prediccion": lambda: calls.append("prediccion"),
    }
    with mock.patch(
        "src.serving.pipeline.PIPELINE_STAGES",
        tuple((name, fn) for name, fn in runners.items()),
    ):
        run_pipeline(["prediccion", "features"])
    assert calls == ["prediccion", "features"]


def test_main_defaults_to_all_stages_in_order():
    calls: list[str] = []
    runners = {name: (lambda n=name: calls.append(n)) for name in STAGE_NAMES}
    with (
        mock.patch(
            "src.serving.pipeline.PIPELINE_STAGES",
            tuple((name, fn) for name, fn in runners.items()),
        ),
        mock.patch("sys.argv", ["pipeline"]),
    ):
        assert pipeline.main() == 0
    assert calls == list(STAGE_NAMES)


def test_main_runs_only_requested_stages():
    calls: list[str] = []
    runners = {name: (lambda n=name: calls.append(n)) for name in STAGE_NAMES}
    with (
        mock.patch(
            "src.serving.pipeline.PIPELINE_STAGES",
            tuple((name, fn) for name, fn in runners.items()),
        ),
        mock.patch("sys.argv", ["pipeline", "tracking"]),
    ):
        assert pipeline.main() == 0
    assert calls == ["tracking"]


@pytest.mark.parametrize(
    ("stage_fn", "module_path"),
    [
        ("stage_clean", "src.data.clean.main"),
        ("stage_features", "src.analysis.features.main"),
        ("stage_split", "src.modeling.split.main"),
        ("stage_train", "src.modeling.train.main"),
        ("stage_tracking", "src.modeling.tracking.main"),
        ("stage_select", "src.modeling.select.main"),
        ("stage_explain", "src.modeling.explain.main"),
        ("stage_predict", "src.serving.predict.main"),
    ],
)
def test_stages_delegate_to_module_main(stage_fn, module_path):
    with mock.patch(module_path) as module_main:
        getattr(pipeline, stage_fn)()
    module_main.assert_called_once_with()
