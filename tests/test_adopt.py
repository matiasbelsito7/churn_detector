"""Pruebas de la adopción de configuración ganadora (T-30)."""

import json

from src.modeling.adopt import (
    CANDIDATE_PARAMS,
    CHAMPION_FAMILY,
    _append_selection_md,
    _write_selection,
    build_payload,
    build_report,
    is_better,
)
from src.modeling.select import MODEL_PREFIX, SELECTION_RULE

METRICS = {
    "auc_roc": 0.85,
    "auc_pr": 0.67,
    "recall_pos": 0.80,
    "precision_pos": 0.51,
    "f1_pos": 0.62,
    "accuracy": 0.74,
    "true_positive": 10,
    "false_negative": 5,
    "false_positive": 3,
    "true_negative": 20,
}


def candidate_metrics() -> dict[str, object]:
    m = dict(METRICS)
    m["auc_pr"] = 0.68
    return m


class FakeResult:
    def __init__(self, metrics: dict[str, object]):
        self.name = CHAMPION_FAMILY
        self.metrics = metrics
        self.pipeline = object()
        self.params = {"C": 1.0}


def test_is_better_prizes_auc_pr_over_recall():
    better = dict(candidate_metrics())
    champ = dict(METRICS)
    assert is_better(better, champ) is True
    assert is_better(champ, better) is False


def test_is_better_tie_break_and_equal():
    champ = dict(METRICS)
    higher_recall = dict(METRICS)
    higher_recall["recall_pos"] = 0.85
    assert is_better(higher_recall, champ) is True
    assert is_better(dict(METRICS), dict(METRICS)) is False
    assert is_better(candidate_metrics(), higher_recall) is True


def test_build_report_adoption_decision():
    report = build_report(
        FakeResult(METRICS),
        FakeResult(candidate_metrics()),
        True,
        {
            "version": 4,
            "test_file": {"path": "data/splits/test.csv", "sha256": "x", "size": 1057},
            "test_metrics": METRICS,
            "artifact": {
                "path": "data/models/churn-logistic-regression.joblib",
                "standalone": True,
            },
        },
        {"version": 3, "test_metrics": METRICS},
    )
    assert "# Adopción de configuración del modelo (T-30)" in report
    assert "**ADOPTAR.**" in report
    assert CHAMPION_FAMILY in report
    assert f"`{MODEL_PREFIX}{CHAMPION_FAMILY}`" in report
    assert "Champion anterior" in report
    assert SELECTION_RULE in report


def test_build_report_discard_does_not_promote():
    report = build_report(
        FakeResult(candidate_metrics()),
        FakeResult(METRICS),
        False,
        None,
        None,
    )
    assert "**DESCARTAR.**" in report
    assert "No se modifica el modelo en producción" in report


def test_build_payload_adopted_contains_registry():
    payload = build_payload(
        FakeResult(METRICS),
        FakeResult(candidate_metrics()),
        True,
        {
            "version": 4,
            "test_file": {"path": "data/splits/test.csv", "sha256": "y", "size": 10},
            "test_metrics": METRICS,
            "artifact": {"path": "data/models/x.joblib", "standalone": True},
        },
        {"version": 3, "test_metrics": {"auc_pr": 0.66, "recall_pos": 0.8}},
    )
    assert payload["task"] == "T-30"
    assert payload["decision"] == "adoptar"
    assert payload["registry"]["version"] == 4
    assert payload["candidate_params"] == CANDIDATE_PARAMS
    assert payload["test_metrics"]["auc_pr"] == 0.67
    assert payload["previous_champion"]["version"] == 3


def test_build_payload_discarded_is_minimal():
    payload = build_payload(
        FakeResult(METRICS), FakeResult(candidate_metrics()), False, None, None
    )
    assert payload["decision"] == "descartar"
    assert "registry" not in payload
    assert "test_metrics" not in payload


def test_write_selection_records_new_champion_and_history(tmp_path, monkeypatch):
    import pandas as pd
    import src.modeling.adopt as adopt

    df = pd.DataFrame({"customerID": ["a"], "Churn": ["No"]})
    test_file = tmp_path / "test.csv"
    df.to_csv(test_file, index=False)
    selection_json = tmp_path / "selection.json"
    monkeypatch.setattr(adopt, "TEST_FILE", test_file)
    monkeypatch.setattr(adopt, "SELECTION_JSON", selection_json)
    previous = {
        "version": 3,
        "validation": {},
        "test_metrics": METRICS,
        "created_at": "t0",
    }
    _write_selection(
        FakeResult(METRICS),
        FakeResult(candidate_metrics()),
        4,
        METRICS,
        tmp_path / "m.joblib",
        previous,
    )
    payload = json.loads(selection_json.read_text(encoding="utf-8"))
    assert payload["selected"] == CHAMPION_FAMILY
    assert payload["version"] == 4
    assert payload["updated_by"] == "T-30 (adopcion)"
    assert payload["previous_champion"]["version"] == 3
    assert payload["registry"]["alias"] == "production"


def test_append_selection_md_documents_change(tmp_path, monkeypatch):
    import src.modeling.adopt as adopt

    selection_md = tmp_path / "selection.md"
    selection_md.write_text("# Selección del modelo (T-14)\n\nbody\n", encoding="utf-8")
    monkeypatch.setattr(adopt, "SELECTION_MD", selection_md)
    _append_selection_md(
        FakeResult(METRICS), FakeResult(candidate_metrics()), 4, METRICS
    )
    text = selection_md.read_text(encoding="utf-8")
    assert "## Adopción de configuración (T-30)" in text
    assert "body" in text
    assert "v4" in text


def test_main_discards_without_touching_registry(tmp_path, monkeypatch):
    from unittest import mock

    import src.modeling.adopt as adopt

    better_champ = FakeResult(candidate_metrics())
    worse_cand = FakeResult(METRICS)
    adoption_md = tmp_path / "adoption.md"
    adoption_json = tmp_path / "adoption.json"
    monkeypatch.setattr(
        adopt,
        "train_candidate",
        lambda name, train, val, **kw: better_champ if not kw else worse_cand,
    )
    monkeypatch.setattr(adopt, "TRAIN_FILE", tmp_path / "train.csv")
    monkeypatch.setattr(adopt, "VAL_FILE", tmp_path / "val.csv")
    monkeypatch.setattr(adopt, "ADOPTION_MD", adoption_md)
    monkeypatch.setattr(adopt, "ADOPTION_JSON", adoption_json)
    monkeypatch.setattr(adopt, "SELECTION_JSON", tmp_path / "selection.json")
    with (
        mock.patch(
            "src.modeling.adopt.pd.read_csv",
            return_value=__import__("pandas").DataFrame({"Churn": ["No"]}),
        ),
        mock.patch("src.modeling.adopt.log_candidate_run") as log_run,
        mock.patch("src.modeling.adopt.promote_model") as promote,
        mock.patch("src.modeling.adopt.load_selected_model") as load_model,
        mock.patch("src.modeling.adopt.export_model") as export,
    ):
        assert adopt.main() == 0
    log_run.assert_not_called()
    promote.assert_not_called()
    load_model.assert_not_called()
    export.assert_not_called()
    decision = json.loads(adoption_json.read_text(encoding="utf-8"))
    assert decision["decision"] == "descartar"
    assert "**DESCARTAR.**" in adoption_md.read_text(encoding="utf-8")


def test_decision_guard_blocks_repeat_evaluation(tmp_path, monkeypatch):
    import src.modeling.adopt as adopt
    from src.hashing import sha256_file

    train_file = tmp_path / "train.csv"
    val_file = tmp_path / "val.csv"
    train_file.write_text("a\n1\n2\n", encoding="utf-8")
    val_file.write_text("b\n1\n2\n", encoding="utf-8")
    adoption_json = tmp_path / "adoption.json"
    adoption_json.write_text(
        json.dumps(
            {
                "task": "T-30",
                "candidate_params": CANDIDATE_PARAMS,
                "decision": "descartar",
                "inputs": {
                    "train_sha256": sha256_file(train_file),
                    "validation_sha256": sha256_file(val_file),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adopt, "TRAIN_FILE", train_file)
    monkeypatch.setattr(adopt, "VAL_FILE", val_file)
    monkeypatch.setattr(adopt, "ADOPTION_JSON", adoption_json)
    assert adopt._decision_guard() is True


def test_decision_guard_allows_when_params_or_data_change(tmp_path, monkeypatch):
    import src.modeling.adopt as adopt

    adoption_json = tmp_path / "adoption.json"
    adoption_json.write_text(
        json.dumps(
            {
                "candidate_params": {"C": 0.5},
                "decision": "descartar",
                "inputs": {"train_sha256": "x", "validation_sha256": "y"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adopt, "ADOPTION_JSON", adoption_json)
    assert adopt._decision_guard() is False


def test_build_rejection_payload_records_test_basis():
    from src.modeling.adopt import build_rejection_payload

    previous = {"version": 3, "test_metrics": METRICS}
    rejected = {
        "version": 4,
        "validation": {
            "champion-default": METRICS,
            "candidate-t30": candidate_metrics(),
        },
        "test_metrics": dict(METRICS, auc_pr=0.63, recall_pos=0.8),
    }
    payload = build_rejection_payload(previous, rejected, 3)
    assert payload["decision"] == "descartar"
    assert payload["basis"] == "test"
    assert payload["test"]["champion"]["auc_pr"] == 0.67
    assert payload["test"]["candidate-rejected"]["auc_pr"] == 0.63
    assert payload["reverted_to"]["version"] == 3
    assert payload["test_improvement"]["auc_pr"] < 0


def test_build_rejection_report_documents_revert():
    from src.modeling.adopt import build_rejection_report

    previous = {"version": 3, "test_metrics": METRICS}
    rejected = {
        "version": 4,
        "validation": {
            "champion-default": METRICS,
            "candidate-t30": candidate_metrics(),
        },
        "test_metrics": dict(METRICS, auc_pr=0.63, recall_pos=0.8),
    }
    report = build_rejection_report(previous, rejected, 3)
    assert "**DESCARTAR.**" in report
    assert "## Comparación sobre test (decisión final)" in report
    assert "Acciones de reversión" in report
    assert "v3" in report
    assert "v4" in report


def test_revert_restores_previous_champion(tmp_path, monkeypatch):
    from unittest import mock

    import pandas as pd
    import src.modeling.adopt as adopt

    selection_json = tmp_path / "selection.json"
    selection_json.write_text(
        json.dumps(
            {
                "task": "T-14",
                "updated_by": "T-30 (adopcion)",
                "version": 4,
                "validation": {
                    "champion-default": METRICS,
                    "candidate-t30": candidate_metrics(),
                },
                "test_file": {
                    "path": "data/splits/test.csv",
                    "sha256": "x",
                    "size": 1057,
                },
                "test_metrics": dict(METRICS, auc_pr=0.63, recall_pos=0.8),
                "registry": {"alias": "production", "version": 4},
                "artifact": {"path": "data/models/x.joblib", "standalone": True},
                "previous_champion": {
                    "version": 3,
                    "validation": {"logistic-regression": METRICS},
                    "test_metrics": METRICS,
                    "created_at": "t0",
                },
                "created_at": "t1",
            }
        ),
        encoding="utf-8",
    )
    selection_md = tmp_path / "selection.md"
    selection_md.write_text(
        "# Selección del modelo (T-14)\n\nbody\n\n"
        "## Adopción de configuración (T-30)\n\ndummy\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "test.csv"
    pd.DataFrame({"customerID": ["a"], "Churn": ["No"]}).to_csv(test_file, index=False)
    adoption_json = tmp_path / "adoption.json"
    adoption_md = tmp_path / "adoption.md"
    model_file = tmp_path / "m.joblib"
    monkeypatch.setattr(adopt, "SELECTION_JSON", selection_json)
    monkeypatch.setattr(adopt, "SELECTION_MD", selection_md)
    monkeypatch.setattr(adopt, "TEST_FILE", test_file)
    monkeypatch.setattr(adopt, "ADOPTION_JSON", adoption_json)
    monkeypatch.setattr(adopt, "ADOPTION_MD", adoption_md)

    class FakePipeline:
        pass

    with (
        mock.patch("src.modeling.adopt.promote_model") as promote,
        mock.patch("src.modeling.adopt._retire_version") as retire,
        mock.patch(
            "src.modeling.adopt.load_selected_model", return_value=FakePipeline()
        ),
        mock.patch(
            "src.modeling.adopt.export_model", return_value=model_file
        ) as export,
        mock.patch("src.serving.predict.main") as predict_main,
    ):
        rc = adopt._reject_and_revert()

    assert rc == 0
    promote.assert_called_once_with(CHAMPION_FAMILY, version=3)
    retire.assert_called_once_with(CHAMPION_FAMILY, 4)
    export.assert_called_once()
    predict_main.assert_called_once()
    restored = json.loads(selection_json.read_text(encoding="utf-8"))
    assert restored["version"] == 3
    assert restored["updated_by"] == "T-30 (revertido)"
    assert restored["test_metrics"]["auc_pr"] == METRICS["auc_pr"]
    assert restored["rejected_attempt"]["version"] == 4
    md = selection_md.read_text(encoding="utf-8")
    assert "## Adopción de configuración (T-30)" not in md
    assert "## Reversión de adopción (T-30)" in md
    decision = json.loads(adoption_json.read_text(encoding="utf-8"))
    assert decision["decision"] == "descartar"
    assert decision["basis"] == "test"
    assert "**DESCARTAR.**" in adoption_md.read_text(encoding="utf-8")
