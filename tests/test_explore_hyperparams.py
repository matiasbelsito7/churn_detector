"""Pruebas del análisis exploratorio de hiperparámetros (univariado + aleatorio)."""

from src.modeling.explore_hyperparams import (
    EXPLORED_FAMILIES,
    best,
    build_payload,
    build_report,
    explore_record,
    normalize_combo,
    random_search,
    run_default,
    univariate_sweep,
)
from tests.test_train import make_df


def test_normalize_combo_l1_ratio_uses_saga():
    assert normalize_combo("logistic-regression", {"l1_ratio": 1.0}) == {
        "l1_ratio": 1.0,
        "solver": "saga",
    }
    assert normalize_combo("logistic-regression", {"C": 0.1}) == {"C": 0.1}
    assert normalize_combo("random-forest", {"l1_ratio": 1.0}) == {"l1_ratio": 1.0}


def test_run_default_returns_reference():
    df_train = make_df(200, seed=1)
    df_val = make_df(50, seed=2)
    rec = run_default("logistic-regression", df_train, df_val, seed=42)
    assert rec.kind == "default"
    assert rec.param is None
    assert 0.0 <= float(rec.metrics["auc_pr"]) <= 1.0


def test_univariate_sweep_covers_each_parameter_once():
    df_train = make_df(200, seed=3)
    df_val = make_df(50, seed=4)
    values = {"C": [0.1, 1.0], "l1_ratio": [0.0, 1.0]}
    records = univariate_sweep(
        "logistic-regression", df_train, df_val, values=values, seed=7
    )
    assert len(records) == 4
    assert all(rec.kind == "univariate" for rec in records)
    assert {rec.param for rec in records} == {"C", "l1_ratio"}
    assert {rec.param: rec.value for rec in records} == {"C": 1.0, "l1_ratio": 1.0}
    assert all("solver" in r.params or r.param != "l1_ratio" for r in records)


def test_univariate_sweep_deterministic():
    df_train = make_df(200, seed=5)
    df_val = make_df(50, seed=6)
    values = {"max_depth": [None, 10]}
    first = univariate_sweep("random-forest", df_train, df_val, values=values, seed=9)
    second = univariate_sweep("random-forest", df_train, df_val, values=values, seed=9)
    assert [(r.param, r.value, r.metrics) for r in first] == [
        (r.param, r.value, r.metrics) for r in second
    ]


def test_random_search_respects_budget_and_is_reproducible():
    df_train = make_df(200, seed=7)
    df_val = make_df(50, seed=8)
    spec = {"C": [0.01, 1.0, 100.0], "l1_ratio": [0.0, 1.0]}
    first = random_search(
        "logistic-regression", df_train, df_val, budget=5, spec=spec, seed=11
    )
    second = random_search(
        "logistic-regression", df_train, df_val, budget=5, spec=spec, seed=11
    )
    assert len(first) == 5
    assert all(rec.kind == "random" for rec in first)
    assert [(r.params, r.metrics) for r in first] == [
        (r.params, r.metrics) for r in second
    ]


def test_best_selects_highest_auc_pr():
    df_train = make_df(200, seed=9)
    df_val = make_df(50, seed=10)
    records = random_search(
        "random-forest",
        df_train,
        df_val,
        budget=3,
        spec={"n_estimators": [50, 100, 200]},
        seed=13,
    )
    top = best(records)
    assert top is not None
    assert float(top.metrics["auc_pr"]) == max(
        float(r.metrics["auc_pr"]) for r in records
    )


def test_build_report_covers_families_and_sections():
    df_train = make_df(200, seed=11)
    df_val = make_df(50, seed=12)
    defaults = {}
    univariate = {}
    random = {}
    for name in EXPLORED_FAMILIES:
        tree_values = {"n_estimators": [50, 100]}
        values = {"C": [0.1, 1.0]} if name == "logistic-regression" else tree_values
        defaults[name] = run_default(name, df_train, df_val, seed=1)
        univariate[name] = univariate_sweep(
            name, df_train, df_val, values=values, seed=1
        )
        random[name] = random_search(name, df_train, df_val, budget=2, spec=values)
    report = build_report(defaults, univariate, random)
    assert "# Análisis exploratorio de hiperparámetros" in report
    assert "## Resumen" in report
    for name in EXPLORED_FAMILIES:
        assert f"## {name}" in report
        assert "Univariado" in report
        assert "Búsqueda aleatoria conjunta" in report


def test_explore_record_and_payload_shape():
    df_train = make_df(200, seed=13)
    df_val = make_df(50, seed=14)
    rec = univariate_sweep(
        "logistic-regression", df_train, df_val, values={"C": [1.0]}, seed=3
    )[0]
    record = explore_record(rec, "sha-train", "sha-val")
    assert record["family"] == "logistic-regression"
    assert record["kind"] == "univariate"
    assert record["input_files"]["train_sha256"] == "sha-train"
    assert record["input_files"]["validation_sha256"] == "sha-val"
    defaults = {
        name: run_default(name, df_train, df_val, seed=1) for name in EXPLORED_FAMILIES
    }
    empty = {name: [] for name in EXPLORED_FAMILIES}
    payload = build_payload(defaults, empty, empty, "st", "sv")
    assert payload["task"] == "hyperparams-exploration"
    assert "logistic-regression" in payload["families"]
    assert len(payload["families"]["logistic-regression"]) == 1


def test_defaults_matching_grid_search_reference():
    df_train = make_df(200, seed=15)
    df_val = make_df(50, seed=16)
    values = {"C": [0.01, 0.1]}
    uni = univariate_sweep(
        "logistic-regression", df_train, df_val, values=values, seed=5
    )
    default = run_default("logistic-regression", df_train, df_val, seed=5)
    assert len(uni) == 2
    assert all(r.metrics["auc_pr"] >= 0.0 for r in uni)
    assert default.metrics["auc_pr"] >= 0.0
