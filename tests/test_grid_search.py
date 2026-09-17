"""Pruebas de la búsqueda exploratoria de hiperparámetros (grid search)."""

import pytest
from src.modeling.grid_search import (
    GRIDS,
    _format_params,
    best_per_family,
    build_report,
    combinations,
    grid_record,
    run_grid_search,
)
from tests.test_train import make_df


def test_combinations_returns_cartesian_product():
    grid = {"a": [1, 2], "b": [True, False]}
    combos = combinations(grid)
    assert len(combos) == 4
    assert {"a": 2, "b": False} in combos
    assert combos[0] == {"a": 1, "b": True}


def test_run_grid_search_includes_default_and_grid_rows():
    df_train = make_df(250, seed=1)
    df_val = make_df(60, seed=2)
    grids = {"logistic-regression": {"C": [0.01, 1.0]}}
    records = run_grid_search(df_train, df_val, grids=grids)
    assert len(records) == 3
    assert sum(rec.is_default for rec in records) == 1
    assert all(rec.name == "logistic-regression" for rec in records)
    aucs = [float(rec.metrics["auc_pr"]) for rec in records]
    assert aucs == sorted(aucs, reverse=True)
    for rec in records:
        for key in ("auc_roc", "auc_pr", "recall_pos", "precision_pos", "f1_pos"):
            assert 0.0 <= float(rec.metrics[key]) <= 1.0


def test_run_grid_search_supports_xgboost():
    df_train = make_df(250, seed=3)
    df_val = make_df(60, seed=4)
    grids = {"xgboost": {"n_estimators": [50]}}
    records = run_grid_search(df_train, df_val, grids=grids)
    assert len(records) == 2
    assert records[0].name == "xgboost"
    assert isinstance(records[0].metrics["auc_pr"], (float, int))


def test_default_rows_match_official_candidates():
    df_train = make_df(250, seed=5)
    df_val = make_df(60, seed=6)
    records = run_grid_search(df_train, df_val)
    names = {rec.name for rec in records}
    assert names == set(GRIDS)
    defaults = [rec for rec in records if rec.is_default]
    assert {rec.name for rec in defaults} == set(GRIDS)


def test_best_per_family_picks_each_family():
    df_train = make_df(250, seed=7)
    df_val = make_df(60, seed=8)
    records = run_grid_search(df_train, df_val, grids={"random-forest": {}})
    best = best_per_family(records)
    assert set(best) == {"random-forest"}
    assert best["random-forest"].metrics == max(
        (rec.metrics for rec in records),
        key=lambda m: (float(m["auc_pr"]), float(m["recall_pos"])),
    )


def test_build_report_lists_results_and_best():
    df_train = make_df(250, seed=9)
    df_val = make_df(60, seed=10)
    records = run_grid_search(df_train, df_val)
    report = build_report(records)
    assert "# Búsqueda de hiperparámetros (exploratoria)" in report
    assert "Mejor configuración por familia:" in report
    assert "Candidato global:" in report
    for name in GRIDS:
        assert name in report


def test_format_params_default_and_overrides():
    assert _format_params({}, True) == "default"
    assert _format_params({"C": 1.0, "max_iter": 500}, False) == "C=1.0, max_iter=500"


def test_grid_record_captures_swept_config():
    df_train = make_df(200, seed=11)
    df_val = make_df(50, seed=12)
    rec = run_grid_search(df_train, df_val, grids={"logistic-regression": {}})[0]
    record = grid_record(rec, GRIDS, "sha-train", "sha-val")
    assert record["name"] == "logistic-regression"
    assert record["input_files"]["train_sha256"] == "sha-train"
    assert record["input_files"]["validation_sha256"] == "sha-val"
    assert record["seed"] == rec.seed
    assert "auc_pr" in record["metrics"]


def test_run_grid_search_deterministic():
    df_train = make_df(250, seed=13)
    df_val = make_df(60, seed=14)
    grids = {"logistic-regression": {"C": [0.01, 1.0]}}
    first = run_grid_search(df_train, df_val, grids=grids)
    second = run_grid_search(df_train, df_val, grids=grids)
    assert [(rec.name, rec.params, rec.metrics) for rec in first] == [
        (rec.name, rec.params, rec.metrics) for rec in second
    ]


def test_unknown_candidate_in_grid_raises():
    df_train = make_df(250, seed=15)
    df_val = make_df(60, seed=16)
    with pytest.raises(ValueError):
        run_grid_search(df_train, df_val, grids={"no-existe": {}})
