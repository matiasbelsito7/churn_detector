"""Análisis exploratorio de hiperparámetros para RF, LR y XGB.

Combina dos estrategias sobre la partición `validation` (mismo preprocessing,
ajustado solo en train, y mismas métricas del problema que `T-13`):

- **Barrido univariado**: para cada hiperparámetro de la familia, variar sus
  valores candidatos con el resto fijado en la configuración por defecto de
  `T-13`. Muestra el efecto individual de cada parámetro.
- **Búsqueda aleatoria conjunta**: `N` configuraciones muestreadas de forma
  determinista (semilla fija) desde el espacio declarado de cada familia, para
  capturar interacciones entre parámetros con costo acotado.

Garantías:
- `test` permanece reservado para `T-14` (solo se evalua sobre `validation`).
- Sin data leakage: cada pipeline/modelo se ajusta únicamente con `train`.
- Reproducible: semilla fija (`RANDOM_SEED`) para el muestreo aleatorio.

Resultados en `reports/hyperparams_exploration.json` y resumen en
`reports/hyperparams_exploration.md`.

Ejecución:
    python -m src.modeling.explore_hyperparams
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from src.hashing import sha256_file
from src.modeling.train import train_candidate
from src.paths import (
    EXPLORE_HP_JSON,
    EXPLORE_HP_MD,
    PROJECT_ROOT,
    TRAIN_FILE,
    VAL_FILE,
)
from src.seeds import RANDOM_SEED

# Valores candidatos por hiperparámetro y familia (espacio declarado).
# Se usan tanto para el barrido univariado como para el muestreo aleatorio.
UNIVARIATE_VALUES: dict[str, dict[str, list[Any]]] = {
    "logistic-regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "l1_ratio": [0.0, 0.5, 1.0],
        "class_weight": ["balanced", None],
    },
    "random-forest": {
        "n_estimators": [50, 100, 200, 300, 400, 600],
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", None],
        "criterion": ["gini", "entropy"],
        "bootstrap": [True, False],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200, 300, 400, 600],
        "max_depth": [2, 3, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.075, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.3, 0.5],
        "reg_lambda": [0.1, 1, 10],
        "scale_pos_weight": [1.0, 1.5, 2.0, 2.5, None],
    },
}

# Presupuesto de configuraciones muestreadas por familia (búsqueda aleatoria).
RANDOM_BUDGET: dict[str, int] = {
    "logistic-regression": 20,
    "random-forest": 40,
    "xgboost": 40,
}

PRIMARY_METRIC = "auc_pr"
TIEBREAK_METRIC = "recall_pos"

DEFAULT_KIND = "default"
UNIVARIATE_KIND = "univariate"
RANDOM_KIND = "random"

EXPLORED_FAMILIES = tuple(UNIVARIATE_VALUES)


def normalize_combo(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Completa parámetros dependientes (p. ej. solver según regularización LR).

    En sklearn>=1.9 `penalty` está deprecado a favor de `l1_ratio`
    (`0` -> l2, `1` -> l1); con `l1_ratio` se usa el solver `saga`, que lo
    soporta sin warnings.
    """
    if name == "logistic-regression" and "l1_ratio" in params:
        return {**params, "solver": "saga"}
    return params


@dataclass(frozen=True)
class ExploreResult:
    family: str
    kind: str
    param: str | None
    value: Any
    params: dict[str, Any]
    metrics: dict[str, object]


def _metric(metrics: dict[str, object], key: str) -> float:
    """Devuelve el valor numérico de una métrica (real o numpy escalar)."""
    return float(cast(float, metrics[key]))


def _sort_key(record: ExploreResult) -> tuple[float, float]:
    return (
        _metric(record.metrics, PRIMARY_METRIC),
        _metric(record.metrics, TIEBREAK_METRIC),
    )


def run_default(
    name: str, df_train: pd.DataFrame, df_val: pd.DataFrame, seed: int
) -> ExploreResult:
    """Configuración por defecto de la familia (referencia `T-13`)."""
    result = train_candidate(name, df_train, df_val, seed=seed)
    return ExploreResult(name, DEFAULT_KIND, None, None, {}, result.metrics)


def univariate_sweep(
    name: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    values: dict[str, list[Any]] | None = None,
    seed: int = RANDOM_SEED,
) -> list[ExploreResult]:
    """Barre un hiperparámetro a la vez con el resto fijado en default."""
    if values is None:
        values = UNIVARIATE_VALUES[name]
    records: list[ExploreResult] = []
    for param, options in values.items():
        for value in options:
            combo = normalize_combo(name, {param: value})
            result = train_candidate(name, df_train, df_val, seed=seed, **combo)
            records.append(
                ExploreResult(
                    name, UNIVARIATE_KIND, param, value, combo, result.metrics
                )
            )
    return records


def random_search(
    name: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    budget: int,
    spec: dict[str, list[Any]] | None = None,
    seed: int = RANDOM_SEED,
) -> list[ExploreResult]:
    """Muestra ``budget`` configuraciones aleatorias (deterministas) del espacio."""
    if spec is None:
        spec = UNIVARIATE_VALUES[name]
    rng = np.random.RandomState(seed)
    seen: set[tuple[tuple[str, str], ...]] = set()
    records: list[ExploreResult] = []
    attempts = 0
    while len(records) < budget and attempts < budget * 20:
        attempts += 1
        combo: dict[str, Any] = {
            param: options[int(rng.randint(len(options)))]
            for param, options in spec.items()
        }
        key = tuple(sorted((k, repr(v)) for k, v in combo.items()))
        if key in seen:
            continue
        seen.add(key)
        combo = normalize_combo(name, combo)
        result = train_candidate(name, df_train, df_val, seed=seed, **combo)
        records.append(
            ExploreResult(name, RANDOM_KIND, None, None, combo, result.metrics)
        )
    return records


def _format_combo(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _format_value(value: Any) -> str:
    return "None" if value is None else str(value)


def _metrics_row(record: ExploreResult) -> str:
    m = record.metrics
    return (
        f"{m['auc_pr']:.4f} | {m['recall_pos']:.4f} | {m['precision_pos']:.4f} "
        f"| {m['f1_pos']:.4f} | {m['accuracy']:.4f}"
    )


def best(records: list[ExploreResult]) -> ExploreResult | None:
    return max(records, key=_sort_key) if records else None


def _brief(record: ExploreResult | None) -> str:
    """AUC-PR/recall formateados para el resumen (o `-` si no hay resultado)."""
    if record is None:
        return "-"
    m = record.metrics
    return f"{_metric(m, PRIMARY_METRIC):.4f}/{_metric(m, TIEBREAK_METRIC):.4f}"


def build_report(
    defaults: dict[str, ExploreResult],
    univariate: dict[str, list[ExploreResult]],
    random: dict[str, list[ExploreResult]],
) -> str:
    lines = [
        "# Análisis exploratorio de hiperparámetros",
        "",
        "Método sobre `validation` (mismo preprocessing y métricas que `T-13`; "
        "`test` reservado para `T-14`): barrido univariado (un hiperparámetro "
        "por vez, resto en default) y búsqueda aleatoria conjunta con semilla "
        f"fija `{RANDOM_SEED}`.",
        "",
    ]
    for name in EXPLORED_FAMILIES:
        default = defaults[name]
        uni = univariate[name]
        rand = random[name]
        lines += [
            f"## {name}",
            "",
            f"Default: AUC-PR {_metric(default.metrics, PRIMARY_METRIC):.4f}, "
            f"recall {_metric(default.metrics, TIEBREAK_METRIC):.4f}.",
            "",
            "### Univariado (un parámetro a la vez)",
            "",
        ]
        for param in UNIVARIATE_VALUES[name]:
            rows = [r for r in uni if r.param == param]
            if not rows:
                continue
            best_row = best(rows)
            assert best_row is not None
            parts = [
                f"{_format_value(r.value):>8} -> "
                f"APR {_metric(r.metrics, PRIMARY_METRIC):.4f} / "
                f"rec {_metric(r.metrics, TIEBREAK_METRIC):.4f}"
                for r in rows
            ]
            lines.append(f"- `{param}`: " + "; ".join(parts))
            lines.append(
                f"  Mejor: {_format_value(best_row.value)} "
                f"(APR {_metric(best_row.metrics, PRIMARY_METRIC):.4f}, "
                f"rec {_metric(best_row.metrics, TIEBREAK_METRIC):.4f})."
            )
        lines += [
            "",
            "### Búsqueda aleatoria conjunta",
            "",
            "| Parámetros | " "AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |",
            "|---|---|---|---|---|---|",
        ]
        for rec in sorted(rand, key=_sort_key, reverse=True):
            lines.append(f"| {_format_combo(rec.params)} | {_metrics_row(rec)} |")
        lines.append("")
    lines += [
        "## Resumen",
        "",
        "| Familia | Default APR/rec | Mejor univariado | Mejor aleatorio |",
        "|---|---|---|---|",
    ]
    for name in EXPLORED_FAMILIES:
        d = defaults[name]
        lines.append(
            f"| {name} | {_brief(d)} | {_brief(best(univariate[name]))} | "
            f"{_brief(best(random[name]))} |"
        )
    all_records = [
        r
        for family in EXPLORED_FAMILIES
        for r in [defaults[family]] + univariate[family] + random[family]
    ]
    overall = best(all_records)
    if overall is not None:
        m = overall.metrics
        lines += [
            "",
            f"Mejor configuración global: **{overall.family}** "
            f"[{overall.kind}] {_format_combo(overall.params)} -> "
            f"AUC-PR {_metric(m, PRIMARY_METRIC):.4f}, "
            f"recall {_metric(m, TIEBREAK_METRIC):.4f}. "
            "Referencia de producción: logistic-regression (default) "
            "AUC-PR 0.6686, recall 0.8043 (selección T-14).",
            "",
            "Detalle completo en `reports/hyperparams_exploration.json`. "
            "La adopción de una configuración ganadora implicaría proponer la "
            "actualización de `docs/tasks.md` y reejecutar `T-17`/`T-14`.",
            "",
        ]
    return "\n".join(lines)


def explore_record(
    record: ExploreResult, train_sha: str, val_sha: str
) -> dict[str, object]:
    return {
        "family": record.family,
        "kind": record.kind,
        "param": record.param,
        "value": record.value,
        "params": record.params,
        "metrics": record.metrics,
        "input_files": {
            "train": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
            "train_sha256": train_sha,
            "validation": str(VAL_FILE.relative_to(PROJECT_ROOT)),
            "validation_sha256": val_sha,
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def build_payload(
    defaults: dict[str, ExploreResult],
    univariate: dict[str, list[ExploreResult]],
    random: dict[str, list[ExploreResult]],
    train_sha: str,
    val_sha: str,
) -> dict[str, object]:
    return {
        "task": "hyperparams-exploration",
        "seed": RANDOM_SEED,
        "family_space": UNIVARIATE_VALUES,
        "random_budget": RANDOM_BUDGET,
        "leakage_note": (
            "Preprocessing y modelos ajustados únicamente con train; "
            "validation solo se evalua; test reservado para T-14."
        ),
        "families": {
            name: [
                explore_record(rec, train_sha, val_sha)
                for rec in [defaults[name]] + univariate[name] + random[name]
            ]
            for name in EXPLORED_FAMILIES
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)

    defaults: dict[str, ExploreResult] = {}
    univariate: dict[str, list[ExploreResult]] = {}
    random: dict[str, list[ExploreResult]] = {}

    for name in EXPLORED_FAMILIES:
        default = run_default(name, df_train, df_val, RANDOM_SEED)
        uni = univariate_sweep(name, df_train, df_val)
        rand = random_search(name, df_train, df_val, RANDOM_BUDGET[name])
        defaults[name] = default
        univariate[name] = uni
        random[name] = rand
        best_uni = best(uni)
        best_rand = best(rand)
        uni_desc = (
            f"{_metric(best_uni.metrics, PRIMARY_METRIC):.4f} "
            f"({best_uni.param}={best_uni.value})"
            if best_uni is not None
            else "-"
        )
        rand_desc = (
            f"{_metric(best_rand.metrics, PRIMARY_METRIC):.4f}"
            if best_rand is not None
            else "-"
        )
        print(
            f"{name}: default "
            f"APR={_metric(default.metrics, PRIMARY_METRIC):.4f}; "
            f"mejor univariado {uni_desc}; mejor aleatorio {rand_desc}"
        )

    train_sha = sha256_file(TRAIN_FILE)
    val_sha = sha256_file(VAL_FILE)
    payload = build_payload(defaults, univariate, random, train_sha, val_sha)
    EXPLORE_HP_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    EXPLORE_HP_MD.write_text(
        build_report(defaults, univariate, random), encoding="utf-8"
    )
    all_records = [
        r
        for name in EXPLORED_FAMILIES
        for r in [defaults[name]] + univariate[name] + random[name]
    ]
    overall = best(all_records)
    assert overall is not None
    print(
        f"Mejor configuracion global: {overall.family} [{overall.kind}] "
        f"(AUC-PR={_metric(overall.metrics, PRIMARY_METRIC):.4f})"
    )
    print(f"Reporte: {EXPLORE_HP_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
