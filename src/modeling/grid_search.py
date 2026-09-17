"""Búsqueda exploratoria de hiperparámetros y candidatos nuevos.

Barre una rejilla de configuraciones para cada familia de modelo registrada en
`src.modeling.train.CANDIDATES` (regresión logística, random forest y
xgboost) reutilizando `train_candidate` de `T-13`: misma partición `train`,
mismo preprocessing (ajustado solo en train), mismas métricas y misma semilla.

Garantías:
- La partición `test` permanece reservada: toda la búsqueda se evalúa solo
  sobre `validation` (una sola familia asistida; `test` se usa únicamente en
  la selección final de `T-14`).
- Sin data leakage: el pipeline de preprocessing y cada modelo se ajustan
  únicamente con `train`.
- Incluye la configuración por defecto de cada familia (fila "default") para
  comparar los resultados contra los candidatos oficiales de `T-13`.

El objetivo es explorar si alguna configuración supera el candidato actual
antes de proponer cambios oficiales (registro en tracking `T-17` y selección
`T-14`). Los resultados se escriben en `reports/grid_search.json` y se
resumen en `reports/grid_search.md`.

Ejecución:
    python -m src.modeling.grid_search
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd

from src.hashing import sha256_file
from src.modeling.train import ExperimentResult, train_candidate
from src.paths import (
    GRID_SEARCH_JSON,
    GRID_SEARCH_MD,
    PROJECT_ROOT,
    TRAIN_FILE,
    VAL_FILE,
)
from src.seeds import RANDOM_SEED

# Rejilla por familia: cada familia se barre con su configuración por defecto
# (fila "default") más el producto cartesiano de los valores siguientes.
GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic-regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
    },
    "random-forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [3, 5],
    },
    "xgboost": {
        "n_estimators": [200, 400],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
}

PRIMARY_METRIC = "auc_pr"
TIEBREAK_METRIC = "recall_pos"


@dataclass(frozen=True)
class GridResult:
    name: str
    params: dict[str, Any]
    metrics: dict[str, object]
    train_size: int
    validation_size: int
    seed: int
    is_default: bool = False


def combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Producto cartesiano de los valores de cada hiperparámetro."""
    keys = list(grid)
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def _metric(metrics: dict[str, object], key: str) -> float:
    """Devuelve el valor numérico de una métrica (real o numpy escalar)."""
    return float(cast(float, metrics[key]))


def run_grid_search(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    grids: dict[str, dict[str, list[Any]]] | None = None,
    seed: int = RANDOM_SEED,
) -> list[GridResult]:
    """Entrena cada combinación de ``grids`` y devuelve los resultados ordenados.

    Cada familia incluye siempre la fila "default" (sin overrides, es decir, la
    configuración oficial de `T-13`). El orden final es descendente por
    `auc_pr` y, en caso de empate, por `recall_pos` (regla de selección de
    `src.modeling.select`).
    """
    if grids is None:
        grids = GRIDS
    records: list[GridResult] = []
    for name, grid in grids.items():
        records.append(
            _grid_result(train_candidate(name, df_train, df_val, seed=seed), True)
        )
        for params in combinations(grid):
            result = train_candidate(name, df_train, df_val, seed=seed, **params)
            records.append(_grid_result(result, False, params))
    records.sort(
        key=lambda r: (
            _metric(r.metrics, PRIMARY_METRIC),
            _metric(r.metrics, TIEBREAK_METRIC),
        ),
        reverse=True,
    )
    return records


def _grid_result(
    result: ExperimentResult, is_default: bool, params: dict[str, Any] | None = None
) -> GridResult:
    """Convierte el resultado de `train_candidate` en `GridResult`."""
    return GridResult(
        name=result.name,
        params={} if params is None else params,
        metrics=result.metrics,
        train_size=result.train_size,
        validation_size=result.validation_size,
        seed=result.seed,
        is_default=is_default,
    )


def _format_params(params: dict[str, Any], is_default: bool) -> str:
    if is_default:
        return "default"
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def best_per_family(records: list[GridResult]) -> dict[str, GridResult]:
    """Mejor resultado (AUC-PR, luego recall) de cada familia."""
    best: dict[str, GridResult] = {}
    for record in records:
        prev = best.get(record.name)
        if prev is None or _better(record, prev):
            best[record.name] = record
    return best


def _better(candidate: GridResult, current: GridResult) -> bool:
    return (
        _metric(candidate.metrics, PRIMARY_METRIC),
        _metric(candidate.metrics, TIEBREAK_METRIC),
    ) > (
        _metric(current.metrics, PRIMARY_METRIC),
        _metric(current.metrics, TIEBREAK_METRIC),
    )


def build_report(records: list[GridResult]) -> str:
    lines = [
        "# Búsqueda de hiperparámetros (exploratoria)",
        "",
        "Barrido de configuraciones por familia de modelo sobre la partición "
        "`validation` (mismo preprocessing y métricas que `T-13`; `test` queda "
        "reservado para `T-14`). Cada familia incluye su fila `default` "
        "(configuración oficial de `T-13`) como referencia. Ordenada por "
        "AUC-PR (desempate: recall).",
        "",
        "| Modelo | Parámetros | AUC-ROC | AUC-PR | Recall | Precisión | F1 | "
        "Accuracy (ref.) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        m = rec.metrics
        lines.append(
            f"| {rec.name} | {_format_params(rec.params, rec.is_default)} | "
            f"{m['auc_roc']:.4f} | {m['auc_pr']:.4f} | {m['recall_pos']:.4f} | "
            f"{m['precision_pos']:.4f} | {m['f1_pos']:.4f} | {m['accuracy']:.4f} |"
        )
    lines += ["", "Mejor configuración por familia:", ""]
    for name, best in best_per_family(records).items():
        m = best.metrics
        lines.append(
            f"- **{name}**: {_format_params(best.params, best.is_default)} → "
            f"AUC-PR {m['auc_pr']:.4f}, recall {m['recall_pos']:.4f}."
        )
    overall = records[0]
    m = overall.metrics
    lines += [
        "",
        f"Candidato global: **{overall.name}** con "
        f"{_format_params(overall.params, overall.is_default)} → "
        f"AUC-PR {m['auc_pr']:.4f}, recall {m['recall_pos']:.4f}. "
        f"Referencia de producción: logistic-regression (default) "
        f"AUC-PR 0.6686, recall 0.8043 (selección T-14).",
        "",
        "Detalle completo (parámetros, métricas, semillas y hashes de datos) en "
        "`reports/grid_search.json`. La adopción de una configuración ganadora "
        "implicaría proponer la actualización de `docs/tasks.md` y reejecutar "
        "`T-17`/`T-14`.",
        "",
    ]
    return "\n".join(lines)


def grid_record(
    result: GridResult,
    grids: dict[str, dict[str, list[Any]]],
    train_sha: str,
    val_sha: str,
) -> dict[str, object]:
    return {
        "name": result.name,
        "params": result.params,
        "is_default": result.is_default,
        "metrics": result.metrics,
        "grid": grids.get(result.name, {}),
        "train_size": result.train_size,
        "validation_size": result.validation_size,
        "seed": result.seed,
        "input_files": {
            "train": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
            "train_sha256": train_sha,
            "validation": str(VAL_FILE.relative_to(PROJECT_ROOT)),
            "validation_sha256": val_sha,
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)

    records = run_grid_search(df_train, df_val)
    payload = {
        "task": "grid-search-exploratorio",
        "seed": RANDOM_SEED,
        "grids": GRIDS,
        "leakage_note": (
            "Preprocessing y modelos ajustados únicamente con train; "
            "validation solo se evalua; test reservado para T-14."
        ),
        "results": [
            grid_record(rec, GRIDS, sha256_file(TRAIN_FILE), sha256_file(VAL_FILE))
            for rec in records
        ],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    GRID_SEARCH_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    GRID_SEARCH_MD.write_text(build_report(records), encoding="utf-8")
    best = records[0]
    print(
        f"Mejor configuracion: {best.name} "
        f"(AUC-PR={best.metrics[PRIMARY_METRIC]:.4f}, "
        f"recall={best.metrics[TIEBREAK_METRIC]:.4f})"
    )
    print(f"Reporte: {GRID_SEARCH_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
