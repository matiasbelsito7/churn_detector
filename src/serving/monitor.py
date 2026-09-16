"""Monitoreo del sistema de predicción de churn (T-22).

Compara la distribución de features, de predicciones y el desempeño del modelo
entre una **referencia** (población de entrenamiento y batch de predicción de
despliegue) y un **batch corriente**, y produce un reporte de estado con
chequeos clasificados como `ok` / `warn` / `alert` según los umbrales definidos
en `docs/monitoring.md`.

Chequeos:

- **Drift de features**: PSI por feature numérica y desplazamiento máximo de
  proporciones por feature categórica (referencia: features de `train`;
  corriente: batch de features en servicio).
- **Distribución de predicciones**: tasa de churn predicha y PSI de la
  probabilidad (referencia: batch de predicción de `T-15`).
- **Drift del target**: prevalencia de `Churn == "Yes"` cuando el batch
  corriente incluye el target.
- **Desempeño**: métricas del problema desbalanceado sobre la clase `Yes`
  (AUC-PR, recall) comparadas con la evaluación de la referencia
  (`reports/selection.json`, `test`).

Ejecución:

    python -m src.serving.monitor

Configuración por variables de entorno (por defecto usa los artefactos
versionados del repositorio): `MONITOR_REFERENCE_FEATURES`,
`MONITOR_CURRENT_FEATURES`, `MONITOR_REFERENCE_PREDICTIONS`,
`MONITOR_CURRENT_PREDICTIONS`, `MONITOR_REFERENCE_METRICS`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.data.contract import TARGET
from src.hashing import sha256_file
from src.modeling.baseline import evaluate_metrics
from src.modeling.preprocessing import CATEGORIC_FEATURE_NAMES, NUMERIC_FEATURE_NAMES
from src.paths import (
    FEATURES_FILE,
    PREDICTIONS_FILE,
    PROJECT_ROOT,
    TRAIN_FILE,
)
from src.paths import (
    MONITORING_REPORT_JSON as REPORT_JSON,
)
from src.paths import (
    MONITORING_REPORT_MD as REPORT_MD,
)
from src.paths import (
    SELECTION_JSON as SELECTION_FILE,
)

MONITOR_REFERENCE_FEATURES = os.environ.get("MONITOR_REFERENCE_FEATURES", "")
MONITOR_CURRENT_FEATURES = os.environ.get(
    "MONITOR_CURRENT_FEATURES", str(FEATURES_FILE)
)
MONITOR_REFERENCE_PREDICTIONS = os.environ.get(
    "MONITOR_REFERENCE_PREDICTIONS", str(PREDICTIONS_FILE)
)
MONITOR_CURRENT_PREDICTIONS = os.environ.get(
    "MONITOR_CURRENT_PREDICTIONS", str(PREDICTIONS_FILE)
)
MONITOR_REFERENCE_METRICS = os.environ.get(
    "MONITOR_REFERENCE_METRICS", str(SELECTION_FILE)
)

PSI_WARN = 0.10
PSI_ALERT = 0.25
CAT_WARN = 0.05
CAT_ALERT = 0.10
RATE_WARN = 0.03
RATE_ALERT = 0.08
METRIC_DROP_WARN = 0.05
METRIC_DROP_ALERT = 0.10
STATUS_RANK = {"ok": 0, "warn": 1, "alert": 2}

POSITIVE = "Yes"


def classify(value: float, warn: float, alert: float) -> str:
    """Clasifica un chequeo: ``ok`` / ``warn`` / ``alert``."""
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre dos distribuciones.

    Bin edges desde los cuantiles de la referencia; ``eps`` evita log(0) en
    bins vacíos. Devuelve 0 si no hay desplazamiento y un valor alto si la
    distribución cambió de forma sustancial.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        same_mass = float(np.unique(ref)[0]) == float(np.unique(cur)[0])
        return 0.0 if same_mass else 0.5
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    ref_pct = ref_hist / ref_hist.sum() + eps
    cur_pct = cur_hist / cur_hist.sum() + eps
    return float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))


def categorical_max_shift(reference: pd.Series, current: pd.Series) -> float:
    """Máximo desplazamiento absoluto de proporciones por categoría."""
    ref_prop = reference.value_counts(normalize=True)
    cur_prop = current.value_counts(normalize=True)
    categories = set(ref_prop.index) | set(cur_prop.index)
    return float(
        max(abs(ref_prop.get(cat, 0.0) - cur_prop.get(cat, 0.0)) for cat in categories)
    )


def feature_checks(ref: pd.DataFrame, cur: pd.DataFrame) -> list[dict[str, object]]:
    """Chequeos de drift por feature numérica (PSI) y categórica (shift)."""
    checks: list[dict[str, object]] = []
    for name in NUMERIC_FEATURE_NAMES:
        value = psi(ref[name], cur[name])
        checks.append(
            {
                "name": name,
                "metric": "psi",
                "value": value,
                "status": classify(value, PSI_WARN, PSI_ALERT),
            }
        )
    for name in CATEGORIC_FEATURE_NAMES:
        value = categorical_max_shift(ref[name], cur[name])
        checks.append(
            {
                "name": name,
                "metric": "categorical_max_shift",
                "value": value,
                "status": classify(value, CAT_WARN, CAT_ALERT),
            }
        )
    return checks


def prediction_checks(
    ref_pred: pd.DataFrame, cur_pred: pd.DataFrame
) -> list[dict[str, object]]:
    """Chequeos de la distribución de predicciones (tasa y probabilidad)."""
    ref_rate = float((ref_pred["churn_class"] == POSITIVE).mean())
    cur_rate = float((cur_pred["churn_class"] == POSITIVE).mean())
    rate_delta = abs(cur_rate - ref_rate)
    prob_psi = psi(ref_pred["churn_prob"], cur_pred["churn_prob"])
    return [
        {
            "name": "churn_rate",
            "metric": "abs_delta",
            "value": rate_delta,
            "reference": ref_rate,
            "current": cur_rate,
            "status": classify(rate_delta, RATE_WARN, RATE_ALERT),
        },
        {
            "name": "churn_prob",
            "metric": "psi",
            "value": prob_psi,
            "status": classify(prob_psi, PSI_WARN, PSI_ALERT),
        },
    ]


def target_checks(
    ref_df: pd.DataFrame, cur_df: pd.DataFrame
) -> list[dict[str, object]]:
    """Drift del target: desplazamiento de la prevalencia observada."""
    if TARGET not in cur_df.columns:
        return []
    ref_prevalence = float((ref_df[TARGET] == POSITIVE).mean())
    cur_prevalence = float((cur_df[TARGET] == POSITIVE).mean())
    delta = abs(cur_prevalence - ref_prevalence)
    return [
        {
            "name": "target_prevalence",
            "metric": "abs_delta",
            "value": delta,
            "reference": ref_prevalence,
            "current": cur_prevalence,
            "status": classify(delta, RATE_WARN, RATE_ALERT),
        }
    ]


def performance_checks(
    cur_features: pd.DataFrame,
    cur_pred: pd.DataFrame,
    reference_metrics: dict[str, float],
) -> list[dict[str, object]]:
    """Desempeño del batch corriente cuando el target está disponible."""
    if TARGET not in cur_features.columns:
        return []
    merged = cur_features[["customerID", TARGET]].merge(
        cur_pred[["customerID", "churn_prob"]], on="customerID"
    )
    if merged.empty:
        return []
    metrics = evaluate_metrics(merged[TARGET], merged["churn_prob"].to_numpy("float"))
    checks: list[dict[str, object]] = []
    for name, ref_value in (
        ("recall_pos", reference_metrics["recall_pos"]),
        ("auc_pr", reference_metrics["auc_pr"]),
    ):
        current_value = float(cast(float, metrics[name]))
        drop = ref_value - current_value
        checks.append(
            {
                "name": name,
                "metric": "drop_vs_reference",
                "value": drop,
                "reference": ref_value,
                "current": current_value,
                "status": classify(drop, METRIC_DROP_WARN, METRIC_DROP_ALERT),
            }
        )
    return checks


def overall_status(checks: list[dict[str, object]]) -> str:
    """Estado general: el peor de todos los chequeos."""
    statuses = [str(check["status"]) for check in checks]
    worst = "ok"
    for status in statuses:
        if STATUS_RANK[status] > STATUS_RANK[worst]:
            worst = status
    return worst


def load_reference_metrics(path: Path) -> dict[str, float]:
    """Métricas de referencia: evaluación sobre `test` de `selection.json`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: float(value) for name, value in dict(payload["test_metrics"]).items()}


def raw_reference_metrics(path: Path) -> dict[str, object]:
    """Valores originales de `test_metrics` (recuentos sin castear a float)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["test_metrics"])


def trace_file(path: Path | None) -> dict[str, object] | None:
    """Trazabilidad (path, sha256, filas) de un artefacto usado."""
    if path is None or not path.exists():
        return None
    try:
        rel = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "sha256": sha256_file(path),
        "rows": int(sum(1 for _ in path.open(encoding="utf-8")) - 1),
    }


def build_report(
    status: str,
    checks: dict[str, list[dict[str, object]]],
    provenance: dict[str, dict[str, object] | None],
) -> str:
    """Devuelve el reporte Markdown del monitoreo."""
    lines = [
        "# Monitoreo del sistema de predicción de churn (T-22)",
        "",
        f"Estado general: **{status.upper()}**.",
        "",
        "Umbrales y acciones: `docs/monitoring.md`. Chequeos por dominio:",
        "",
    ]
    headers = {
        "features": "Drift de features (referencia = `train`, corriente = batch)",
        "predictions": "Distribución de predicciones",
        "target": "Drift del target",
        "performance": "Desempeño (target disponible)",
    }
    for key, title in headers.items():
        items = checks.get(key, [])
        if not items:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Chequeo | Métrica | Valor | Estado |")
        lines.append("|---|---|---|---|")
        for check in items:
            value = cast(float, check.get("value"))
            lines.append(
                f"| {check['name']} | {check['metric']} "
                f"| {value:.4f} | {check['status']} |"
            )
        lines.append("")
    lines.append("### Trazabilidad")
    lines.append("")
    lines.append("| Artefacto | Origen | Filas | SHA256 |")
    lines.append("|---|---|---|---|")
    for label, trace in provenance.items():
        if trace is None:
            continue
        lines.append(
            f"| {label} | {trace['path']} | {trace['rows']} "
            f"| `{str(trace['sha256'])[:12]}…` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    reference_features_path: Path | None = None
    if MONITOR_REFERENCE_FEATURES:
        reference_features_path = Path(MONITOR_REFERENCE_FEATURES)
        ref_features = pd.read_csv(reference_features_path)
    else:
        features_all = pd.read_csv(FEATURES_FILE)
        train_ids = set(pd.read_csv(TRAIN_FILE, usecols=["customerID"])["customerID"])
        ref_features = features_all[features_all["customerID"].isin(train_ids)]

    current_features = pd.read_csv(Path(MONITOR_CURRENT_FEATURES))
    ref_pred = pd.read_csv(Path(MONITOR_REFERENCE_PREDICTIONS))
    current_pred = pd.read_csv(Path(MONITOR_CURRENT_PREDICTIONS))
    reference_metrics = load_reference_metrics(Path(MONITOR_REFERENCE_METRICS))
    reference_metrics_raw = raw_reference_metrics(Path(MONITOR_REFERENCE_METRICS))

    checks = {
        "features": feature_checks(ref_features, current_features),
        "predictions": prediction_checks(ref_pred, current_pred),
        "target": target_checks(ref_features, current_features),
        "performance": performance_checks(
            current_features, current_pred, reference_metrics
        ),
    }
    all_checks = [check for group in checks.values() for check in group]
    status = overall_status(all_checks)

    provenance = {
        "features_referencia": (
            trace_file(reference_features_path)
            if reference_features_path
            else {
                "path": str(FEATURES_FILE.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(FEATURES_FILE),
                "rows": int(len(ref_features)),
            }
        ),
        "features_corriente": trace_file(Path(MONITOR_CURRENT_FEATURES)),
        "predicciones_referencia": trace_file(Path(MONITOR_REFERENCE_PREDICTIONS)),
        "predicciones_corriente": trace_file(Path(MONITOR_CURRENT_PREDICTIONS)),
    }

    REPORT_JSON.write_text(
        json.dumps(
            {
                "task": "T-22",
                "status": status,
                "checks": checks,
                "checks_count": {
                    "ok": sum(1 for c in all_checks if c["status"] == "ok"),
                    "warn": sum(1 for c in all_checks if c["status"] == "warn"),
                    "alert": sum(1 for c in all_checks if c["status"] == "alert"),
                },
                "reference_metrics": reference_metrics_raw,
                "provenance": provenance,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(build_report(status, checks, provenance), encoding="utf-8")
    try:
        print(f"Monitoreo generado: {REPORT_MD.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"Monitoreo generado: {REPORT_MD}")
    print(f"  Estado general: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
