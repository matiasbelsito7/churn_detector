"""Evaluación y selección del modelo sobre la partición de test (T-14).

Compara los candidatos de `T-13` (métricas sobre `validation`, ya registradas
en MLflow por `T-17`), selecciona el ganador según una regla documentada y
evalúa **una sola vez** sobre la partición `test` (reservada desde `T-11`).

Regla de selección (alineada con `docs/specs.md` §7 y `docs/constitution.md`
§5):
- Métrica primaria: **AUC-PR** sobre la clase de interés `Yes` (independiente
  del umbral y sensible a la prevalencia del problema desbalanceado).
- Desempate: **recall** sobre `Yes` (prioriza no perder clientes en churn).

El modelo seleccionado se carga desde el *Model Registry* de MLflow (artefacto
publicable para inferencia), se evalúa sobre `test`, se promueve con el alias
`production` y la corrida de evaluación queda registrada en el tracking.

Ejecución:
    python -m src.modeling.select
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.contract import TARGET
from src.hashing import sha256_file
from src.modeling.baseline import evaluate_metrics
from src.modeling.preprocessing import FEATURE_NAMES
from src.modeling.tracking import (
    EXPERIMENT_NAME,
    METRIC_KEYS,
    MODEL_PREFIX,
    TRACKING_URI,
    _as_float,
)
from src.modeling.train import prob_yes
from src.paths import (
    EXPERIMENTS_JSON as EXPERIMENTS_FILE,
)
from src.paths import (
    MODELS_DIR,
    PROJECT_ROOT,
    SELECTION_JSON,
    SELECTION_MD,
    TEST_FILE,
)

PRIMARY_METRIC = "auc_pr"
TIEBREAK_METRIC = "recall_pos"

SELECTION_RULE = (
    "Métrica primaria: AUC-PR sobre la clase `Yes`; desempate por recall sobre "
    "`Yes`. Métricas adecuadas al problema desbalanceado (specs.md §7)."
)


def select_candidate(records: list[dict[str, Any]]) -> str:
    """Devuelve el candidato ganador según `SELECTION_RULE`."""
    best = max(
        records,
        key=lambda rec: (
            rec["metrics"][PRIMARY_METRIC],
            rec["metrics"][TIEBREAK_METRIC],
        ),
    )
    return best["name"]


def load_selected_model(name: str, tracking_uri: str = TRACKING_URI) -> Pipeline:
    """Carga del Model Registry el pipeline del modelo con alias `production`."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{MODEL_PREFIX}{name}@production")


def promote_model(name: str, tracking_uri: str = TRACKING_URI) -> int:
    """Promueve la última versión registrada de ``name`` al alias `production`."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_PREFIX}{name}'")
    if not versions:
        raise ValueError(
            f"No hay versiones registradas para el modelo {MODEL_PREFIX}{name}"
        )
    version = max(versions, key=lambda v: int(v.version)).version
    client.set_registered_model_alias(MODEL_PREFIX + name, "production", version)
    return int(version)


def log_selection_run(
    name: str,
    version: int,
    metrics: dict[str, object],
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Registra en el tracking la corrida de evaluación del modelo seleccionado."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"seleccion-{name}"):
        mlflow.log_params(
            {
                "selected_candidate": name,
                "selection_rule": SELECTION_RULE,
                "registered_version": version,
            }
        )
        mlflow.log_metrics({key: _as_float(metrics[key]) for key in METRIC_KEYS})
        mlflow.log_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            "test_metrics.json",
        )
        run = mlflow.active_run()
        assert run is not None
        return str(run.info.run_id)


def build_selection_report(
    records: list[dict[str, Any]],
    selected: str,
    test_metrics: dict[str, object],
    test_size: int,
    test_sha: str,
) -> str:
    lines = [
        "# Selección del modelo (T-14)",
        "",
        "Procedimiento: los candidatos de `T-13` se comparan con las mismas "
        "métricas sobre `validation` (registradas en el tracking de MLflow por "
        f"`T-17`). El ganador se carga desde el Model Registry como "
        f"`{MODEL_PREFIX}{selected}/production`, se evalúa sobre `test` "
        "(partición usada una sola vez) y la corrida queda registrada.",
        "",
        "| Modelo | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Accuracy (ref.) |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        m = rec["metrics"]
        lines.append(
            f"| {rec['name']} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} | "
            f"{m['recall_pos']:.4f} | {m['precision_pos']:.4f} | "
            f"{m['f1_pos']:.4f} | {m['accuracy']:.4f} |"
        )
    lines += [
        "",
        f"Candidato seleccionado: **{selected}**.",
        "",
        "Justificación: " + SELECTION_RULE,
        "",
        f"Evaluación del seleccionado sobre `test` ({test_size} filas, "
        f"sha256 `{test_sha}`):",
        "",
        "| Métrica | Valor (clase `Yes`) |",
        "|---|---|",
        f"| AUC-ROC | {test_metrics['auc_roc']:.4f} |",
        f"| AUC-PR | {test_metrics['auc_pr']:.4f} |",
        f"| Recall | {test_metrics['recall_pos']:.4f} |",
        f"| Precisión | {test_metrics['precision_pos']:.4f} |",
        f"| F1 | {test_metrics['f1_pos']:.4f} |",
        f"| Accuracy (referencia) | {test_metrics['accuracy']:.4f} |",
        "",
        "Matriz de confusión (umbral 0.5):",
        "",
        f"- TP={test_metrics['true_positive']}, "
        f"FN={test_metrics['false_negative']}, "
        f"FP={test_metrics['false_positive']}, "
        f"TN={test_metrics['true_negative']}.",
        "",
        "Auditabilidad: experimento `churn-detector` en el tracking de MLflow "
        "(candidatos con su configuración, métricas y artefactos); modelo "
        f"`{MODEL_PREFIX}{selected}` con alias `production`. El registro en "
        "`reports/experiments.json` conserva la trazabilidad de datos.",
        "",
    ]
    return "\n".join(lines)


def export_model(pipeline: Pipeline, name: str) -> Path:
    """Exporta el pipeline a un archivo joblib standalone para despliegue."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{MODEL_PREFIX}{name}.joblib"
    joblib.dump(pipeline, path)
    return path


def main() -> int:
    payload = json.loads(EXPERIMENTS_FILE.read_text(encoding="utf-8"))
    records = payload["experiments"]
    selected = select_candidate(records)

    df_test = pd.read_csv(TEST_FILE)
    version = promote_model(selected)
    pipeline = load_selected_model(selected)
    model_file = export_model(pipeline, selected)
    prob = prob_yes(pipeline, df_test[list(FEATURE_NAMES)])
    test_metrics = evaluate_metrics(df_test[TARGET], prob)

    log_selection_run(selected, version, test_metrics)

    test_sha = sha256_file(TEST_FILE)
    SELECTION_JSON.write_text(
        json.dumps(
            {
                "task": "T-14",
                "selection_rule": SELECTION_RULE,
                "selected": selected,
                "version": version,
                "validation": {rec["name"]: rec["metrics"] for rec in records},
                "test_file": {
                    "path": str(TEST_FILE.relative_to(PROJECT_ROOT)),
                    "sha256": test_sha,
                    "size": int(len(df_test)),
                },
                "test_metrics": test_metrics,
                "registry": {
                    "model": f"{MODEL_PREFIX}{selected}",
                    "alias": "production",
                    "experiment": EXPERIMENT_NAME,
                    "tracking_uri": TRACKING_URI,
                },
                "artifact": {
                    "path": str(model_file.relative_to(PROJECT_ROOT)),
                    "standalone": True,
                },
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    SELECTION_MD.write_text(
        build_selection_report(records, selected, test_metrics, len(df_test), test_sha),
        encoding="utf-8",
    )
    print(f"Seleccionado: {selected} (test AUC-PR={test_metrics['auc_pr']:.4f})")
    print(f"Informe: {SELECTION_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
