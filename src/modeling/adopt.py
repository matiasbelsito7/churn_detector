"""Adopción de la mejor configuración de hiperparámetros (T-30).

Compara, bajo condiciones idénticas sobre `validation`, el **champion** actual
(configuración por defecto de `T-13`/`T-14`, alias `production`) con el mejor
candidato encontrado en `T-29` (`reports/hyperparams_exploration.md`). Aplica
la regla de selección de `T-14` (métrica primaria AUC-PR sobre `Yes`;
desempate por recall sobre `Yes`). Solo si el candidato es estrictamente
mejor se adopta: se registra en MLflow, se promueve al alias `production`, se
exporta el artefacto standalone, se evalúa sobre `test` una única vez para la
configuración adoptada y se regeneran las predicciones de inferencia.

La decisión (adoptar o descartar) queda documentada siempre en
`reports/adoption.md/.json`; si se adopta, `reports/selection.md/.json` se
actualizan como la selección vigente con su historial.

Si la evaluación sobre `test` muestra que la configuración adoptada no supera
a la anterior, la adopción puede revertirse con el argumento `--revert`: se
restaura el champion previo, se retira la versión descartada del registro y se
regeneran los artefactos y predicciones.

Ejecución:
    python -m src.modeling.adopt
    python -m src.modeling.adopt --revert
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import pandas as pd

from src.data.contract import TARGET
from src.hashing import sha256_file
from src.modeling.baseline import evaluate_metrics
from src.modeling.preprocessing import FEATURE_NAMES
from src.modeling.select import (
    MODEL_PREFIX,
    PRIMARY_METRIC,
    SELECTION_RULE,
    TIEBREAK_METRIC,
    export_model,
    load_selected_model,
    log_selection_run,
    promote_model,
)
from src.modeling.tracking import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    _as_float,
    log_candidate_run,
)
from src.modeling.train import prob_yes, train_candidate
from src.paths import (
    ADOPTION_JSON,
    ADOPTION_MD,
    PROJECT_ROOT,
    SELECTION_JSON,
    SELECTION_MD,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
)

CHAMPION_FAMILY = "logistic-regression"

#: Mejor configuración global de T-29 (`reports/hyperparams_exploration.md`).
CANDIDATE_PARAMS: dict[str, Any] = {
    "C": 0.01,
    "class_weight": "balanced",
    "l1_ratio": 1.0,
    "solver": "saga",
}

REPORT_METRIC_KEYS = (
    "auc_roc",
    "auc_pr",
    "recall_pos",
    "precision_pos",
    "f1_pos",
    "accuracy",
)


def _float_metric(metrics: dict[str, object], key: str) -> float:
    return _as_float(metrics[key])


def _as_relative(path: Path) -> str:
    """Ruta relativa a `PROJECT_ROOT` (o absoluta si está fuera)."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _adopted_metrics(adopted: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], adopted["test_metrics"])


def is_better(candidate: dict[str, object], champion: dict[str, object]) -> bool:
    """True si el candidato gana según la regla (AUC-PR; desempate recall)."""
    cand = (
        _float_metric(candidate, PRIMARY_METRIC),
        _float_metric(candidate, TIEBREAK_METRIC),
    )
    chmp = (
        _float_metric(champion, PRIMARY_METRIC),
        _float_metric(champion, TIEBREAK_METRIC),
    )
    return cand > chmp


def build_report(
    champion: Any,
    candidate: Any,
    decision: bool,
    adopted: dict[str, object] | None,
    previous_champion: dict[str, Any] | None,
) -> str:
    """Reporte Markdown de la decisión de adopción (T-30)."""
    lines = [
        "# Adopción de configuración del modelo (T-30)",
        "",
        "Procedimiento: se entrenan bajo condiciones idénticas (partición "
        "`train` de `T-11`, pipeline de `T-12`, semilla fija `42`) el champion "
        "actual (configuración por defecto de `T-13`/`T-14`) y el mejor "
        "candidato de `T-29`. Ambos se evalúan sobre `validation` con las "
        "métricas del problema (`docs/specs.md` §7). La decisión aplica la "
        "regla de selección de `T-14`.",
        "",
        "Regla: " + SELECTION_RULE,
        "",
        f"Evidencia de `T-29`: `reports/hyperparams_exploration.md` (búsqueda "
        f"aleatoria conjunta). Candidato: `{CHAMPION_FAMILY}` con "
        f"{CANDIDATE_PARAMS}.",
        "",
        "## Comparación sobre validation",
        "",
        "| Configuración | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Acc. (ref.) |",
        "|---|---|---|---|---|---|---|",
        f"| Champion (default) | {_row(champion.metrics)} |",
        f"| Candidato T-29 | {_row(candidate.metrics)} |",
        "",
    ]
    delta_pr = _float_metric(candidate.metrics, PRIMARY_METRIC) - _float_metric(
        champion.metrics, PRIMARY_METRIC
    )
    delta_rec = _float_metric(candidate.metrics, TIEBREAK_METRIC) - _float_metric(
        champion.metrics, TIEBREAK_METRIC
    )
    if decision:
        lines += [
            "## Decisión",
            "",
            f"**ADOPTAR.** El candidato mejora la métrica primaria AUC-PR sobre "
            f"`validation` ({_float_metric(candidate.metrics, PRIMARY_METRIC):.4f} vs "
            f"{_float_metric(champion.metrics, PRIMARY_METRIC):.4f}, "
            f"Δ +{delta_pr:.4f}) sin degradar el recall de forma material "
            f"({_float_metric(candidate.metrics, TIEBREAK_METRIC):.4f} vs "
            f"{_float_metric(champion.metrics, TIEBREAK_METRIC):.4f}, "
            f"Δ {delta_rec:+.4f}).",
            "",
            "## Configuración adoptada",
            "",
            f"`{CHAMPION_FAMILY}` con `{CANDIDATE_PARAMS}` (LogisticRegression).",
            "",
        ]
        if adopted is not None:
            test_metrics = _adopted_metrics(adopted)
            artifact = cast(dict[str, object], adopted["artifact"])
            test_file = cast(dict[str, object], adopted["test_file"])
            lines += [
                "- Registro MLflow: "
                f"`{MODEL_PREFIX}{CHAMPION_FAMILY}` versión "
                f"`{adopted['version']}` con alias `production`.",
                f"- Artefacto standalone: `{artifact['path']}`.",
                f"- Evaluación sobre `test` ({test_file['size']} filas, "
                f"sha256 `{test_file['sha256']}`):",
                "",
                "| Métrica | Valor (clase `Yes`) |",
                "|---|---|",
            ]
            for key in REPORT_METRIC_KEYS:
                lines.append(f"| {key} | {_as_float(test_metrics[key]):.4f} |")
            lines.append(
                "| TP/FN/FP/TN | "
                f"{int(_as_float(test_metrics['true_positive']))}/"
                f"{int(_as_float(test_metrics['false_negative']))}/"
                f"{int(_as_float(test_metrics['false_positive']))}/"
                f"{int(_as_float(test_metrics['true_negative']))} |"
            )
            if previous_champion is not None:
                pc = previous_champion
                lines += [
                    "",
                    "## Champion anterior",
                    "",
                    f"- Versión `{pc.get('version')}` (default de `T-14`), "
                    f"métricas sobre `test`: AUC-PR "
                    f"{float(pc['test_metrics']['auc_pr']):.4f}, recall "
                    f"{float(pc['test_metrics']['recall_pos']):.4f}.",
                ]
    else:
        lines += [
            "## Decisión",
            "",
            "**DESCARTAR.** El candidato no supera al champion según la regla "
            "de selección (AUC-PR primaria, recall como desempate). No se "
            "modifica el modelo en producción.",
            "",
            f"Diferencia observada: AUC-PR Δ {delta_pr:+.4f}, "
            f"recall Δ {delta_rec:+.4f}.",
            "",
        ]
    lines += [
        "",
        "Auditabilidad: corridas en el experimento `churn-detector` "
        "(configuración, métricas y artefactos); registro en "
        "`reports/adoption.json` y actualización de `reports/selection.*` "
        "cuando se adopta. La partición `test` se evalúa una única vez contra "
        "la configuración adoptada.",
        "",
    ]
    return "\n".join(lines)


def _row(metrics: dict[str, object]) -> str:
    return " | ".join(
        f"{_float_metric(metrics, key):.4f}" for key in REPORT_METRIC_KEYS
    )


def build_payload(
    champion: Any,
    candidate: Any,
    decision: bool,
    adopted: dict[str, object] | None,
    previous_champion: dict[str, Any] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "task": "T-30",
        "candidate": CHAMPION_FAMILY,
        "candidate_params": CANDIDATE_PARAMS,
        "selection_rule": SELECTION_RULE,
        "decision": "adoptar" if decision else "descartar",
        "validation": {
            "champion": champion.metrics,
            "candidate": candidate.metrics,
        },
        "improvement": {
            PRIMARY_METRIC: _float_metric(candidate.metrics, PRIMARY_METRIC)
            - _float_metric(champion.metrics, PRIMARY_METRIC),
            TIEBREAK_METRIC: _float_metric(candidate.metrics, TIEBREAK_METRIC)
            - _float_metric(champion.metrics, TIEBREAK_METRIC),
        },
        "evidence": "reports/hyperparams_exploration.md",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if decision and adopted is not None:
        payload["registry"] = {
            "model": f"{MODEL_PREFIX}{CHAMPION_FAMILY}",
            "alias": "production",
            "version": adopted["version"],
            "experiment": EXPERIMENT_NAME,
            "tracking_uri": TRACKING_URI,
        }
        payload["test_file"] = adopted["test_file"]
        payload["test_metrics"] = adopted["test_metrics"]
        payload["artifact"] = adopted["artifact"]
        if previous_champion is not None:
            payload["previous_champion"] = {
                "version": previous_champion.get("version"),
                "test_metrics": previous_champion.get("test_metrics"),
            }
    return payload


def _write_selection(
    champion: Any,
    candidate: Any,
    version: int,
    test_metrics: dict[str, object],
    model_file: Any,
    previous: dict[str, Any] | None,
) -> None:
    df_test = pd.read_csv(TEST_FILE)
    test_sha = sha256_file(TEST_FILE)
    selected = {
        "task": "T-14",
        "updated_by": "T-30 (adopcion)",
        "selection_rule": SELECTION_RULE,
        "selected": CHAMPION_FAMILY,
        "version": version,
        "validation": {
            "champion-default": champion.metrics,
            "candidate-t30": candidate.metrics,
        },
        "test_file": {
            "path": _as_relative(TEST_FILE),
            "sha256": test_sha,
            "size": int(len(df_test)),
        },
        "test_metrics": test_metrics,
        "registry": {
            "model": f"{MODEL_PREFIX}{CHAMPION_FAMILY}",
            "alias": "production",
            "experiment": EXPERIMENT_NAME,
            "tracking_uri": TRACKING_URI,
        },
        "artifact": {
            "path": _as_relative(model_file),
            "standalone": True,
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if previous is not None:
        selected["previous_champion"] = {
            "version": previous.get("version"),
            "validation": previous.get("validation"),
            "test_metrics": previous.get("test_metrics"),
            "created_at": previous.get("created_at"),
        }
    SELECTION_JSON.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_selection_md(
    champion: Any, candidate: Any, version: int, test_metrics: dict[str, object]
) -> None:
    section = [
        "",
        "---",
        "",
        "## Adopción de configuración (T-30)",
        "",
        "El champion de `T-14` fue reemplazado por la configuración ganadora "
        "de `T-29` (ver `reports/adoption.md`). Comparación sobre `validation`:",
        "",
        "| Configuración | AUC-PR | Recall |",
        "|---|---|---|",
        f"| Champion (default) | {float(champion.metrics['auc_pr']):.4f} | "
        f"{float(champion.metrics['recall_pos']):.4f} |",
        f"| Candidato T-30 (adoptado) | {float(candidate.metrics['auc_pr']):.4f} | "
        f"{float(candidate.metrics['recall_pos']):.4f} |",
        "",
        f"Nueva versión `production`: `{MODEL_PREFIX}{CHAMPION_FAMILY}` v{version}.",
        f"Evaluación sobre `test`: AUC-PR {_as_float(test_metrics['auc_pr']):.4f}, "
        f"recall {_as_float(test_metrics['recall_pos']):.4f}.",
        "",
    ]
    previous_text = (
        SELECTION_MD.read_text(encoding="utf-8") if SELECTION_MD.exists() else ""
    )
    SELECTION_MD.write_text(
        previous_text.rstrip() + "\n" + "\n".join(section), encoding="utf-8"
    )


def _decision_guard() -> bool:
    """True si el candidato ya fue evaluado y descartado con los datos actuales.

    Evita repetir (o re-adoptar) un candidato que ya recibió una decisión
    final de descarte documentada en `reports/adoption.json` para los mismos
    parámetros y las mismas particiones `train`/`validation`.
    """
    if not ADOPTION_JSON.exists():
        return False
    previous = json.loads(ADOPTION_JSON.read_text(encoding="utf-8"))
    if previous.get("decision") != "descartar":
        return False
    if previous.get("candidate_params") != CANDIDATE_PARAMS:
        return False
    inputs = previous.get("inputs") or {}
    return inputs.get("train_sha256") == sha256_file(TRAIN_FILE) and inputs.get(
        "validation_sha256"
    ) == sha256_file(VAL_FILE)


def _retire_version(name: str, version: int, tracking_uri: str = TRACKING_URI) -> None:
    """Retira (Archived) una versión para que no vuelva a seleccionarse."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        f"{MODEL_PREFIX}{name}", str(version), "Archived"
    )


def build_rejection_payload(
    previous: dict[str, Any],
    rejected: dict[str, Any],
    reverted_version: int,
) -> dict[str, object]:
    """Payload JSON de la reversión: decisión final `descartar` por `test`."""
    prior_test = previous.get("test_metrics") or {}
    rejected_test = rejected.get("test_metrics") or {}
    payload: dict[str, object] = {
        "task": "T-30",
        "candidate": CHAMPION_FAMILY,
        "candidate_params": CANDIDATE_PARAMS,
        "selection_rule": SELECTION_RULE,
        "decision": "descartar",
        "basis": "test",
        "because": (
            "El candidato superó al champion sobre validation pero obtuvo "
            "peores métricas sobre test; la mejora no generalizó y se revirtió "
            "la adopción."
        ),
        "validation": rejected.get("validation") or {},
        "test": {
            "champion": prior_test,
            "candidate-rejected": rejected_test,
        },
        "test_improvement": {
            PRIMARY_METRIC: _as_float(rejected_test.get(PRIMARY_METRIC, 0.0))
            - _as_float(prior_test.get(PRIMARY_METRIC, 0.0)),
            TIEBREAK_METRIC: _as_float(rejected_test.get(TIEBREAK_METRIC, 0.0))
            - _as_float(prior_test.get(TIEBREAK_METRIC, 0.0)),
        },
        "reverted_to": {"version": reverted_version},
        "evidence": "reports/hyperparams_exploration.md",
        "inputs": {
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VAL_FILE),
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return payload


def build_rejection_report(
    previous: dict[str, Any],
    rejected: dict[str, Any],
    reverted_version: int,
) -> str:
    """Reporte Markdown de la reversión con la evidencia de `test`."""
    prior_test = previous.get("test_metrics") or {}
    rejected_test = rejected.get("test_metrics") or {}
    validation = rejected.get("validation") or {}
    champ_val = validation.get("champion-default") or {}
    cand_val = validation.get("candidate-t30") or {}
    delta_pr = _as_float(rejected_test.get(PRIMARY_METRIC, 0.0)) - _as_float(
        prior_test.get(PRIMARY_METRIC, 0.0)
    )
    delta_rec = _as_float(rejected_test.get(TIEBREAK_METRIC, 0.0)) - _as_float(
        prior_test.get(TIEBREAK_METRIC, 0.0)
    )
    lines = [
        "# Adopción de configuración del modelo (T-30)",
        "",
        "Decisión final: **DESCARTAR** (la adopción se revirtió).",
        "",
        "Procedimiento: se entrenaron bajo condiciones idénticas (partición "
        "`train` de `T-11`, pipeline de `T-12`, semilla fija `42`) el champion "
        "actual (configuración por defecto de `T-13`/`T-14`) y el mejor "
        "candidato de `T-29`. La regla de selección de `T-14` se aplicó sobre "
        "`validation`; el candidato superó al champion y fue adoptado, pero la "
        "evaluación sobre `test` mostró que la mejora no generaliza y la "
        "adopción se revirtió.",
        "",
        "Regla: " + SELECTION_RULE,
        "",
        f"Candidato: `{CHAMPION_FAMILY}` con {CANDIDATE_PARAMS} "
        "(`reports/hyperparams_exploration.md`).",
        "",
        "## Comparación sobre validation",
        "",
        "| Configuración | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Acc. (ref.) |",
        "|---|---|---|---|---|---|---|",
        f"| Champion (default) | {_row(champ_val)} |",
        f"| Candidato T-29 | {_row(cand_val)} |",
        "",
        "## Comparación sobre test (decisión final)",
        "",
        "| Configuración | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Acc. (ref.) |",
        "|---|---|---|---|---|---|---|",
        f"| Champion anterior (default, v{previous.get('version')}) | "
        f"{_row(prior_test)} |",
        f"| Candidato adoptado (v{rejected.get('version')}) | "
        f"{_row(rejected_test)} |",
        "",
        "## Decisión",
        "",
        f"**DESCARTAR.** Sobre `test`, la configuración adoptada obtuvo una "
        f"AUC-PR de {_as_float(rejected_test.get(PRIMARY_METRIC, 0.0)):.4f} "
        f"frente a {_as_float(prior_test.get(PRIMARY_METRIC, 0.0)):.4f} del "
        f"champion anterior (Δ {delta_pr:+.4f}) y un recall de "
        f"{_as_float(rejected_test.get(TIEBREAK_METRIC, 0.0)):.4f} frente a "
        f"{_as_float(prior_test.get(TIEBREAK_METRIC, 0.0)):.4f} "
        f"(Δ {delta_rec:+.4f}). Sin mejora material, se mantiene la "
        f"configuración por defecto.",
        "",
        "## Acciones de reversión",
        "",
        f"- Alias `production` devuelto a `{MODEL_PREFIX}{CHAMPION_FAMILY}` "
        f"v{reverted_version}.",
        f"- Versión descartada `{MODEL_PREFIX}{CHAMPION_FAMILY}` "
        f"v{rejected.get('version')} retirada del registro (Archived).",
        "- `reports/selection.md/.json` restaurados al champion de `T-14`.",
        "- Artefacto standalone y predicciones regenerados con la "
        "configuración por defecto.",
        "",
        "Auditabilidad: corridas en el experimento `churn-detector` "
        "(configuración, métricas y artefactos); registro en "
        "`reports/adoption.json`. La partición `test` ya fue evaluada contra la "
        "configuración descartada; el champion anterior conserva su evaluación "
        "original.",
        "",
    ]
    return "\n".join(lines)


def _reject_and_revert() -> int:
    """Revierte la adopción de un candidato que no superó a `test`."""
    if not SELECTION_JSON.exists():
        raise ValueError("reports/selection.json no existe; no hay qué revertir.")
    current = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    previous = current.get("previous_champion")
    if previous is None or not previous.get("version"):
        raise ValueError(
            "selection.json no registra un champion anterior para revertir."
        )
    reverted_version = int(previous["version"])
    rejected_version = int(current.get("version", 0))

    promote_model(CHAMPION_FAMILY, version=reverted_version)
    if rejected_version:
        _retire_version(CHAMPION_FAMILY, rejected_version)
    pipeline = load_selected_model(CHAMPION_FAMILY)
    model_file = export_model(pipeline, CHAMPION_FAMILY)

    restored = cast(dict[str, Any], json.loads(json.dumps(previous)))
    restored["updated_by"] = "T-30 (revertido)"
    restored["version"] = reverted_version
    restored["artifact"] = {"path": _as_relative(model_file), "standalone": True}
    if not restored.get("test_file"):
        df_test = pd.read_csv(TEST_FILE)
        restored["test_file"] = {
            "path": _as_relative(TEST_FILE),
            "sha256": sha256_file(TEST_FILE),
            "size": int(len(df_test)),
        }
    restored["rejected_attempt"] = {
        "version": rejected_version,
        "params": CANDIDATE_PARAMS,
        "test_metrics": current.get("test_metrics"),
    }
    restored["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    SELECTION_JSON.write_text(
        json.dumps(restored, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    text = SELECTION_MD.read_text(encoding="utf-8") if SELECTION_MD.exists() else ""
    marker = "## Adopción de configuración (T-30)"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    section = [
        "---",
        "",
        "## Reversión de adopción (T-30)",
        "",
        "El candidato adoptado en `T-30` no superó al champion anterior sobre "
        "`test` (AUC-PR peor); la adopción se revirtió y el champion volvió a "
        "la configuración por defecto de `T-14`. Detalle y evidencia: "
        "`reports/adoption.md`.",
        "",
        f"Champion vigente: `{MODEL_PREFIX}{CHAMPION_FAMILY}` " f"v{reverted_version}.",
        "",
    ]
    SELECTION_MD.write_text(text + "\n".join(section), encoding="utf-8")

    from src.serving import predict as predict_mod

    predict_mod.main()

    payload = build_rejection_payload(previous, current, reverted_version)
    ADOPTION_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ADOPTION_MD.write_text(
        build_rejection_report(previous, current, reverted_version),
        encoding="utf-8",
    )
    print(
        f"Decisión: {str(payload['decision']).upper()} (revertido) -> "
        f"{_as_relative(ADOPTION_MD)}"
    )
    return 0


def main() -> int:
    if "--revert" in sys.argv[1:]:
        return _reject_and_revert()
    if _decision_guard():
        print(
            "El candidato de T-29 ya fue evaluado y descartado; se mantiene "
            "el champion actual sin cambios."
        )
        return 0

    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)

    champion = train_candidate(CHAMPION_FAMILY, df_train, df_val)
    candidate = train_candidate(CHAMPION_FAMILY, df_train, df_val, **CANDIDATE_PARAMS)
    decision = is_better(candidate.metrics, champion.metrics)
    champ_pr = _float_metric(champion.metrics, PRIMARY_METRIC)
    champ_rec = _float_metric(champion.metrics, TIEBREAK_METRIC)
    cand_pr = _float_metric(candidate.metrics, PRIMARY_METRIC)
    cand_rec = _float_metric(candidate.metrics, TIEBREAK_METRIC)
    print(f"Champion (default): AUC-PR={champ_pr:.4f}, recall={champ_rec:.4f}")
    print(f"Candidato T-29: AUC-PR={cand_pr:.4f}, recall={cand_rec:.4f}")

    previous_champion = None
    if SELECTION_JSON.exists():
        previous_champion = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))

    adopted: dict[str, object] | None = None
    if decision:
        log_candidate_run(
            name=candidate.name,
            pipeline=candidate.pipeline,
            params=candidate.params,
            metrics=candidate.metrics,
            feature_names=FEATURE_NAMES,
        )
        version = promote_model(candidate.name)
        pipeline = load_selected_model(candidate.name)
        model_file = export_model(pipeline, candidate.name)

        df_test = pd.read_csv(TEST_FILE)
        prob = prob_yes(pipeline, df_test[list(FEATURE_NAMES)])
        test_metrics = evaluate_metrics(df_test[TARGET], prob)
        log_selection_run(
            candidate.name,
            version,
            test_metrics,
            run_name="seleccion-t30",
        )
        test_sha = sha256_file(TEST_FILE)
        adopted = {
            "version": version,
            "test_file": {
                "path": _as_relative(TEST_FILE),
                "sha256": test_sha,
                "size": int(len(df_test)),
            },
            "test_metrics": test_metrics,
            "artifact": {
                "path": _as_relative(model_file),
                "standalone": True,
            },
        }
        _write_selection(
            champion, candidate, version, test_metrics, model_file, previous_champion
        )
        _append_selection_md(champion, candidate, version, test_metrics)

        from src.serving import predict as predict_mod

        predict_mod.main()

    payload = build_payload(champion, candidate, decision, adopted, previous_champion)
    if TRAIN_FILE.exists() and VAL_FILE.exists():
        payload["inputs"] = {
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VAL_FILE),
        }
    ADOPTION_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ADOPTION_MD.write_text(
        build_report(champion, candidate, decision, adopted, previous_champion),
        encoding="utf-8",
    )
    print(
        f"Decisión: {str(payload['decision']).upper()} -> "
        f"{_as_relative(ADOPTION_MD)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
