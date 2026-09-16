"""Explicabilidad global del modelo seleccionado con SHAP (T-28).

Calcula la importancia de las features del modelo seleccionado (`T-14`) con
valores SHAP y produce un reporte reproducible sobre la partición de
`validation` (`T-11`). La explicación es **descriptiva**: no reentrena ni
altera la selección, y describe cómo las features contribuyen a la
probabilidad de churn.

Método:
- El pipeline del modelo (`T-12` + estimador) se carga desde el Model Registry
  (alias `production`). Se aplica el paso `preprocessing` a la partición de
  validación y se usa `shap.LinearExplainer` sobre el estimador lineal
  (`T-14`), con un fondo de referencia acotado (`BACKGROUND_SIZE`).
- Las columnas one-hot de las features categóricas se **agrupan** de nuevo en
  su feature original (mediante `get_feature_names_out`), de modo que el
  reporte se expresa en términos interpretables del esquema de features.

Salidas (`reports/`):
- `explainability.md` — informe con la importancia global, figuras y trazabilidad.
- `explainability.json` — datos estructurados.
- `figures/shap_*.png` — barra de importancia, summary plot y dependence de
  las features más relevantes.

Ejecución:
    python -m src.modeling.explain
    python -m src.serving.pipeline explicabilidad
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.hashing import sha256_file
from src.modeling.preprocessing import (  # noqa: E402
    CATEGORIC_FEATURE_NAMES,
    FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
)
from src.modeling.select import load_selected_model  # noqa: E402
from src.paths import (  # noqa: E402
    EXPLAINABILITY_JSON,
    EXPLAINABILITY_MD,
    FIGURES_DIR,
    PROJECT_ROOT,
    VAL_FILE,
)

SELECTED_MODEL_NAME = "logistic-regression"
BACKGROUND_SIZE = 500
MAX_DISPLAY = 20
TOP_N_DEPENDENCE = 3

DEFAULT_FIGURES = (
    "shap_importance_bar.png",
    "shap_summary.png",
)


def preprocessed_matrix(
    pipeline: Pipeline, df: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Aplica el paso `preprocessing` del pipeline y devuelve la matriz y nombres.

    El transformador ya está ajustado (train, `T-12`); aquí solo se transforma,
    de modo que no se filtra información de la partición hacia el modelo.
    """
    X = pipeline.named_steps["preprocessing"].transform(df[list(FEATURE_NAMES)])
    names = list(pipeline.named_steps["preprocessing"].get_feature_names_out())
    return np.asarray(X), names


def map_column_to_original(name: str) -> str:
    """Devuelve la feature original a la que pertenece una columna preprocesada."""
    if name in NUMERIC_FEATURE_NAMES:
        return name
    for feature in CATEGORIC_FEATURE_NAMES:
        if name.startswith(feature + "_"):
            return feature
    raise ValueError(f"Columna preprocesada desconocida: {name}")


def aggregate_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa las contribuciones SHAP de las columnas codificadas por feature.

    Suma las atribuciones de cada one-hot a su feature original (las columnas
    resultantes conservan el orden de primera aparición).
    """
    roots = [map_column_to_original(col) for col in shap_df.columns]
    return shap_df.T.groupby(roots, sort=False).sum().T


@dataclass(frozen=True)
class ShapResult:
    """Resultado de la explicación: valor base y contribuciones por feature."""

    base_value: float
    shap_df: pd.DataFrame


def generate_shap(
    pipeline: Pipeline,
    df: pd.DataFrame,
    background_size: int = BACKGROUND_SIZE,
) -> ShapResult:
    """Calcula los valores SHAP del pipeline sobre ``df`` (agrupados por feature)."""
    X, names = preprocessed_matrix(pipeline, df)
    background = X[: min(background_size, X.shape[0])]
    explainer = shap.LinearExplainer(pipeline.named_steps["model"], background)
    shap_matrix = np.asarray(explainer.shap_values(X))
    shap_full = pd.DataFrame(shap_matrix, columns=names)
    base_value = float(np.asarray(explainer.expected_value).ravel()[0])
    shap_agg = aggregate_shap(shap_full).reindex(columns=list(FEATURE_NAMES))
    return ShapResult(base_value=base_value, shap_df=shap_agg)


def feature_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Importancia global: media del |SHAP| por feature, de mayor a menor."""
    return shap_df.abs().mean().sort_values(ascending=False)


def write_figures(
    shap_df: pd.DataFrame, features: pd.DataFrame, figures_dir: Path
) -> None:
    """Genera los gráficos de explicabilidad en ``figures_dir`` (creado si falta)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(shap_df.columns)
    features = features[list(FEATURE_NAMES)]

    plt.figure()
    shap.summary_plot(
        shap_df.values,
        plot_type="bar",
        feature_names=feature_names,
        max_display=MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_importance_bar.png", bbox_inches="tight", dpi=150)
    plt.close("all")

    plt.figure()
    shap.summary_plot(
        shap_df.values,
        features,
        feature_names=feature_names,
        max_display=MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close("all")

    importance = feature_importance(shap_df)
    top_features = list(importance.index[:TOP_N_DEPENDENCE])
    for feature in top_features:
        plt.figure()
        shap.dependence_plot(
            feature_names.index(feature),
            shap_df.values,
            features,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"shap_dependence_{feature}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close("all")


def build_report(
    result: ShapResult,
    df: pd.DataFrame,
    importance: pd.Series,
    val_sha: str,
    model_name: str,
) -> str:
    """Construye el informe de explicabilidad (markdown)."""
    total = float(importance.sum())
    importance_sorted = importance.sort_values(ascending=False)
    means_signed = result.shap_df.mean().reindex(importance_sorted.index)

    lines = [
        "# Explicabilidad del modelo (T-28)",
        "",
        "Método: valores SHAP con `shap.LinearExplainer` sobre el pipeline del "
        "modelo seleccionado (features preprocesadas con OHE y escalado, `T-12`). "
        "Las contribuciones de las columnas codificadas se agrupan en su feature "
        "original para un reporte interpretable. La explicación es descriptiva y "
        "no reentrena ni altera la selección de `T-14`.",
        "",
        f"- Modelo explicado: `churn-{model_name}@production` (specs.md §7).",
        f"- Partición: `{VAL_FILE.relative_to(PROJECT_ROOT).as_posix()}` "
        f"({len(df)} filas, sha256 `{val_sha}`). `test` queda reservado a la "
        "evaluación única de `T-14`.",
        f"- Fondo de referencia: primeras {min(BACKGROUND_SIZE, len(df))} filas de "
        "la partición (features preprocesadas).",
        f"- Base value (log-odds medio del modelo): {result.base_value:.4f}.",
        "",
        "## Importancia global de features",
        "",
        "| Feature | mean \\|SHAP\\| | % contribución | media (SHAP firmado) |",
        "|---|---|---|---|",
    ]
    for feature in importance_sorted.index:
        value = float(importance_sorted[feature])
        pct = value / total * 100 if total > 0 else 0.0
        signed = float(means_signed[feature])
        lines.append(f"| {feature} | {value:.4f} | {pct:.1f}% | {signed:+.4f} |")
    top3 = list(importance_sorted.index[:3])
    lines += [
        "",
        f"Las features con mayor contribución absoluta son **{', '.join(top3)}**, "
        "coherentes con los hallazgos del EDA de `T-09` sobre la relación entre "
        "antigüedad, contrato y modalidad de cobro con el churn. El signo de la "
        "media SHAP indica la dirección media de cada feature sobre la "
        "probabilidad de churn.",
        "",
        "## Figuras",
        "",
        f"- Importancia (global): "
        f"{FIGURES_DIR.relative_to(PROJECT_ROOT).as_posix()}/shap_importance_bar.png",
        f"- Summary plot: "
        f"{FIGURES_DIR.relative_to(PROJECT_ROOT).as_posix()}/shap_summary.png",
    ]
    for feature in top3:
        lines.append(
            f"- Dependence de `{feature}`: "
            f"{FIGURES_DIR.relative_to(PROJECT_ROOT).as_posix()}/"
            f"shap_dependence_{feature}.png"
        )
    lines += [
        "",
        "## Trazabilidad",
        "",
        "- Partición `validation` (T-11) con hashes en "
        "`data/splits/partition_log.json`.",
        "- Modelo cargado desde el Model Registry con alias `production` (T-14).",
        "- Artefactos estructurados en `reports/explainability.json`.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(
    result: ShapResult,
    df: pd.DataFrame,
    md_path: Path,
    json_path: Path,
    figures_dir: Path,
    val_sha: str,
    model_name: str = SELECTED_MODEL_NAME,
) -> None:
    """Escribe el informe, los datos estructurados y las figuras (T-28)."""
    write_figures(result.shap_df, df, figures_dir)
    importance = feature_importance(result.shap_df)
    total = float(importance.sum())
    signed = result.shap_df.mean()
    payload = {
        "task": "T-28",
        "model": f"churn-{model_name}",
        "alias": "production",
        "method": "shap.LinearExplainer sobre features preprocesadas",
        "partition": {
            "path": str(VAL_FILE.relative_to(PROJECT_ROOT)),
            "sha256": val_sha,
            "rows": int(len(df)),
        },
        "background_size": min(BACKGROUND_SIZE, int(len(df))),
        "base_value": result.base_value,
        "feature_importance": [
            {
                "feature": feature,
                "mean_abs_shap": float(value),
                "share_pct": float(value / total * 100) if total > 0 else 0.0,
                "mean_shap": float(signed[feature]),
            }
            for feature, value in importance.items()
        ],
        "figures": {
            name: str(FIGURES_DIR.relative_to(PROJECT_ROOT) / name)
            for name in (
                *DEFAULT_FIGURES,
                *(
                    f"shap_dependence_{f}.png"
                    for f in importance.index[:TOP_N_DEPENDENCE]
                ),
            )
        },
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        build_report(result, df, importance, val_sha, model_name),
        encoding="utf-8",
    )


def main() -> int:
    pipeline = load_selected_model(SELECTED_MODEL_NAME)
    df_val = pd.read_csv(VAL_FILE)
    result = generate_shap(pipeline, df_val)
    val_sha = sha256_file(VAL_FILE)
    figures_dir = FIGURES_DIR
    write_artifacts(
        result, df_val, EXPLAINABILITY_MD, EXPLAINABILITY_JSON, figures_dir, val_sha
    )
    importance = feature_importance(result.shap_df)
    top = importance.index[0]
    print(
        f"Explicabilidad generada: feature top '{top}' "
        f"(mean |SHAP|={importance.iloc[0]:.4f})."
    )
    print(f"Informe: {EXPLAINABILITY_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
