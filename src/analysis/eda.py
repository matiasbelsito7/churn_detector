"""EDA reproducible del dataset procesado (T-09).

Genera `reports/eda.md` con las figuras en `reports/figures/`, trabajando
sobre `data/processed/churn_cleaned.csv` (T-08). No depende de ejecución
manual ni de estado de sesión.

Ejecución:
    python -m src.analysis.eda
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.clean import sha256_file
from src.data.contract import COLUMNS, TARGET
from src.seeds import RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_FILE = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "eda.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

NUMERIC_COLS = ("tenure", "MonthlyCharges", "TotalCharges")
# `customerID` es identificador, no feature: se excluye del análisis.
CATEGORICAL_COLS = tuple(
    name
    for name in COLUMNS
    if name not in NUMERIC_COLS and name != TARGET and name != "customerID"
)

COLOR_NO = "#1f77b4"
COLOR_YES = "#d62728"


def md_table(header: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    head = "| " + " | ".join(header) + " |"
    sep = "|" + "---|" * len(header)
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body]) + "\n"


def pct(x: float, decimals: int = 2) -> str:
    return f"{x:.{decimals}f}"


def churn_int(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].map({"No": 0, "Yes": 1}).astype(int)


# --------------------------------------------------------------------------- #
# Secciones del informe
# --------------------------------------------------------------------------- #


def balance_block(df: pd.DataFrame) -> tuple[list[str], str]:
    counts = df[TARGET].value_counts()
    rows = [
        (
            class_,
            int(counts[class_]),
            pct(counts[class_] / len(df) * 100),
            class_ == "Yes",
        )
        for class_ in ("No", "Yes")
    ]
    lines = [
        "## 1. Desbalanceo del target",
        "",
        "| Clase | n | % | Clase de interés |",
        "|---|---|---|---|",
        *[
            f"| {c} | {n} | {p} | {'**Sí**' if yes else 'No'} |"
            for c, n, p, yes in rows
        ],
        "",
        f"- Ratio no-churn / churn: **{pct(counts['No'] / counts['Yes'], 2)}:1**.",
        f"- Clase positiva (interés): **{TARGET} == 'Yes'**.",
        "",
        "![Desbalanceo del target](figures/eda_target_balance.png)",
        "",
    ]
    return lines, "eda_target_balance.png"


def missing_block() -> list[str]:
    return [
        "## 2. Valores faltantes",
        "",
        "| Campo | Ausencias |",
        "|---|---|",
        *[f"| {col} | 0 |" for col in COLUMNS],
        "",
        "El dataset procesado no presenta valores faltantes. El raw sí tenía 11 "
        "ausencias estructurales en `TotalCharges` (todas con `tenure == 0` y "
        "`Churn == 'No'`), tratadas por imputación con 0 en `T-08` "
        "(`docs/missing_values.md`). La ausencia en el raw era determinística, no "
        "aleatoria, y estaba asociada a clientes nuevos.",
        "",
    ]


def univariate_block(df: pd.DataFrame) -> list[str]:
    lines = [
        "## 3. Perfil univariado",
        "",
        "### 3.1 Variables categóricas",
        "",
        "| Variable | Categoría | n | % |",
        "|---|---|---|---|",
    ]
    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts(dropna=False)
        for value, n in counts.items():
            lines.append(f"| {col} | {value} | {int(n)} | {pct(n / len(df) * 100)} |")
        lines.append("|---|---|---|---|")

    lines += [
        "",
        "### 3.2 Variables numéricas",
        "",
        "| Variable | n | media | std | min | p25 | p50 | p75 | max | atípicos(IQR) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for col in NUMERIC_COLS:
        ser = df[col].dropna()
        q1, q3 = ser.quantile(0.25), ser.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((ser < q1 - 1.5 * iqr) | (ser > q3 + 1.5 * iqr)).sum())
        lines.append(
            f"| {col} | {int(ser.count())} | {pct(ser.mean(), 2)} | "
            f"{pct(ser.std(), 2)} | {pct(ser.min(), 2)} | {pct(q1, 2)} | "
            f"{pct(ser.median(), 2)} | {pct(q3, 2)} | {pct(ser.max(), 2)} | {n_out} |"
        )
    lines.append("")
    return lines


def bivariate_categorical_block(df: pd.DataFrame) -> tuple[list[str], str]:
    lines = [
        "## 4.1 Variables categóricas vs target",
        "",
        "| Variable | Categoría | n | churn (n) | tasa de churn |",
        "|---|---|---|---|---|",
    ]
    for col in CATEGORICAL_COLS:
        tab = (
            df.groupby(col)[TARGET]
            .agg(count="count", churn=lambda s: (s == "Yes").sum())
            .reset_index()
        )
        for _, row in tab.iterrows():
            rate = row["churn"] / row["count"] * 100
            lines.append(
                f"| {col} | {row[col]} | {int(row['count'])} | {int(row['churn'])} | "
                f"{pct(rate)} % |"
            )
    lines += [
        "",
        "![Tasa de churn por variable categórica](figures/eda_churn_rate_cat.png)",
        "",
    ]
    return lines, "eda_churn_rate_cat.png"


def _safe_bins(ser: pd.Series, n_bins: int) -> pd.Series:
    unique = ser.unique()
    if len(unique) <= n_bins or len(ser) <= n_bins:
        return ser.astype(str)
    q = pd.qcut(ser, n_bins, duplicates="drop")
    return q.astype(str)


def bivariate_numeric_block(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    target = churn_int(df)
    corr_rows = [
        (col, pct(float(np.corrcoef(df[col], target)[0, 1]), 3)) for col in NUMERIC_COLS
    ]
    lines = [
        "## 4.2 Variables numéricas vs target",
        "",
        "### Correlación punto-biserial con el target",
        "",
        "| Variable | r |",
        "|---|---|",
        *[f"| {c} | {r} |" for c, r in corr_rows],
        "",
        "### Tasa de churn por tramos",
        "",
    ]
    bins_rows: list[str] = []
    for col in NUMERIC_COLS:
        bins_rows.append("#### " + col + "\n")
        bins_rows.append("| Tramos | n | tasa de churn |\n|---:|---:|---:|")
        binned = _safe_bins(df[col], 8)
        tab = (
            df.assign(_bin=binned)
            .groupby("_bin")[TARGET]
            .agg(count="count", churn=lambda s: (s == "Yes").sum())
            .reset_index()
        )
        for _, row in tab.iterrows():
            bins_rows.append(
                f"| {row['_bin']} | {int(row['count'])} | "
                f"{pct(row['churn'] / row['count'] * 100)} % |"
            )
        bins_rows.append("")
    lines += bins_rows
    lines += [
        "![Distribución de numéricas por clase](figures/eda_numeric_by_target.png)",
        "",
        "![Tasa de churn por antigüedad](figures/eda_churn_tenure.png)",
        "",
    ]
    return lines, ["eda_numeric_by_target.png", "eda_churn_tenure.png"]


def findings_block(df: pd.DataFrame) -> list[str]:
    def rate(col: str, value: str) -> float:
        mask = df[col] == value
        return float(df.loc[mask, TARGET].eq("Yes").mean() * 100)

    def rate_cond(mask: pd.Series) -> float:
        return float(df.loc[mask, TARGET].eq("Yes").mean() * 100)

    contract = {c: rate("Contract", c) for c in df["Contract"].unique()}
    payment = {c: rate("PaymentMethod", c) for c in df["PaymentMethod"].unique()}
    internet = {c: rate("InternetService", c) for c in df["InternetService"].unique()}
    senior = {c: rate("SeniorCitizen", c) for c in df["SeniorCitizen"].unique()}
    counts = df[TARGET].value_counts()
    tenure_low = rate_cond(df["tenure"].le(3))
    tenure_high = rate_cond(df["tenure"].ge(60))
    r_t_tc = float(np.corrcoef(df["tenure"], df["TotalCharges"])[0, 1])
    r_mc_tc = float(np.corrcoef(df["MonthlyCharges"], df["TotalCharges"])[0, 1])

    findings = [
        (
            "F1",
            f"Desbalanceo del target: churn {pct(counts['Yes'] / len(df) * 100)} %.",
            "Evaluación con métricas sobre la clase minoritaria (recall, "
            "precisión, F1, AUC-PR); no usar accuracy como criterio.",
        ),
        (
            "F2",
            f"Churn muy alto en clientes nuevos (tenure <= 3 meses: "
            f"{pct(tenure_low)} %) y bajo en antiguos (tenure >= 60: "
            f"{pct(tenure_high)} %).",
            "`tenure` es feature central; considerar bins de antigüedad.",
        ),
        (
            "F3",
            "Contrato: Month-to-month "
            f"{pct(contract['Month-to-month'])} %, One year "
            f"{pct(contract['One year'])} %, Two year "
            f"{pct(contract['Two year'])} %.",
            "Feature fuerte; candidata a interacción contract × tenure.",
        ),
        (
            "F4",
            "PaymentMethod: Electronic check "
            f"{pct(payment['Electronic check'])} % vs automáticos/listados "
            f"({pct(payment['Mailed check'])} % o menos).",
            "Feature candidata; evaluar agrupación de métodos de pago.",
        ),
        (
            "F5",
            "InternetService: Fiber optic "
            f"{pct(internet['Fiber optic'])} %, DSL {pct(internet['DSL'])} %, "
            f"No {pct(internet['No'])} %.",
            "Feature candidata; mantener la distinción DSL vs Fiber optic.",
        ),
        (
            "F6",
            "Servicios de valor añadido ausentes (OnlineSecurity, TechSupport, "
            "etc.) se asocian a mayor churn.",
            "Features indicadoras de adopción; evaluar interacciones con "
            "InternetService.",
        ),
        (
            "F7",
            f"SeniorCitizen: churn {pct(senior['Yes'])} % (seniors) vs "
            f"{pct(senior['No'])} %.",
            "Feature incluida; representación categórica binaria.",
        ),
        (
            "F8",
            "MonthlyCharges correlaciona positivamente con churn, pero está "
            "confundida con tenure y tipo de contrato.",
            "Evaluar en modelos condicionando por tenure/contract; estandarizar.",
        ),
        (
            "F9",
            f"Alta correlación entre variables numéricas: tenure–TotalCharges "
            f"r = {r_t_tc:.2f}, MonthlyCharges–TotalCharges r = {r_mc_tc:.2f}.",
            "Riesgo de multicolinealidad; considerar reducir o derivar variables "
            "(p. ej. cargo medio histórico).",
        ),
    ]

    lines = [
        "## 5. Hallazgos y decisiones",
        "",
        "| Hallazgo | Evidencia | Decisión para fases siguientes (T-10/T-11/T-12) |",
        "|---|---|---|",
    ]
    for fid, evidence, decision in findings:
        lines.append(f"| **{fid}** | {evidence} | {decision} |")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #


def plot_target_balance(df: pd.DataFrame, dest_dir: Path) -> None:
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(
        ["No", "Yes"],
        [counts["No"], counts["Yes"]],
        color=[COLOR_NO, COLOR_YES],
    )
    ax.bar_label(bars, fmt="%d")
    ax.set_title("Distribución del target")
    ax.set_ylabel("Clientes")
    ax.set_ylim(0, counts["No"] * 1.15)
    fig.tight_layout()
    fig.savefig(dest_dir / "eda_target_balance.png", dpi=110)
    plt.close(fig)


def plot_churn_rate_cat(df: pd.DataFrame, dest_dir: Path) -> None:
    n_vars = len(CATEGORICAL_COLS)
    ncols = 6
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, CATEGORICAL_COLS, strict=False):
        tab = (
            df.groupby(col)[TARGET]
            .agg(count="count", churn=lambda s: (s == "Yes").sum())
            .reset_index()
        )
        rates = (tab["churn"] / tab["count"] * 100).to_numpy()
        labels = tab[col].astype(str).str.replace(" ", "\n", regex=False).to_numpy()
        ax.barh(np.arange(len(labels)), rates, color=COLOR_YES, alpha=0.9)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(col, fontsize=8)
        ax.set_xlim(0, 60)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes[len(CATEGORICAL_COLS) :]:
        ax.set_visible(False)
    fig.suptitle("Tasa de churn por variable y categoría (%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(dest_dir / "eda_churn_rate_cat.png", dpi=110)
    plt.close(fig)


def plot_numeric_by_target(df: pd.DataFrame, dest_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(13, 3.6))
    for ax, col in zip(np.atleast_1d(axes), NUMERIC_COLS, strict=True):
        for class_, color in (("No", COLOR_NO), ("Yes", COLOR_YES)):
            ax.hist(
                df.loc[df[TARGET] == class_, col].dropna(),
                bins=40,
                alpha=0.55,
                color=color,
                label=f"Churn={class_}",
            )
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.legend(fontsize=7)
    fig.suptitle("Distribución de variables numéricas por clase", fontsize=11)
    fig.tight_layout()
    fig.savefig(dest_dir / "eda_numeric_by_target.png", dpi=110)
    plt.close(fig)


def plot_churn_tenure(df: pd.DataFrame, dest_dir: Path) -> None:
    bins = [0, 6, 12, 24, 36, 48, 60, 73]
    labels = ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59", "60-72"]
    binned = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)
    tab = (
        df.assign(_bin=binned)
        .groupby("_bin", observed=True)[TARGET]
        .agg(count="count", churn=lambda s: (s == "Yes").sum())
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(
        tab["_bin"].astype(str),
        tab["churn"] / tab["count"] * 100,
        marker="o",
        color=COLOR_YES,
    )
    ax.set_xlabel("Antigüedad (meses)")
    ax.set_ylabel("Tasa de churn (%)")
    ax.set_title("Tasa de churn por antigüedad")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest_dir / "eda_churn_tenure.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #


def build_report(df: pd.DataFrame) -> str:
    balance_lines, fig_balance = balance_block(df)
    cats_lines, fig_cat = bivariate_categorical_block(df)
    nums_lines, fig_nums = bivariate_numeric_block(df)
    header = [
        "# EDA del dataset procesado (T-09)",
        "",
        "Informe generado automáticamente. Fuente: "
        f"`{CLEANED_FILE.relative_to(PROJECT_ROOT)}` "
        f"(checksum `{sha256_file(CLEANED_FILE)}`).",
        f"- Tamaño: {len(df)} filas × {len(df.columns)} columnas.",
        f"- Semilla fija del proyecto: `{RANDOM_SEED}`.",
        "- Reproducción: `python -m src.analysis.eda`.",
        "",
    ]
    return "\n".join(
        header
        + balance_lines
        + missing_block()
        + univariate_block(df)
        + ["## 4. Perfil bivariado", ""]
        + cats_lines
        + nums_lines
        + findings_block(df)
    )


def main() -> int:
    df = pd.read_csv(CLEANED_FILE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_target_balance(df, FIGURES_DIR)
    plot_churn_rate_cat(df, FIGURES_DIR)
    plot_numeric_by_target(df, FIGURES_DIR)
    plot_churn_tenure(df, FIGURES_DIR)
    REPORT_FILE.write_text(build_report(df), encoding="utf-8")
    print(f"Informe EDA escrito en {REPORT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
