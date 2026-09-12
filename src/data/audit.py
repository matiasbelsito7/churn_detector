"""Data audit del dataset crudo (T-05).

Genera `reports/audit_raw.md` con hallazgos por campo: tipos reales vs
esperados, nulos, valores fuera de dominio, cardinalidad, duplicados y
coherencia entre campos, contrastados contra el contrato de datos.

No modifica el raw: solo lo lee y escribe el informe en `reports/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data.contract import COLUMNS, INTERNET_DEPENDENT_FIELDS, SPECS, TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "audit_raw.md"

TYPE_LABEL = {
    "id": "identificador",
    "category": "categórica",
    "int": "entera",
    "float": "flotante",
    "target": "target (categórica binaria)",
}


def numeric_column(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "if":
        return series
    return pd.to_numeric(series, errors="coerce")


def multivariate_checks(df: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    no_phone_service = df["MultipleLines"].eq("No phone service")
    no_phone = df["PhoneService"].eq("No")
    if not (no_phone_service == no_phone).all():
        n = int((no_phone_service != no_phone).sum())
        findings.append(
            f"- **Coherencia (MultipleLines/PhoneService):** {n} filas "
            "incoherentes. Se exige `MultipleLines == 'No phone service'` si y "
            "solo si `PhoneService == 'No'`."
        )

    no_internet = df["InternetService"].eq("No")
    for field in INTERNET_DEPENDENT_FIELDS:
        contradict = df[field].eq("No internet service") != no_internet
        n = int(contradict.sum())
        if n:
            findings.append(
                f"- **Coherencia ({field}/InternetService):** {n} filas "
                f"incoherentes con la regla `{field} == 'No internet service'` "
                "si y solo si `InternetService == 'No'`."
            )

    tenure_zero_missing_tc = df["tenure"].eq(0) & df["TotalCharges"].isna()
    missing_not_zero = df["TotalCharges"].isna() & ~df["tenure"].eq(0)
    n_missing = int(df["TotalCharges"].isna().sum())
    if missing_not_zero.any():
        n = int(missing_not_zero.sum())
        findings.append(
            f"- **Ausencia TotalCharges:** {n} filas con `TotalCharges` ausente "
            "teniendo `tenure > 0` (se exige ausencia solo cuando `tenure == 0`)."
        )
    if n_missing:
        findings.append(
            f"- **Ausencia TotalCharges (esperada):** {n_missing} ausencias, "
            "todas con `tenure == 0` (ausencia estructural, no aleatoria)."
            if not tenure_zero_missing_tc.sum() != n_missing
            else f"- **Ausencia TotalCharges (esperada parcial):** {n_missing} "
            "ausencias, de las cuales "
            f"{int(tenure_zero_missing_tc.sum())} con `tenure == 0`."
        )

    if not findings:
        findings.append("- Sin incoherencias entre campos detectadas.")
    return findings


def audit(df: pd.DataFrame) -> str:
    expected_cols = list(COLUMNS)
    lines: list[str] = [
        "# Audit del dataset crudo (T-05)",
        "",
        "Informe generado automáticamente desde el raw versionado.",
        "",
        f"- **Archivo:** `{RAW_FILE.relative_to(PROJECT_ROOT)}`",
        f"- **Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas",
        f"- **Filas duplicadas (completas):** {int(df.duplicated().sum())}",
        f"- **Identificadores repetidos (`customerID`):** "
        f"{int(df['customerID'].duplicated().sum())}",
        "",
        "## 1. Esquema: tipos reales vs esperados",
        "",
        "| Campo | Tipo esperado | Tipo real (raw) | Desviación |",
        "|---|---|---|---|",
    ]

    for col in expected_cols:
        spec = SPECS[col]
        inferred = "object" if df[col].dtype == object else str(df[col].dtype)
        desired = TYPE_LABEL[spec.logical_type]
        if spec.logical_type in ("float", "int"):
            coerced = numeric_column(df[col])
            if coerced.isna().sum() != df[col].isna().sum():
                n_unparsed = int((df[col].notna() & coerced.isna()).sum())
                deviation = (
                    f"**SÍ — {n_unparsed} valores no parseables como "
                    f"{desired} (antes: {inferred})**"
                )
            elif spec.logical_type == "float" and df[col].dtype.kind in "if":
                deviation = (
                    "No"
                    if str(df[col].dtype) == "float64"
                    else f"SÍ — dtype `{inferred}`"
                )
            elif spec.logical_type == "int" and df[col].dtype.kind == "i":
                deviation = "No"
            else:
                deviation = f"SÍ — almacenado como `{inferred}`"
        else:
            deviation = "No" if df[col].dtype == object else f"SÍ — dtype `{inferred}`"

        if spec.nullable and col == "TotalCharges":
            deviation += " (nulos esperados si `tenure == 0`)"

        lines.append(f"| {col} | {desired} | `{inferred}` | {deviation} |")

    lines += [
        "",
        "## 2. Valores nulos y ausencias",
        "",
        "| Campo | Nulos |",
        "|---|---|",
    ]
    for col in expected_cols:
        if col == "TotalCharges":
            n = int(numeric_column(df[col]).isna().sum())
        else:
            n = int(df[col].isna().sum())
        lines.append(f"| {col} | {n} |")

    lines += [
        "",
        "## 3. Dominios y rangos",
        "",
        "| Campo | Dominio/rango esperado | Valores observados | Cumple |",
        "|---|---|---|---|",
    ]
    for col in expected_cols:
        spec = SPECS[col]
        series = df[col].dropna()
        if spec.logical_type == "id":
            obs_unique = int(series.nunique())
            expected = "único (patrón `^\\d{4}-[A-Z]{5}$`)"
            matches = series.astype(str).str.match(r"^\d{4}-[A-Z]{5}$")
            ok = bool(obs_unique == len(series) and int((~matches).sum()) == 0)
            values = f"{obs_unique} valores únicos"
        elif spec.logical_type in ("category", "target"):
            observed = sorted(set(series.unique().tolist()), key=str)
            ok = set(observed) <= set(spec.allowed or ())
            expected = ", ".join(spec.allowed or ())
            values = ", ".join(observed)
        else:
            num = numeric_column(series)
            obs_min = num.min()
            obs_max = num.max()
            expected = (
                f"[{spec.minimum}, {spec.maximum}]"
                if spec.maximum is not None
                else f">= {spec.minimum}"
            )
            lower_ok = spec.minimum is None or obs_min >= spec.minimum
            upper_ok = spec.maximum is None or obs_max <= spec.maximum
            ok = bool(lower_ok and upper_ok)
            values = f"[{obs_min}, {obs_max}]"
        lines.append(f"| {col} | {expected} | {values} | {'Sí' if ok else '**No**'} |")

    lines += [
        "",
        "## 4. Coherencia entre campos",
        "",
        *multivariate_checks(df),
        "",
        "## 5. Resumen de hallazgos",
        "",
    ]

    findings: list[str] = []
    if df["TotalCharges"].dtype == object:
        findings.append(
            f"- `TotalCharges` se almacena como texto con "
            f"{int(numeric_column(df['TotalCharges']).isna().sum())} celdas no "
            "numéricas (solo espacios), todas con `tenure == 0` (ausencia "
            "estructural, no aleatoria). Debe convertirse a flotante en `T-07`."
        )
    findings.append(
        "- `SeniorCitizen` usa codificación numérica `0/1` mientras el resto de "
        "los binarios usa `No/Yes`; normalizar en `T-07` para un esquema "
        "consistente."
    )
    no_internet_map = {
        f: int(
            (df[f].eq("No internet service") != df["InternetService"].eq("No")).sum()
        )
        for f in INTERNET_DEPENDENT_FIELDS
    }
    if any(v for v in no_internet_map.values()):
        findings.append(
            "- Existen incoherencias con la regla de `No internet service` "
            "(ver sección 4)."
        )
    target_counts = df[COLUMNS[-1]].value_counts()
    lines += findings or ["- Sin hallazgos adicionales detectados."]

    lines += [
        "",
        "## 6. Desbalanceo del target",
        "",
    ]
    total = len(df)
    for cls in SPECS[TARGET].allowed or ():
        count = int(target_counts.get(cls, 0))
        pct = count / total * 100
        lines.append(f"- `{cls}`: {count} ({pct:.2f}%)")
    lines.append(
        f"- Clase de interés (positiva): `Yes` ({target_counts.get('Yes', 0)})."
    )
    return "\n".join(lines)


def main() -> int:
    df = pd.read_csv(RAW_FILE)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(audit(df), encoding="utf-8")
    print(f"Informe de audit escrito en {REPORT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
