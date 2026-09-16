"""Checks de calidad de datos automatizados (T-06).

Implementa las reglas R1-R7 del contrato (`docs/data_contract.md`, sección 4).
Cada check devuelve un resultado explícito (aprobado/rechazado). El conjunto se
ejecuta sobre un ``DataFrame``: si alguna regla falla, el dataset queda
bloqueado para la siguiente fase.

Ejecución como reporte:
    python -m src.data.quality_checks

Genera ``reports/quality_report.md`` y termina con código de salida 0 (aprobado)
o 1 (rechazado).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.data.contract import (
    COLUMNS,
    INTERNET_DEPENDENT_FIELDS,
    SPECS,
)
from src.paths import PROJECT_ROOT, RAW_FILE
from src.paths import QUALITY_REPORT as REPORT_FILE

CUSTOMER_ID_PATTERN = re.compile(r"^\d{4}-[A-Z]{5}$")


@dataclass(frozen=True)
class CheckResult:
    rule_id: str
    name: str
    passed: bool
    details: str


def _numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "if":
        return series
    return pd.to_numeric(series, errors="coerce")


def _blank_as_missing(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "if":
        return series
    return series.replace(r"^\s*$", pd.NA, regex=True)


def check_schema(df: pd.DataFrame) -> CheckResult:
    expected = list(COLUMNS)
    actual = list(df.columns)
    passed = actual == expected
    detail = (
        "El esquema coincide con el contrato."
        if passed
        else f"Esquema distinto. Esperado: {expected}. Actual: {actual}."
    )
    return CheckResult("R1", "Esquema de columnas", passed, detail)


def check_domains(df: pd.DataFrame) -> CheckResult:
    problems: list[str] = []
    for name in COLUMNS:
        spec = SPECS[name]
        if spec.logical_type not in ("category", "target"):
            continue
        series = df[name]
        unexpected = sorted(
            {
                v
                for v in series.dropna().unique().tolist()
                if v not in (spec.allowed or ())
            },
            key=str,
        )
        if unexpected:
            problems.append(f"{name}: valores fuera de dominio {unexpected}.")
        n_missing = int(series.isna().sum())
        if n_missing:
            problems.append(f"{name}: {n_missing} valores ausentes.")
    passed = not problems
    return CheckResult(
        "R2",
        "Dominios categóricos",
        passed,
        " ".join(problems) or "Todos los dominios se respetan.",
    )


def check_numeric(df: pd.DataFrame) -> CheckResult:
    problems: list[str] = []
    for name in COLUMNS:
        spec = SPECS[name]
        if spec.logical_type not in ("int", "float"):
            continue
        series = df[name]
        blanked = _blank_as_missing(series)
        num = _numeric(blanked)
        unparsed = blanked.notna() & num.isna()
        if unparsed.any():
            problems.append(f"{name}: {int(unparsed.sum())} valores no numéricos.")
        n_missing = int(num.isna().sum())
        allowed_missing = spec.nullable
        if n_missing and not allowed_missing:
            problems.append(f"{name}: {n_missing} valores ausentes (no admite nulos).")
        non_null = num.dropna()
        if spec.minimum is not None and non_null.lt(spec.minimum).any():
            problems.append(
                f"{name}: {int(non_null.lt(spec.minimum).sum())} valores bajo el "
                f"mínimo {spec.minimum}."
            )
        if spec.maximum is not None and non_null.gt(spec.maximum).any():
            problems.append(
                f"{name}: {int(non_null.gt(spec.maximum).sum())} valores sobre el "
                f"máximo {spec.maximum}."
            )
    passed = not problems
    return CheckResult(
        "R3",
        "Números: parseo y rango",
        passed,
        " ".join(problems) or "Números válidos dentro de rango.",
    )


def check_unique_ids(df: pd.DataFrame) -> CheckResult:
    ids = df["customerID"]
    n_dups = int(ids.duplicated().sum())
    non_matching = int(
        (~ids.astype(str).map(lambda v: bool(CUSTOMER_ID_PATTERN.match(v)))).sum()
    )
    passed = n_dups == 0 and non_matching == 0
    detail = "customerID único y con formato válido."
    if n_dups:
        detail = f"{n_dups} customerID duplicados."
    if non_matching:
        detail += f" {non_matching} customerID con formato inválido."
    return CheckResult("R4", "Unicidad y formato de customerID", passed, detail)


def check_no_duplicates(df: pd.DataFrame) -> CheckResult:
    n_dups = int(df.duplicated().sum())
    passed = n_dups == 0
    detail = "No hay filas duplicadas." if passed else f"{n_dups} filas duplicadas."
    return CheckResult("R5", "Filas duplicadas", passed, detail)


def check_coherence(df: pd.DataFrame) -> CheckResult:
    problems: list[str] = []
    inconsistent = df["MultipleLines"].eq("No phone service") != df["PhoneService"].eq(
        "No"
    )
    if inconsistent.any():
        problems.append(
            f"{int(inconsistent.sum())} filas con MultipleLines/PhoneService "
            "incoherentes."
        )
    no_internet = df["InternetService"].eq("No")
    for field in INTERNET_DEPENDENT_FIELDS:
        inconsistent = df[field].eq("No internet service") != no_internet
        if inconsistent.any():
            problems.append(
                f"{int(inconsistent.sum())} filas con "
                f"{field}/InternetService incoherentes."
            )
    passed = not problems
    return CheckResult(
        "R6",
        "Coherencia entre campos",
        passed,
        " ".join(problems) or "Sin incoherencias entre campos.",
    )


def check_missing_totalcharges(df: pd.DataFrame) -> CheckResult:
    missing = _blank_as_missing(df["TotalCharges"]).isna()
    unexpected = missing & ~df["tenure"].eq(0)
    n_missing = int(missing.sum())
    n_unexpected = int(unexpected.sum())
    passed = n_unexpected == 0
    detail = f"{n_missing} ausencias de TotalCharges."
    if n_unexpected:
        detail += f" {n_unexpected} ausencias con tenure > 0 (no permitidas)."
    return CheckResult(
        "R7",
        "Ausencia de TotalCharges solo con tenure == 0",
        passed,
        detail,
    )


CHECKS: tuple[tuple[str, Callable[[pd.DataFrame], CheckResult]], ...] = (
    ("R1", check_schema),
    ("R2", check_domains),
    ("R3", check_numeric),
    ("R4", check_unique_ids),
    ("R5", check_no_duplicates),
    ("R6", check_coherence),
    ("R7", check_missing_totalcharges),
)


def run_all(df: pd.DataFrame) -> list[CheckResult]:
    """Ejecuta, en orden, todos los checks registrados en ``CHECKS`` (OCP).

    Añadir una regla nueva no toca esta función: se define el check y se
    registra en ``CHECKS``.
    """
    return [check(df) for _rule_id, check in CHECKS]


def build_report(results: list[CheckResult], source: str) -> str:
    approved = all(r.passed for r in results)
    status = (
        "**APROBADO** — puede avanzar a la siguiente fase."
        if approved
        else "**RECHAZADO** — queda bloqueado para la siguiente fase."
    )
    lines = [
        "# Informe de calidad de datos (T-06)",
        "",
        f"- **Fuente validada:** `{source}`",
        f"- **Estado del dataset:** {status}",
        "",
        "| Regla | Descripción | Resultado | Detalle |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.rule_id} | {r.name} | "
            f"{'Pasa' if r.passed else '**Falla**'} | {r.details} |"
        )
    lines += [
        "",
        f"Reglas cumplidas: {sum(r.passed for r in results)}/{len(results)}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    df = pd.read_csv(RAW_FILE)
    results = run_all(df)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        build_report(results, str(RAW_FILE.relative_to(PROJECT_ROOT))),
        encoding="utf-8",
    )
    for r in results:
        print(
            f"[{'PASA' if r.passed else 'FALLA'}] {r.rule_id}: {r.name} — {r.details}"
        )
    approved = all(r.passed for r in results)
    print(f"Estado del dataset: {'APROBADO' if approved else 'RECHAZADO'}")
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())
