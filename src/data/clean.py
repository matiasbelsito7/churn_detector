"""Limpieza reproducible del dataset (T-07) y tratamiento de faltantes (T-08).

Aplica al raw versionado los pasos de normalización derivados del audit
(`T-05`) y genera un artefacto derivado trazable en `data/processed/`:

1. `TotalCharges`: las celdas de solo espacios pasan a ausencia (NaN) y la
   columna a flotante.
2. `SeniorCitizen`: la codificación numérica `0/1` se normaliza a `No/Yes`,
   coherente con el resto de los binarios.
3. Valores faltantes de `TotalCharges` imputados a 0 (ausencia determinística
   con `tenure == 0`; ver `docs/missing_values.md`).

El raw no se modifica. Antes de limpiar se ejecutan los checks de calidad
(`T-06`); si el raw no los supera, el proceso aborta (dataset bloqueado).

Ejecución:
    python -m src.data.clean
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.data.quality_checks import run_all
from src.seeds import RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"
LOG_FILE = PROJECT_ROOT / "data" / "processed" / "processing_log.json"

STEPS = [
    "TotalCharges: celdas de solo espacios convertidas a ausencia (NaN) y "
    "columna convertida a flotante.",
    "SeniorCitizen: codificación 0/1 normalizada a No/Yes para un esquema "
    "coherente con el resto de binarios.",
    "Valores faltantes de TotalCharges imputados a 0 (ausencia determinística "
    "con tenure == 0).",
    "El resto de columnas se mantiene sin cambios respecto del contrato.",
]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia limpia de ``df`` sin modificar la entrada."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
        errors="coerce",
    )
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int).map({0: "No", 1: "Yes"})
    return out


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_log(
    input_sha: str,
    output_sha: str,
    df_clean: pd.DataFrame,
) -> dict[str, object]:
    return {
        "task": "T-07",
        "input_file": str(RAW_FILE.relative_to(PROJECT_ROOT)),
        "input_sha256": input_sha,
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "output_sha256": output_sha,
        "rows": int(len(df_clean)),
        "columns": int(len(df_clean.columns)),
        "seed": RANDOM_SEED,
        "steps": STEPS,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    df = pd.read_csv(RAW_FILE)
    results = run_all(df)
    if not all(r.passed for r in results):
        print("Error: el raw no pasa los checks de calidad; se bloquea la limpieza.")
        return 1

    df_clean = clean_raw(df)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)

    log = build_log(sha256_file(RAW_FILE), sha256_file(OUTPUT_FILE), df_clean)
    LOG_FILE.write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Limpieza completa: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
