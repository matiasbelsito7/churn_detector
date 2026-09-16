"""Feature engineering reproducible derivado del EDA (T-10).

Genera `data/features/churn_features.csv` a partir del dataset procesado de
`T-08`. Cada feature nueva se justifica por un hallazgo del EDA (`T-09`,
sección 5 de `reports/eda.md`); el detalle vive en
`docs/feature_dictionary.md`.

Garantía de consistencia entrenamiento/inferencia: `engineer_features` es una
transformación pura y determinista (misma definición y orden de columnas para
cualquier entrada con el esquema del procesado), válida para aplicar de forma
idéntica en train y en inferencia.

Ejecución:
    python -m src.analysis.features
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.data.contract import COLUMNS, TARGET
from src.hashing import sha256_file
from src.paths import (
    FEATURE_LOG as LOG_FILE,
)
from src.paths import (
    FEATURES_FILE as OUTPUT_FILE,
)
from src.paths import (
    PROCESSED_FILE as INPUT_FILE,
)
from src.paths import (
    PROJECT_ROOT,
)
from src.seeds import RANDOM_SEED

# Rangos de antigüedad alineados con la figura "Tasa de churn por antigüedad"
# del EDA (T-09).
TENURE_BIN_EDGES = [0, 6, 12, 24, 36, 48, 60, 73]
TENURE_BIN_LABELS = ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59", "60-72"]

NEW_CLIENT_MAX_TENURE = 3  # hallazgo F2: churn alto en tenure <= 3 meses.
ADDON_FIELDS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport")

FEATURE_COLUMNS = (
    "tenure_bin",
    "is_new_client",
    "is_month_to_month",
    "is_electronic_check",
    "is_fiber_optic",
    "addon_missing_count",
    "avg_monthly_charge_hist",
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de ``df`` con las features derivadas añadidas.

    La operación se aplica solo sobre columnas del esquema procesado y agrega
    las features en un orden fijo, de modo que cualquier entrada compatible
    produce el mismo esquema de features.
    """
    out = df.copy()
    tenure = out["tenure"]

    out["tenure_bin"] = (
        pd.cut(
            tenure,
            bins=TENURE_BIN_EDGES,
            labels=TENURE_BIN_LABELS,
            right=False,
        )
        .astype("string")
        .astype(object)
    )
    out["is_new_client"] = np.where(tenure <= NEW_CLIENT_MAX_TENURE, "Yes", "No")
    out["is_month_to_month"] = np.where(
        out["Contract"] == "Month-to-month", "Yes", "No"
    )
    out["is_electronic_check"] = np.where(
        out["PaymentMethod"] == "Electronic check", "Yes", "No"
    )
    out["is_fiber_optic"] = np.where(
        out["InternetService"] == "Fiber optic", "Yes", "No"
    )
    out["addon_missing_count"] = sum(
        (out[field] == "No").astype(int) for field in ADDON_FIELDS
    )
    out["avg_monthly_charge_hist"] = np.where(
        tenure > 0, out["TotalCharges"] / tenure, 0.0
    )
    return out


def build_log(input_sha: str, output_sha: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "task": "T-10",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "input_sha256": input_sha,
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "output_sha256": output_sha,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "features": list(FEATURE_COLUMNS),
        "seed": RANDOM_SEED,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    df = pd.read_csv(INPUT_FILE)
    if set(COLUMNS) != set(df.columns):
        print("Error: el archivo de entrada no respeta el esquema procesado.")
        return 1

    df_features = engineer_features(df)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(OUTPUT_FILE, index=False)

    log = build_log(sha256_file(INPUT_FILE), sha256_file(OUTPUT_FILE), df_features)
    LOG_FILE.write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    suffix = f" (target {TARGET} incluido)" if TARGET in df_features.columns else ""
    print(f"Features generadas: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}" f"{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
