"""Particiones reproducibles de entrenamiento/validación/prueba (T-11).

Genera `data/splits/{train,validation,test}.csv` a partir del dataset con
features de `T-10`, estratificando por el target y con semillas fijas.

Garantías:
- Reproducible: mismas filas por partición en cada ejecución (semilla fija).
- Sin solapamiento: cada cliente pertenece a exactamente una partición.
- Estratificación por `Churn` para preservar el desbalanceo en cada partición.

Ejecución:
    python -m src.modeling.split
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from src.analysis.features import FEATURE_COLUMNS
from src.data.contract import COLUMNS, TARGET
from src.hashing import sha256_file
from src.paths import (
    FEATURES_FILE as INPUT_FILE,
)
from src.paths import (
    PARTITION_LOG as LOG_FILE,
)
from src.paths import (
    PROJECT_ROOT,
    SPLITS_DIR,
)
from src.seeds import RANDOM_SEED

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

EXPECTED_COLUMNS = (*COLUMNS, *FEATURE_COLUMNS)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def create_partitions(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> Split:
    """Divide ``df`` en train/validation/test estratificados por el target.

    ``df`` debe incluir la columna ``TARGET``. El `test` se separa primero; a
    continuación el resto se reparte entre `train` y `validation` manteniendo
    las proporciones pedidas sobre el total.
    """
    rest, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df[TARGET],
        random_state=seed,
    )
    val_of_rest = val_frac / (train_frac + val_frac)
    train, validation = train_test_split(
        rest,
        test_size=val_of_rest,
        stratify=rest[TARGET],
        random_state=seed + 1,
    )
    return Split(train=train, validation=validation, test=test)


def build_log(
    input_sha: str,
    output_shas: dict[str, str],
    split: Split,
) -> dict[str, object]:
    counts = {
        name: int(len(getattr(split, name))) for name in ("train", "validation", "test")
    }
    return {
        "task": "T-11",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "input_sha256": input_sha,
        "output_files": {name: f"data/splits/{name}.csv" for name in counts},
        "output_sha256": output_shas,
        "rows": counts,
        "fractions": {
            "train": TRAIN_FRAC,
            "validation": VAL_FRAC,
            "test": TEST_FRAC,
        },
        "seed": RANDOM_SEED,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    df = pd.read_csv(INPUT_FILE)
    if list(df.columns) != list(EXPECTED_COLUMNS):
        print("Error: el archivo de entrada no respeta el esquema con features.")
        return 1

    split = create_partitions(df)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    output_shas: dict[str, str] = {}
    for name, part in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        path = SPLITS_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        output_shas[name] = sha256_file(path)

    log = build_log(sha256_file(INPUT_FILE), output_shas, split)
    LOG_FILE.write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Particiones generadas: train={len(split.train)}, "
        f"validation={len(split.validation)}, test={len(split.test)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
