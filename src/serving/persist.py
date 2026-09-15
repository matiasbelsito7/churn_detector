"""Persistencia de resultados de predicción en PostgreSQL (T-16).

Carga las predicciones de `T-15` (`data/predictions/predictions.csv`) en la
base PostgreSQL local definida en `docker-compose.yml` y deja el esquema
consultable y trazable a su origen versionado:

- `predictions`: salida de inferencia por cliente (probabilidad y clase),
  con el modelo, su versión y el checksum del dataset de features usado.
- `source_files`: línea de datos persistidos a su origen versionado
  (path + sha256 + filas del archivo de features).

Configuración por variables de entorno (valores por defecto para el contenedor
de desarrollo local):

    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD,
    POSTGRES_DB

Arranque del contenedor y carga/verificación:

    docker compose up -d postgres
    python -m src.serving.persist
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
LOG_FILE = PROJECT_ROOT / "data" / "predictions" / "prediction_log.json"
REPORT_MD = PROJECT_ROOT / "reports" / "persistence.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "persistence.json"

DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
DB_USER = os.environ.get("POSTGRES_USER", "churn")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "churn")
DB_NAME = os.environ.get("POSTGRES_DB", "churn")


def build_dsn(
    host: str = DB_HOST,
    port: int = DB_PORT,
    user: str = DB_USER,
    password: str = DB_PASSWORD,
    db: str = DB_NAME,
) -> str:
    return f"host={host} port={port} user={user} password={password} dbname={db}"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_files (
    sha256         varchar(64) PRIMARY KEY,
    path           varchar(255) NOT NULL,
    rows           integer      NOT NULL,
    registered_at  timestamptz  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    customer_id    varchar(20)  PRIMARY KEY,
    churn_prob     numeric(8,6) NOT NULL,
    churn_class    varchar(3)   NOT NULL CHECK (churn_class IN ('Yes', 'No')),
    model          varchar(100) NOT NULL,
    model_version  varchar(20)  NOT NULL,
    source_sha256  varchar(64)  NOT NULL REFERENCES source_files(sha256),
    threshold      numeric(4,2) NOT NULL CHECK (threshold >= 0 AND threshold <= 1),
    predicted_at   timestamptz  NOT NULL DEFAULT now()
);
"""


def prediction_rows(csv_path: Path, log: dict[str, Any]) -> list[tuple[object, ...]]:
    """Extrae las filas de predicción y su metadato de trazabilidad del log."""
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header, *body = lines
    assert header.split(",") == ["customerID", "churn_prob", "churn_class"]
    features = log["input_features"]
    return [
        (
            rid,
            float(prob),
            cls,
            log["model"],
            log.get("model_version", "production"),
            features["sha256"],
            log["threshold"],
        )
        for rid, prob, cls in (line.rsplit(",", 2) for line in body)
    ]


def load_predictions(conn: psycopg.Connection) -> tuple[int, int]:
    """Carga las predicciones y su fuente; devuelve (predicciones, fuentes)."""
    log = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    rows = prediction_rows(PREDICTIONS_FILE, log)
    features = log["input_features"]
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO source_files (sha256, path, rows)
        VALUES (%s, %s, %s)
        ON CONFLICT (sha256) DO NOTHING
        """,
        (features["sha256"], features["path"], features["rows"]),
    )
    cur.executemany(
        """
        INSERT INTO predictions (
            customer_id, churn_prob, churn_class, model, model_version,
            source_sha256, threshold
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id) DO NOTHING
        """,
        rows,
    )
    cur.execute("SELECT count(*) FROM source_files")
    sources_row = cur.fetchone()
    assert sources_row is not None
    sources = int(sources_row[0])
    conn.commit()
    return len(rows), sources


def verification_queries(conn: psycopg.Connection) -> dict[str, object]:
    """Devuelve los resultados de las consultas de verificación del esquema."""
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM predictions")
    total_row = cur.fetchone()
    assert total_row is not None
    total = int(total_row[0])
    cur.execute("SELECT count(DISTINCT customer_id) FROM predictions")
    distinct_row = cur.fetchone()
    assert distinct_row is not None
    distinct = int(distinct_row[0])
    cur.execute(
        "SELECT count(*) FROM predictions WHERE churn_class NOT IN ('Yes', 'No')"
    )
    invalid_row = cur.fetchone()
    assert invalid_row is not None
    invalid_class = int(invalid_row[0])
    cur.execute(
        "SELECT count(*) FROM predictions p JOIN source_files s "
        "ON p.source_sha256 = s.sha256"
    )
    joined_row = cur.fetchone()
    assert joined_row is not None
    joined = int(joined_row[0])
    cur.execute(
        "SELECT churn_class, count(*) FROM predictions GROUP BY churn_class ORDER BY 1"
    )
    by_class = {row[0]: int(row[1]) for row in cur.fetchall()}
    cur.execute("SELECT min(churn_prob), max(churn_prob) FROM predictions")
    prob_row = cur.fetchone()
    assert prob_row is not None
    prob_min, prob_max = prob_row
    return {
        "predicciones": total,
        "clientes_distintos": distinct,
        "clases_invalidas": invalid_class,
        "predicciones_con_origen": joined,
        "distribucion_por_clase": by_class,
        "rango_probabilidad": [float(prob_min), float(prob_max)],
    }


def build_report(loaded: tuple[int, int], checks: dict[str, Any]) -> str:
    by_class = cast(dict[str, int], checks["distribucion_por_clase"])
    rango = cast(list[float], checks["rango_probabilidad"])
    return "\n".join(
        [
            "# Persistencia en PostgreSQL (T-16)",
            "",
            "Esquema cargado en el contenedor PostgreSQL local "
            "(`docker-compose.yml`): tabla `predictions` con la salida de "
            "inferencia por cliente y tabla `source_files` para la trazabilidad "
            "al origen versionado (sha256 del dataset de features).",
            "",
            f"Predicciones cargadas: **{loaded[0]}** (fuentes: {loaded[1]}).",
            "",
            "Verificación por SQL:",
            "",
            f"- Predicciones totales: {checks['predicciones']}.",
            f"- Clientes distintos: {checks['clientes_distintos']}.",
            f"- Clases fuera de {{Yes, No}}: {checks['clases_invalidas']}.",
            f"- Predicciones con origen trazable: "
            f"{checks['predicciones_con_origen']}.",
            f"- Distribución por clase: Yes={by_class.get('Yes', 0)}, "
            f"No={by_class.get('No', 0)}.",
            f"- Probabilidad en rango [0,1]: " f"[{rango[0]:.4f}, {rango[1]:.4f}].",
            "",
        ]
    )


def main() -> int:
    with psycopg.connect(build_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        loaded = load_predictions(conn)
        checks = verification_queries(conn)

    REPORT_JSON.write_text(
        json.dumps(
            {
                "task": "T-16",
                "database": {
                    "host": DB_HOST,
                    "port": DB_PORT,
                    "dbname": DB_NAME,
                    "user": DB_USER,
                },
                "loaded": {"predictions": loaded[0], "sources": loaded[1]},
                "checks": checks,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(build_report(loaded, checks), encoding="utf-8")
    print(f"Persistencia verificada: {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"  {checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
