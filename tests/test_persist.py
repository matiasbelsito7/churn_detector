"""Pruebas de la persistencia en PostgreSQL (T-16).

Cubren la lógica pura y las funciones que tocan la base usando un cursor
ficticio; no requieren un contenedor PostgreSQL real.
"""

import json
from unittest import mock

import pytest
from src.serving import persist


def sample_log(tmp_path) -> tuple[str, dict]:
    features = {
        "path": "data/features/churn_features.csv",
        "sha256": "abc123",
        "rows": 3,
    }
    log = {
        "task": "T-15",
        "model": "churn-logistic-regression@production",
        "model_version": "production",
        "threshold": 0.5,
        "input_features": features,
    }
    return tmp_path, log


@pytest.fixture
def csv_and_log(tmp_path):
    log = {
        "task": "T-15",
        "model": "churn-logistic-regression@production",
        "model_version": "production",
        "threshold": 0.5,
        "input_features": {
            "path": "data/features/churn_features.csv",
            "sha256": "abc123",
            "rows": 2,
        },
    }
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "customerID,churn_prob,churn_class\n"
        "0001-AAAAA,0.7311,Yes\n"
        "0002-BBBBB,0.3420,No\n",
        encoding="utf-8",
    )
    return csv_path, log


def test_build_dsn_includes_credentials():
    dsn = persist.build_dsn(host="db", port=5433, user="u", password="p", db="d")
    assert dsn == "host=db port=5433 user=u password=p dbname=d"


def test_prediction_rows_parses_csv(csv_and_log):
    csv_path, log = csv_and_log
    rows = persist.prediction_rows(csv_path, log)
    assert rows == [
        (
            "0001-AAAAA",
            0.7311,
            "Yes",
            "churn-logistic-regression@production",
            "production",
            "abc123",
            0.5,
        ),
        (
            "0002-BBBBB",
            0.342,
            "No",
            "churn-logistic-regression@production",
            "production",
            "abc123",
            0.5,
        ),
    ]


def test_load_predictions_inserts_rows(csv_and_log, monkeypatch, tmp_path):
    csv_path, log = csv_and_log
    log_file = tmp_path / "prediction_log.json"
    log_file.write_text(json.dumps(log), encoding="utf-8")
    monkeypatch.setattr(persist, "PREDICTIONS_FILE", csv_path)
    monkeypatch.setattr(persist, "LOG_FILE", log_file)

    cursor = mock.Mock()
    cursor.execute.side_effect = None
    cursor.fetchone.return_value = (2,)
    conn = mock.Mock()
    conn.cursor.return_value = cursor

    loaded, sources = persist.load_predictions(conn)

    assert (loaded, sources) == (2, 2)
    cursor.executemany.assert_called_once()
    insert_sql = cursor.execute.call_args_list[0][0][0]
    assert "INSERT INTO source_files" in insert_sql
    insert_sql = cursor.execute.call_args_list[1][0][0]
    assert "SELECT count(*) FROM source_files" in insert_sql
    rows = cursor.executemany.call_args.args[1]
    assert len(rows) == 2


def test_verification_queries_runs_all_checks():
    cursor = mock.Mock()
    cursor.fetchone.side_effect = [(5,), (5,), (0,), (5,), (0.1, 0.9)]
    cursor.fetchall.return_value = [("No", 3), ("Yes", 2)]
    conn = mock.Mock()
    conn.cursor.return_value = cursor

    checks = persist.verification_queries(conn)

    assert checks["predicciones"] == 5
    assert checks["clientes_distintos"] == 5
    assert checks["clases_invalidas"] == 0
    assert checks["predicciones_con_origen"] == 5
    assert checks["distribucion_por_clase"] == {"No": 3, "Yes": 2}
    assert checks["rango_probabilidad"] == [0.1, 0.9]
    assert cursor.execute.call_count == 6


def test_build_report_reflects_loaded_and_checks():
    checks = {
        "predicciones": 5,
        "clientes_distintos": 5,
        "clases_invalidas": 0,
        "predicciones_con_origen": 5,
        "distribucion_por_clase": {"No": 3, "Yes": 2},
        "rango_probabilidad": [0.1, 0.9],
    }
    report = persist.build_report((5, 1), checks)
    assert "# Persistencia en PostgreSQL (T-16)" in report
    assert "Predicciones cargadas: **5** (fuentes: 1)." in report
    assert "Yes=2" in report
    assert "[0.1000, 0.9000]" in report


def test_main_writes_reports_with_mocked_db(csv_and_log, monkeypatch, tmp_path):
    csv_path, log = csv_and_log
    monkeypatch.setattr(persist, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(persist, "PREDICTIONS_FILE", csv_path)
    monkeypatch.setattr(persist, "LOG_FILE", "unused.json")
    report_json = tmp_path / "persistence.json"
    report_md = tmp_path / "persistence.md"
    monkeypatch.setattr(persist, "REPORT_JSON", report_json)
    monkeypatch.setattr(persist, "REPORT_MD", report_md)

    load_predictions = mock.Mock(return_value=(5, 1))
    verification_queries = mock.Mock(
        return_value={
            "predicciones": 5,
            "clientes_distintos": 5,
            "clases_invalidas": 0,
            "predicciones_con_origen": 5,
            "distribucion_por_clase": {"No": 3, "Yes": 2},
            "rango_probabilidad": [0.1, 0.9],
        }
    )
    with (
        mock.patch.object(persist, "load_predictions", load_predictions),
        mock.patch.object(persist, "verification_queries", verification_queries),
        mock.patch("src.serving.persist.psycopg") as psycopg_mock,
    ):
        conn = mock.MagicMock()
        psycopg_mock.connect.return_value.__enter__.return_value = conn
        assert persist.main() == 0

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["task"] == "T-16"
    assert payload["loaded"] == {"predictions": 5, "sources": 1}
    md = report_md.read_text(encoding="utf-8")
    assert "Predicciones cargadas: **5**" in md
