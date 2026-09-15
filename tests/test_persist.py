"""Pruebas de la persistencia en PostgreSQL (T-16)."""

from src.serving.persist import build_dsn, prediction_rows


def test_build_dsn_defaults():
    dsn = build_dsn()
    assert dsn.startswith("host=localhost port=5432")
    assert "user=churn" in dsn
    assert "dbname=churn" in dsn


def test_build_dsn_overrides():
    dsn = build_dsn(host="db", port=5433, user="u", password="p", db="d")
    assert "host=db" in dsn
    assert "port=5433" in dsn
    assert "user=u" in dsn
    assert "password=p" in dsn
    assert "dbname=d" in dsn


def test_prediction_rows_from_cvs(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "customerID,churn_prob,churn_class\n"
        "7590-VHVEG,0.1406,No\n"
        "5575-GNVDE,0.7331,Yes\n",
        encoding="utf-8",
    )
    log = {
        "model": "churn-logistic-regression@production",
        "threshold": 0.5,
        "input_features": {
            "sha256": "abc123",
            "path": "data/features/churn_features.csv",
            "rows": 3,
        },
    }
    rows = prediction_rows(csv_path, log)
    assert rows[0] == (
        "7590-VHVEG",
        0.1406,
        "No",
        "churn-logistic-regression@production",
        "production",
        "abc123",
        0.5,
    )
    assert rows[1][4] == "production"
    assert len(rows) == 2


def test_prediction_rows_uses_log_model_version_when_present(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "customerID,churn_prob,churn_class\nX-1,0.5,Yes\n", encoding="utf-8"
    )
    log = {
        "model": "churn-logistic-regression@production",
        "model_version": "3",
        "threshold": 0.4,
        "input_features": {"sha256": "def456", "path": "x.csv", "rows": 1},
    }
    rows = prediction_rows(csv_path, log)
    assert rows[0][4] == "3"
    assert rows[0][6] == 0.4
