"""Rutas canónicas de los artefactos del proyecto (T-24).

Centraliza la ubicación de los datos derivados, modelos, predicciones y
reportes que hoy se definían repetidamente en cada módulo. Los módulos importan
desde aquí la ruta (o conjunto de rutas) que necesitan, de modo que cualquier
cambio de ubicación se hace en un único punto.

Los atributos de módulo (p. ej. `src.data.clean.RAW_FILE`) mantienen los nombres
que ya usaban los tests y demás consumidores (compatibilidad).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Datos crudos inmutables (T-03)
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROVENANCE_FILE = PROJECT_ROOT / "data" / "raw" / "PROVENANCE.yaml"

# Datos procesados (T-07/T-08)
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"
PROCESSING_LOG = PROJECT_ROOT / "data" / "processed" / "processing_log.json"

# Features derivadas (T-10)
FEATURES_FILE = PROJECT_ROOT / "data" / "features" / "churn_features.csv"
FEATURE_LOG = PROJECT_ROOT / "data" / "features" / "feature_log.json"

# Particiones (T-11)
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_FILE = SPLITS_DIR / "train.csv"
VAL_FILE = SPLITS_DIR / "validation.csv"
TEST_FILE = SPLITS_DIR / "test.csv"
PARTITION_LOG = SPLITS_DIR / "partition_log.json"

# Modelo seleccionado exportado (T-14)
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Predicciones e inferencia (T-15)
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
PREDICTIONS_FILE = PREDICTIONS_DIR / "predictions.csv"
PREDICTION_LOG = PREDICTIONS_DIR / "prediction_log.json"

# Tracking de experimentos MLflow (T-17, no versionado)
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Reportes (reports/)
AUDIT_REPORT = PROJECT_ROOT / "reports" / "audit_raw.md"
QUALITY_REPORT = PROJECT_ROOT / "reports" / "quality_report.md"
EDA_REPORT = PROJECT_ROOT / "reports" / "eda.md"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
BASELINE_REPORT_MD = PROJECT_ROOT / "reports" / "baseline.md"
BASELINE_REPORT_JSON = PROJECT_ROOT / "reports" / "baseline_metrics.json"
EXPERIMENTS_JSON = PROJECT_ROOT / "reports" / "experiments.json"
TRAINING_REPORT = PROJECT_ROOT / "reports" / "training.md"
SELECTION_JSON = PROJECT_ROOT / "reports" / "selection.json"
SELECTION_MD = PROJECT_ROOT / "reports" / "selection.md"
PERSISTENCE_REPORT_MD = PROJECT_ROOT / "reports" / "persistence.md"
PERSISTENCE_REPORT_JSON = PROJECT_ROOT / "reports" / "persistence.json"
MONITORING_REPORT_MD = PROJECT_ROOT / "reports" / "monitoring.md"
MONITORING_REPORT_JSON = PROJECT_ROOT / "reports" / "monitoring.json"
