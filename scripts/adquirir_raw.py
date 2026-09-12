"""Descarga reproducible del dataset crudo de churn (T-03).

Descarga el archivo bruto desde la fuente documentada en ``PROVENANCE.yaml`` y
verifica su integridad contra el checksum SHA-256 versionado junto al proyecto.

Si el archivo ya existe y su checksum coincide, no se modifica (escritura única
/ inmutabilidad del raw). Toda transformación posterior debe operar sobre
artefactos derivados, no sobre este archivo.

Requiere únicamente la biblioteca estándar de Python.
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "Telco-Customer-Churn.csv"
PROVENANCE_FILE = RAW_DIR / "PROVENANCE.yaml"
RAW_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def load_expected_checksum(provenance: Path) -> str:
    """Leo el checksum SHA-256 esperado del registro de procedencia."""
    for line in provenance.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*checksum_sha256:\s*([0-9A-Fa-f]{64})\s*$", line)
        if match:
            return match.group(1)
    raise SystemExit(f"Checksum esperado no encontrado en {provenance}")


def sha256_of(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_schema(file: Path) -> tuple[int, int]:
    with file.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, len(header)


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())


def main() -> int:
    expected = load_expected_checksum(PROVENANCE_FILE)

    if RAW_FILE.exists():
        actual = sha256_of(RAW_FILE)
        if actual.upper() == expected.upper():
            rows, cols = validate_schema(RAW_FILE)
            print(
                f"OK (ya presente y valido): {RAW_FILE} ({rows} filas, {cols} columnas)"
            )
            return 0
        print(
            "Error: el archivo existente no coincide con el checksum esperado.\n"
            f"  esperado: {expected}\n  actual:   {actual}"
        )
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {RAW_URL} -> {RAW_FILE}")
    download(RAW_URL, RAW_FILE)

    actual = sha256_of(RAW_FILE)
    if actual.upper() != expected.upper():
        RAW_FILE.unlink()
        print(
            "Error: el checksum no coincide tras la descarga.\n"
            f"  esperado: {expected}\n  actual:   {actual}"
        )
        return 1

    rows, cols = validate_schema(RAW_FILE)
    print(f"OK: {RAW_FILE} descargado y verificado ({rows} filas, {cols} columnas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())