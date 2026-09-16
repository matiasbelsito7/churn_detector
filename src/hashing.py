"""Utilidades de hashing de archivos (T-24).

`sha256_file` es la utilidad de digest de artefactos usada en todas las capas
para la trazabilidad reproducible (checksums de entradas y salidas). Vive fuera
de la capa de datos para que las capas superiores no dependan de un detalle de
`src.data`.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Devuelve el digest sha256 de un archivo (streaming, files grandes OK)."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
