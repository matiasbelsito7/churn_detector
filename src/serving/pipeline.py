"""Pipeline end-to-end reproducible (T-15).

Orquesta las etapas `datos → features → modelo → predicción` usando los
módulos existentes de cada capa, de modo que todo el sistema se puede
regenerar desde el dato crudo versionado con una secuencia única de comandos.

Etapas (orden):
1. `datos`: limpieza reproducible del raw versionado (`src.data.clean`).
2. `features`: feature engineering determinista (`src.analysis.features`).
3. `particiones`: particiones reproducibles (`src.modeling.split`).
4. `entrenamiento`: candidatos y registro JSON (`src.modeling.train`).
5. `tracking`: candidatos registrados en MLflow (`src.modeling.tracking`).
6. `seleccion`: modelo elegido y promovido (`src.modeling.select`).
7. `adopcion`: se aplica la configuración ganadora de `T-29` si mejora la
   evaluación (`src.modeling.adopt`, T-30).
8. `explicabilidad`: reporte SHAP del modelo seleccionado (`src.modeling.explain`).
9. `prediccion`: salida de inferencia para todos los clientes
   (`src.serving.predict`).

Reproducibilidad: todas las etapas usan entradas versionadas, semillas fijas y
configuraciones explícitas; ejecutar el pipeline dos veces con los mismos datos
produce artefactos con los mismos checksums y las mismas métricas.

Ejecución (todo o por etapa):
    python -m src.serving.pipeline
    python -m src.serving.pipeline features prediccion
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable


def stage_clean() -> None:
    from src.data import clean

    clean.main()


def stage_features() -> None:
    from src.analysis import features

    features.main()


def stage_split() -> None:
    from src.modeling import split

    split.main()


def stage_train() -> None:
    from src.modeling import train

    train.main()


def stage_tracking() -> None:
    from src.modeling import tracking

    tracking.main()


def stage_select() -> None:
    from src.modeling import select

    select.main()


def stage_adopt() -> None:
    from src.modeling import adopt

    adopt.main()


def stage_explain() -> None:
    from src.modeling import explain

    explain.main()


def stage_predict() -> None:
    from src.serving import predict

    predict.main()


PIPELINE_STAGES: tuple[tuple[str, Callable[[], None]], ...] = (
    ("datos", stage_clean),
    ("features", stage_features),
    ("particiones", stage_split),
    ("entrenamiento", stage_train),
    ("tracking", stage_tracking),
    ("seleccion", stage_select),
    ("adopcion", stage_adopt),
    ("explicabilidad", stage_explain),
    ("prediccion", stage_predict),
)

STAGE_NAMES = tuple(name for name, _ in PIPELINE_STAGES)


def run_pipeline(stages: Iterable[str]) -> int:
    """Ejecuta las etapas indicadas en el orden canónico del pipeline."""
    upto: dict[str, Callable[[], None]] = dict(PIPELINE_STAGES)
    for name in stages:
        if name not in upto:
            raise ValueError(
                f"Etapa desconocida: {name}. Válidas: {', '.join(STAGE_NAMES)}"
            )
        runner = upto[name]
        print(f"\n=== {name} ===")
        runner()
    return 0


def main() -> int:
    args = sys.argv[1:]
    stages = args if args else STAGE_NAMES
    return run_pipeline(stages)


if __name__ == "__main__":
    sys.exit(main())
