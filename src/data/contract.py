"""Definición del contrato de datos en forma reproducible.

Espejo en código del documento `docs/data_contract.md`. Lo consumen el audit
(`T-05`) y los checks de calidad automatizados (`T-06`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    logical_type: str  # "id" | "category" | "int" | "float" | "target"
    allowed: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    nullable: bool = False
    description: str = ""


TARGET = "Churn"
POSITIVE_CLASS = "Yes"

COLUMNS = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET,
)

SPECS: dict[str, FieldSpec] = {spec.name: spec for spec in (
    FieldSpec("customerID", "id", description="Identificador único del cliente."),
    FieldSpec("gender", "category", ("Female", "Male"), description="Género del cliente."),
    FieldSpec("SeniorCitizen", "int", minimum=0, maximum=1, description="1 si adulto mayor, 0 si no."),
    FieldSpec("Partner", "category", ("No", "Yes"), description="Tiene pareja."),
    FieldSpec("Dependents", "category", ("No", "Yes"), description="Tiene personas a cargo."),
    FieldSpec("tenure", "int", minimum=0, maximum=72, description="Meses de antigüedad."),
    FieldSpec("PhoneService", "category", ("No", "Yes"), description="Tiene servicio telefónico."),
    FieldSpec("MultipleLines", "category", ("No", "No phone service", "Yes"), description="Tiene múltiples líneas."),
    FieldSpec("InternetService", "category", ("DSL", "Fiber optic", "No"), description="Tipo de servicio de internet."),
    FieldSpec("OnlineSecurity", "category", ("No", "No internet service", "Yes"), description="Seguridad online."),
    FieldSpec("OnlineBackup", "category", ("No", "No internet service", "Yes"), description="Respaldo online."),
    FieldSpec("DeviceProtection", "category", ("No", "No internet service", "Yes"), description="Protección de dispositivo."),
    FieldSpec("TechSupport", "category", ("No", "No internet service", "Yes"), description="Soporte técnico."),
    FieldSpec("StreamingTV", "category", ("No", "No internet service", "Yes"), description="Streaming de TV."),
    FieldSpec("StreamingMovies", "category", ("No", "No internet service", "Yes"), description="Streaming de películas."),
    FieldSpec("Contract", "category", ("Month-to-month", "One year", "Two year"), description="Tipo de contrato."),
    FieldSpec("PaperlessBilling", "category", ("No", "Yes"), description="Facturación sin papel."),
    FieldSpec("PaymentMethod", "category", ("Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"), description="Método de pago."),
    FieldSpec("MonthlyCharges", "float", minimum=18.25, maximum=118.75, description="Cargo mensual."),
    FieldSpec("TotalCharges", "float", minimum=0.0, nullable=True, description="Cargo total acumulado; ausente si tenure == 0."),
    FieldSpec(TARGET, "target", ("No", "Yes"), description="Target: abandonó el servicio (Yes)."),
)}

INTERNET_DEPENDENT_FIELDS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)