"""Pydantic v2 request/response schemas for the NeoGen V2 API.

All request bodies and response payloads for ``neogen.api.v2`` live
here. Endpoints depend on this module exclusively for their I/O
shape so the OpenAPI contract is trivially auditable.

Design rules:
  * Pydantic v2 only. No ``class Config`` — use ``model_config``
    when needed.
  * Every field has an explicit type and a ``Field(..., description=)``
    docstring so the auto-generated OpenAPI docs are useful.
  * Response models never contain numpy arrays. Concentration
    profiles are plain ``list[float]`` so the payload is
    JSON-serializable by default.
  * Optional fields default to ``None`` so clients can omit them;
    the router decides what the engine default means.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ════════════════════════════════════════════════════════════════
# SHARED TYPES
# ════════════════════════════════════════════════════════════════


RouteLiteral = Literal["iv_bolus", "oral", "iv_infusion"]
DDIMode = Literal["reversible", "induction", "mbi"]
Sex = Literal["male", "female"]
Phenotype = Literal[
    "poor_metabolizer",
    "intermediate_metabolizer",
    "normal_metabolizer",
    "ultra_rapid_metabolizer",
]


class PKMetricsPayload(BaseModel):
    """Serialized view of :class:`neogen.engine.simulation_result.PKMetrics`."""

    model_config = ConfigDict(frozen=True)

    cmax_mg_per_L: float = Field(..., description="Peak plasma concentration (mg/L)")
    tmax_h: float = Field(..., description="Time to peak (h)")
    auc_0_inf_mg_h_per_L: float = Field(
        ..., description="AUC extrapolated to infinity (mg·h/L)"
    )
    auc_0_last_mg_h_per_L: float = Field(
        ..., description="AUC to last measurable time (mg·h/L)"
    )
    half_life_h: float = Field(..., description="Terminal elimination half-life (h)")
    cl_apparent_L_per_h: float = Field(
        ..., description="Apparent clearance CL/F (L/h)"
    )
    vss_apparent_L: float = Field(..., description="Apparent Vss (L)")
    mrt_h: float = Field(..., description="Mean residence time (h)")


class OrganFunctionPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    hepatic: float = Field(1.0, ge=0.0, le=1.0)
    renal: float = Field(1.0, ge=0.0, le=1.0)
    cardiac: float = Field(1.0, ge=0.0, le=1.0)
    pulmonary: float = Field(1.0, ge=0.0, le=1.0)


class GeneticProfilePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    cyp3a4: Phenotype = "normal_metabolizer"
    cyp2d6: Phenotype = "normal_metabolizer"
    cyp2c9: Phenotype = "normal_metabolizer"
    cyp2c19: Phenotype = "normal_metabolizer"
    cyp1a2: Phenotype = "normal_metabolizer"
    cyp2b6: Phenotype = "normal_metabolizer"


class PatientProfilePayload(BaseModel):
    """Optional patient override for V2 simulations.

    When omitted the engine uses the ICRP reference male.
    """

    model_config = ConfigDict(frozen=True)

    weight_kg: float = Field(70.0, gt=0.0, le=300.0)
    height_cm: float = Field(176.0, gt=0.0, le=250.0)
    age_years: float = Field(30.0, ge=0.0, le=110.0)
    sex: Sex = "male"
    organ_function: Optional[OrganFunctionPayload] = None
    genetics: Optional[GeneticProfilePayload] = None
    is_smoker: bool = False


# ════════════════════════════════════════════════════════════════
# HEALTH + INFO
# ════════════════════════════════════════════════════════════════


class ServiceInfo(BaseModel):
    """Root info payload — what the service is and who built it."""

    service: str = "neogen-v2-api"
    engine_version: str
    available_drugs_count: int
    supported_routes: List[str] = Field(
        default_factory=lambda: ["iv_bolus", "oral", "iv_infusion"]
    )
    docs_url: str = "/v2/docs"


class HealthResponse(BaseModel):
    """Liveness probe payload. Used by Kubernetes / Railway."""

    status: Literal["ok", "degraded", "failing"] = "ok"
    engine_version: str
    imports_ok: bool = True
    message: str = ""


class DrugListResponse(BaseModel):
    """Listing of drugs with literature parameter overrides."""

    count: int
    drugs: List[str]


# ════════════════════════════════════════════════════════════════
# SIMULATE — single drug
# ════════════════════════════════════════════════════════════════


class SimulateRequest(BaseModel):
    """Single-drug simulation request."""

    model_config = ConfigDict(frozen=True)

    drug_name: str = Field(
        ...,
        description="Drug name (must be in LITERATURE_DRUGS for V1).",
        examples=["midazolam"],
    )
    dose_mg: float = Field(..., gt=0.0, description="Dose in mg")
    route: RouteLiteral = Field("iv_bolus", description="Administration route")
    patient: Optional[PatientProfilePayload] = Field(
        None,
        description=(
            "Optional patient override. Omit to run against the ICRP "
            "reference male. When provided, the engine scales "
            "physiology and CLint per the population module."
        ),
    )
    return_curves: bool = Field(
        False,
        description=(
            "If true, the response includes plasma concentration-time "
            "curves. Large payload; set false for batch jobs."
        ),
    )
    t_max_h: Optional[float] = Field(
        None,
        gt=0.0,
        description="Optional override for simulation duration (h).",
    )


class SimulateResponse(BaseModel):
    drug_name: str
    dose_mg: float
    route: str
    engine_version: str
    pk_metrics: PKMetricsPayload
    cl_hepatic_L_per_h: float
    cl_renal_L_per_h: float
    cl_total_L_per_h: float
    solver_success: bool

    # Optional curves (populated only when return_curves=true)
    time_h: Optional[List[float]] = None
    plasma_conc_mg_per_L: Optional[List[float]] = None
    # Organ profiles map name → list of concentrations on the same time grid
    organ_profiles: Optional[dict] = None


# ════════════════════════════════════════════════════════════════
# DDI — pair AUCR
# ════════════════════════════════════════════════════════════════


class DDIRequest(BaseModel):
    """DDI pair AUCR request. Matches the low-level DDIOrchestrator path."""

    model_config = ConfigDict(frozen=True)

    victim_name: str = Field(..., examples=["midazolam"])
    victim_dose_mg: float = Field(5.0, gt=0.0)
    victim_route: RouteLiteral = Field("iv_bolus")
    perpetrator_name: str = Field(..., examples=["ketoconazole"])
    perpetrator_iu_um: float = Field(
        ...,
        ge=0.0,
        description="Unbound perpetrator concentration at hepatic inlet (µM)",
    )
    mode: DDIMode = Field("reversible")
    duration_on_h: float = Field(
        500.0,
        gt=0.0,
        description=(
            "Perpetrator exposure window. For reversible: any value "
            "≥ victim t½ is fine. For induction/MBI: 14×24 h recommended "
            "to reach enzyme pool plateau."
        ),
    )


class DDIResponse(BaseModel):
    victim_name: str
    perpetrator_name: str
    mode: DDIMode
    auc_baseline_mg_h_per_L: float
    auc_ddi_mg_h_per_L: float
    aucr_predicted: float
    aucr_predicted_within_2_fold_of_clinical: Optional[bool] = None
    clinical_aucr: Optional[float] = Field(
        None,
        description=(
            "If the requested pair matches a known reference, the "
            "clinical AUCR is echoed for convenience."
        ),
    )
    citation: Optional[str] = None
    engine_version: str


# ════════════════════════════════════════════════════════════════
# POPULATION — Monte Carlo
# ════════════════════════════════════════════════════════════════


class PercentileSummaryPayload(BaseModel):
    n: int
    median: Optional[float] = None
    p05: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    geomean: Optional[float] = None
    geomean_cv_pct: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class PopulationRequest(BaseModel):
    """Virtual-patient Monte Carlo request."""

    model_config = ConfigDict(frozen=True)

    drug_name: str = Field(..., examples=["midazolam"])
    dose_mg: float = Field(..., gt=0.0)
    route: RouteLiteral = Field("iv_bolus")
    n_patients: int = Field(
        50, ge=1, le=500, description="Virtual patient count (capped at 500)"
    )
    seed: Optional[int] = Field(
        None, description="Deterministic seed for reproducibility"
    )
    label: str = Field("", description="Free-text label stored with the result")


class CohortDemographics(BaseModel):
    n: int
    male_fraction: float
    smoker_fraction: float
    mean_weight_kg: float
    mean_age_years: float
    cyp2d6_pm_fraction: float
    cyp2d6_um_fraction: float


class PopulationResponse(BaseModel):
    drug_name: str
    dose_mg: float
    route: str
    n_requested: int
    n_completed: int
    n_errored: int
    seed: Optional[int] = None
    label: str = ""
    runtime_s: float
    demographics: CohortDemographics
    summaries: dict = Field(
        ...,
        description="Map metric_name → PercentileSummaryPayload for "
        "cmax, AUC, half_life, CL, Vss",
    )
    engine_version: str


# ════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════


class ValidationTierRequest(BaseModel):
    """Request for per-tier validation endpoints."""

    model_config = ConfigDict(frozen=True)

    save_report: bool = Field(
        False,
        description="If true, persist JSON + Markdown under reports/validation/.",
    )


class ValidationFullRequest(ValidationTierRequest):
    include_tier1: bool = True
    include_tier2: bool = True
    include_tier3: bool = True
    include_tier4: bool = True


class ValidationResponse(BaseModel):
    """Wraps the JSON form of ValidationReport."""

    engine_version: str
    run_timestamp_utc: str
    overall_pass: bool
    report: dict = Field(
        ...,
        description="Full ValidationReport as a JSON-serializable dict.",
    )


class ErrorResponse(BaseModel):
    """Standard error payload for non-2xx responses."""

    error: str
    detail: Optional[str] = None
