"""Pydantic v2 request / response schemas for the V1 simulation API.

Design notes
------------
* All models are ``frozen=True`` (immutable). Every request object that
  the route handler touches is constructed once and never mutated —
  this mirrors the frozen-dataclass discipline the engine itself uses
  and keeps the API layer from becoming a side-effect soup.
* Field validators fail-fast with clear messages so the OpenAPI docs
  double as a spec.
* Response models surface exactly the engine outputs a realistic
  consumer needs (PK metrics, clearances, optional time series) —
  nothing more. Organ-by-organ time courses are deliberately excluded
  from V1: they balloon payload size and nobody has asked for them.
* Time-series inclusion is opt-in (``include_time_series=True``) so
  that default responses stay small enough to be cheap to log.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ════════════════════════════════════════════════════════════════
# ENUMS / LITERALS
# ════════════════════════════════════════════════════════════════

_VALID_ROUTES = ("iv", "oral")


def _normalize_route(value: str) -> str:
    """Accept case-insensitive route strings, reject anything unknown."""
    if not isinstance(value, str):
        raise ValueError(f"route must be a string, got {type(value).__name__}")
    lowered = value.strip().lower()
    if lowered not in _VALID_ROUTES:
        raise ValueError(
            f"route must be one of {_VALID_ROUTES}, got {value!r}"
        )
    return lowered


# ════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════════════════════════════


class PatientSpec(BaseModel):
    """Optional patient adjustment. All fields optional — if nothing is
    set the engine uses the reference male defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    age_years: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=120.0,
        description="Age in years (reference adult = 35).",
    )
    weight_kg: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=300.0,
        description="Body weight in kg (reference adult male = 70).",
    )
    gfr_ml_per_min: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=250.0,
        description=(
            "Glomerular filtration rate in mL/min. Lower values scale "
            "renal clearance down proportionally (CKD modelling)."
        ),
    )
    cyp_phenotype: Optional[str] = Field(
        default=None,
        description=(
            "Free-form CYP phenotype label (e.g. 'CYP2D6_PM'). The "
            "engine's population adapter maps known phenotypes onto "
            "fm multipliers; unknown labels are ignored."
        ),
    )
    is_smoker: Optional[bool] = Field(
        default=None,
        description="If True, CYP1A2 induction is applied.",
    )
    is_fasted: Optional[bool] = Field(
        default=None,
        description=(
            "Fed vs fasted GI state for oral simulations. When False, "
            "the absorption engine switches to the fed-state GI tract "
            "(delayed gastric emptying, raised bile-salt load in the "
            "duodenum and upper jejunum, buffered gastric pH, and "
            "larger upper-SI fluid volumes). Defaults to True (fasted) "
            "to match the literature benchmark suite."
        ),
    )


class PerpetratorSpec(BaseModel):
    """One DDI perpetrator record.

    Wraps :class:`neogen.engine.ddi.PerpetratorExposure` with friendlier
    defaults (7-day pretreatment to match clinical protocols, enzyme
    pool engaged so MBI + induction work out of the box).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    drug_name: str = Field(
        ...,
        min_length=1,
        description="Literature drug name (lowercase, e.g. 'ketoconazole').",
    )
    steady_state_conc_um: float = Field(
        ...,
        ge=0.0,
        description=(
            "Unbound liver-site steady-state concentration (µM). "
            "For reversible inhibitors this drives the R-factor; "
            "for inducers/MBIs it feeds the enzyme-pool ODE."
        ),
    )
    duration_on_h: float = Field(
        default=72.0,
        gt=0.0,
        description="Dosing window during the scored simulation.",
    )
    pretreatment_h: float = Field(
        default=168.0,
        ge=0.0,
        description=(
            "Hours of perpetrator dosing before the victim starts, so "
            "induction/MBI have reached steady state. Default 7 days."
        ),
    )
    include_enzyme_pool: bool = Field(
        default=True,
        description=(
            "Whether to evolve the CYP enzyme pool (induction + MBI). "
            "Set False for pure reversible inhibitors to skip the ODE."
        ),
    )


class SimulationRequest(BaseModel):
    """Monotherapy simulation request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    drug_name: str = Field(
        ...,
        min_length=1,
        description="Literature drug name (e.g. 'midazolam').",
    )
    dose_mg: float = Field(
        ...,
        gt=0.0,
        le=10_000.0,
        description="Dose in milligrams.",
    )
    route: str = Field(
        default="oral",
        description="Administration route: 'iv' or 'oral'.",
    )
    include_time_series: bool = Field(
        default=False,
        description=(
            "If True, the response includes the full plasma "
            "concentration vs. time arrays. Omit for smaller payloads."
        ),
    )
    include_clinical_report: bool = Field(
        default=False,
        description=(
            "If True, the response carries a layered plaintext report "
            "(header → PK → DDI → absorption → PD) rendered from the "
            "engine's ``to_clinical_report`` helper. Useful for quick "
            "CLI/log summaries without reassembling the JSON envelope."
        ),
    )
    patient: Optional[PatientSpec] = Field(
        default=None,
        description="Optional patient adjustment.",
    )

    @field_validator("route")
    @classmethod
    def _validate_route(cls, value: str) -> str:
        return _normalize_route(value)

    @field_validator("drug_name")
    @classmethod
    def _normalize_drug_name(cls, value: str) -> str:
        return value.strip().lower()


class DDISimulationRequest(BaseModel):
    """DDI simulation request — victim + one or more perpetrators."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    victim_drug_name: str = Field(..., min_length=1)
    victim_dose_mg: float = Field(..., gt=0.0, le=10_000.0)
    victim_route: str = Field(default="oral")
    perpetrators: List[PerpetratorSpec] = Field(
        ...,
        min_length=1,
        description="One or more perpetrators acting on the victim.",
    )
    include_time_series: bool = Field(default=False)
    include_clinical_report: bool = Field(
        default=False,
        description=(
            "If True, the response carries the layered plaintext report. "
            "See ``SimulationRequest.include_clinical_report``."
        ),
    )
    patient: Optional[PatientSpec] = Field(default=None)

    @field_validator("victim_route")
    @classmethod
    def _validate_route(cls, value: str) -> str:
        return _normalize_route(value)

    @field_validator("victim_drug_name")
    @classmethod
    def _normalize_drug_name(cls, value: str) -> str:
        return value.strip().lower()


# ════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ════════════════════════════════════════════════════════════════


class PKMetricsResponse(BaseModel):
    """Flat mirror of :class:`neogen.engine.simulation_result.PKMetrics`."""

    model_config = ConfigDict(frozen=True)

    cmax_mg_per_L: float
    tmax_h: float
    auc_0_inf_mg_h_per_L: float
    auc_0_last_mg_h_per_L: float
    half_life_h: float
    cl_apparent_L_per_h: float
    vss_apparent_L: float
    mrt_h: float


class ClearanceBreakdown(BaseModel):
    """Hepatic + renal split from the steady-state engine calculation."""

    model_config = ConfigDict(frozen=True)

    hepatic_L_per_h: float
    renal_L_per_h: float
    total_L_per_h: float


class TimeSeriesPayload(BaseModel):
    """Plasma concentration vs. time — only present when requested."""

    model_config = ConfigDict(frozen=True)

    time_h: Tuple[float, ...]
    plasma_conc_mg_per_L: Tuple[float, ...]


class AbsorptionSummary(BaseModel):
    """Flat mirror of a P2 :class:`AbsorptionProfile` for HTTP consumers.

    Populated only when the request used an oral route (IV schedules
    bypass the P2 absorption layer). When multiple oral doses are in
    play the response carries the *first* absorption profile — every
    dose in a V1 repeated schedule shares the same parameters, so a
    single snapshot is faithful to the engine.

    ``bcs_class`` mirrors the upstream :class:`BCSClass` enum as its
    string value (e.g. ``"BCS_II"``) so that JSON consumers don't have
    to know about Python enums.
    """

    model_config = ConfigDict(frozen=True)

    drug_name: str
    dose_mg: float
    formulation_type: str
    fa: float = Field(
        ..., description="Fraction absorbed from lumen (0-1)."
    )
    fg: float = Field(
        ..., description="Fraction escaping gut-wall metabolism (0-1)."
    )
    f_portal: float = Field(
        ...,
        description=(
            "Combined portal-side bioavailability, fa × fg. Hepatic "
            "first-pass is applied downstream inside the PBPK solver."
        ),
    )
    effective_ka_per_h: float = Field(
        ..., description="First-order absorption rate constant (1/h)."
    )
    tlag_h: float = Field(
        ..., description="Absorption lag time (h)."
    )
    bcs_class: Optional[str] = Field(
        default=None,
        description=(
            "BCS class value from the drug entity (e.g. 'BCS_II'). "
            "None when the underlying drug had no bcs_class assigned."
        ),
    )
    solver_success: bool = Field(
        ...,
        description="Whether the P2 dissolution/permeation ODE converged.",
    )


class PerpetratorSummary(BaseModel):
    """One perpetrator entry in the DDI summary.

    Mirrors the engine's :class:`PerpetratorExposure` without leaking the
    internal Drug object — only the name and the exposure parameters the
    caller already sent. A ``mechanism`` label is derived from the
    perpetrator's inhibition / induction profiles so UI consumers can
    render mechanism icons without re-running the classification.
    """

    model_config = ConfigDict(frozen=True)

    drug_name: str
    steady_state_conc_um: float
    duration_on_h: float
    pretreatment_h: float
    mechanism: str = Field(
        ...,
        description=(
            "One of 'inhibition', 'mbi', 'induction', 'mixed', or "
            "'unknown'. Derived from the perpetrator's CYP profile."
        ),
    )


class DDISummaryResponse(BaseModel):
    """Flat DDI summary surfaced on the simulation response.

    The V1 engine does not currently compute an AUCR inside the
    orchestrator — callers that need fold-change compare two simulation
    results themselves. This summary therefore only carries the
    perpetrator list and mechanism attribution. The ``aucr`` field is
    populated only when the caller sends a prior monotherapy AUC to
    compare against (future extension).
    """

    model_config = ConfigDict(frozen=True)

    n_perpetrators: int
    perpetrators: Tuple[PerpetratorSummary, ...]
    dominant_mechanism: Optional[str] = Field(
        default=None,
        description=(
            "Single label summarizing the DDI: 'inhibition' if every "
            "perpetrator is reversible, 'induction' if every perpetrator "
            "is an inducer, 'mbi' if any is mechanism-based, 'mixed' if "
            "multiple classes are present. None for empty lists."
        ),
    )


class ClinicalEffectResponse(BaseModel):
    """One triggered clinical effect from the toxicity evaluator."""

    model_config = ConfigDict(frozen=True)

    organ: str
    severity: str
    symptom: str
    triggered_at_h: float
    triggered_concentration_mg_per_L: float


class PDOutcomeResponse(BaseModel):
    """Flat mirror of the P4 :class:`ClinicalOutcome`.

    Every field is optional because a drug's ``PDProfile`` may have any
    subset of (effect compartment + Emax), (receptor kinetics), and
    (toxicity thresholds) populated. A ``verdict`` string is echoed at
    the top level so downstream UIs can render a one-line summary
    without reassembling the individual scalars.
    """

    model_config = ConfigDict(frozen=True)

    verdict: str

    peak_effect_fraction: Optional[float] = Field(
        default=None,
        description="effect_peak / Emax — fraction of max PD effect reached.",
    )
    time_to_peak_effect_h: Optional[float] = None
    hysteresis_lag_h: Optional[float] = Field(
        default=None,
        description="t_effect_max - t_plasma_max (h), clamped to ≥ 0.",
    )
    effect_site_cmax_mg_per_L: Optional[float] = None

    receptor_peak_occupancy: Optional[float] = Field(
        default=None,
        description="Max receptor occupancy fraction (0-1).",
    )
    receptor_time_above_therapeutic_h: Optional[float] = None
    receptor_irreversible: Optional[bool] = None

    overall_severity: Optional[str] = Field(
        default=None,
        description="'mild' / 'moderate' / 'severe' / 'critical' or None.",
    )
    therapeutic_index: Optional[float] = Field(
        default=None,
        description="Severe-threshold / EC50 ratio. Higher = safer.",
    )
    n_toxicity_effects: int = 0
    triggered_effects: Tuple[ClinicalEffectResponse, ...] = ()


class SimulationResponse(BaseModel):
    """Canonical response envelope for ``/simulate`` and ``/simulate_ddi``.

    Always includes PK metrics and clearance; the absorption / DDI / PD
    sub-objects are populated only when the corresponding layer ran.
    Time series is opt-in via ``include_time_series`` on the request.
    """

    model_config = ConfigDict(frozen=True)

    engine_version: str
    drug_name: str
    dose_mg: float
    route: str
    pk_metrics: PKMetricsResponse
    clearance: ClearanceBreakdown
    ddi_applied: bool = False
    n_perpetrators: int = 0
    time_series: Optional[TimeSeriesPayload] = None

    # ── Optional enrichment from the sub-layers ──────────────────
    absorption: Optional[AbsorptionSummary] = Field(
        default=None,
        description=(
            "P2 absorption summary for oral schedules. None for IV."
        ),
    )
    ddi_summary: Optional[DDISummaryResponse] = Field(
        default=None,
        description=(
            "P3 DDI summary for /simulate_ddi calls. None for monotherapy."
        ),
    )
    pd_outcome: Optional[PDOutcomeResponse] = Field(
        default=None,
        description=(
            "P4 clinical outcome — effect, receptor, and toxicity "
            "scalars. None when the drug has no PDProfile attached."
        ),
    )
    clinical_report: Optional[Tuple[str, ...]] = Field(
        default=None,
        description=(
            "Layered plaintext summary lines produced by "
            "``SimulationResultV2.to_clinical_report``. Populated only "
            "when the request sets ``include_clinical_report=True``."
        ),
    )


# ════════════════════════════════════════════════════════════════
# META / ERROR MODELS
# ════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    """Simple liveness record that also carries the engine version so
    clients can assert compatibility."""

    model_config = ConfigDict(frozen=True)

    status: str
    engine_version: str


class DrugListResponse(BaseModel):
    """The list of literature drugs supported by the API."""

    model_config = ConfigDict(frozen=True)

    count: int
    drugs: Tuple[str, ...]


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for all 4xx/5xx responses."""

    model_config = ConfigDict(frozen=True)

    error: str
    detail: str


# ════════════════════════════════════════════════════════════════
# BENCHMARK MODELS
# ════════════════════════════════════════════════════════════════


class BenchmarkMetadata(BaseModel):
    """Metadata block from a ``reports/benchmarks/v1_benchmark_*.json``."""

    model_config = ConfigDict(frozen=True)

    timestamp_utc: str
    engine_version: str
    host: Optional[str] = None
    git_sha: Optional[str] = None
    notes: Optional[str] = None


class BenchmarkHeadline(BaseModel):
    """Headline gate snapshot — combined PK + DDI view."""

    model_config = ConfigDict(frozen=True)

    total_scored: int
    total_within_tolerance: int
    combined_pct_within_tolerance: float
    passes_v1_gate: bool


class BenchmarkGate(BaseModel):
    """One benchmark gate (PK or DDI).

    Mirrors the subset of the raw gate dict that the API wants to surface
    without echoing the full per-case table. Downstream callers that need
    per-case detail can fetch the raw artefact directly from the server
    filesystem; the API response is intentionally summary-only.
    """

    model_config = ConfigDict(frozen=True)

    engine_version: str
    n_cases: int
    n_successful: int
    passes_v1_gate: bool
    total_scored: Optional[int] = None
    total_within_tolerance: Optional[int] = None
    overall_pct_within_tolerance: Optional[float] = None
    overall_gmfe: Optional[float] = None


class BenchmarkLatestResponse(BaseModel):
    """Response for ``GET /benchmarks/latest``.

    Returns the headline scores from the most recent benchmark run the
    server can see on disk plus the PK and DDI gate summaries. Callers
    should treat ``filename`` as opaque — it is exposed purely so log
    readers can correlate the API response with an on-disk artefact.
    """

    model_config = ConfigDict(frozen=True)

    filename: str
    metadata: BenchmarkMetadata
    headline: BenchmarkHeadline
    pk: BenchmarkGate
    ddi: BenchmarkGate
