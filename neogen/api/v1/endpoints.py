"""Route handlers for the V1 simulation API.

All handlers are thin: parse + validate the Pydantic request, translate
to engine primitives, call :class:`OrchestratorV2.run`, map the frozen
result back onto the response schema. No business logic lives here —
that is the whole point of the engine-as-library pattern.

Error strategy
--------------
Unknown drug names, unknown routes, or unsupported patient fields are
translated to HTTP 400 with a structured :class:`ErrorResponse`. Any
exception escaping the engine itself is translated to HTTP 500 and
surfaces the exception class name + message, but never an internal
stack trace.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from neogen.api.v1.schemas import (
    AbsorptionSummary,
    BenchmarkGate,
    BenchmarkHeadline,
    BenchmarkLatestResponse,
    BenchmarkMetadata,
    ClearanceBreakdown,
    ClinicalEffectResponse,
    DDISimulationRequest,
    DDISummaryResponse,
    DrugListResponse,
    ErrorResponse,
    HealthResponse,
    PatientSpec,
    PDOutcomeResponse,
    PerpetratorSpec,
    PerpetratorSummary,
    PKMetricsResponse,
    SimulationRequest,
    SimulationResponse,
    TimeSeriesPayload,
)
from neogen.core.constants.literature_params import LITERATURE_DRUGS
from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.patient import (
    CYPPhenotype,
    GeneticProfile,
    OrganFunction,
    PatientProfile,
    Sex,
)
from neogen.engine.absorption.absorption_profile import AbsorptionProfile
from neogen.engine.ddi.ddi_orchestrator import (
    DDIOrchestrator,
    PerpetratorExposure,
)
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.orchestrator_v2 import (
    ENGINE_VERSION,
    OrchestratorV2,
    SimulationResultV2,
)
from neogen.engine.pd.pd_result import ClinicalOutcome
from neogen.engine.simulation_result import PKMetrics, SimulationResult

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# SHARED ENGINE SINGLETONS
# ════════════════════════════════════════════════════════════════
# These are stateless — constructing them once per process is a tiny
# perf win and keeps the import graph warm. They are safe to share
# across concurrent requests because every ``run`` call operates on
# fresh inputs and returns a frozen result.

_drug_factory = DrugFactory()
_scheduler = DoseScheduler()
_engine = OrchestratorV2()
_ddi = DDIOrchestrator()

# The GFR default matches the reference male used in the benchmark
# suite. Over-riding via PatientSpec.gfr_ml_per_min scales the renal
# arm linearly.
_REFERENCE_GFR_ML_PER_MIN = 120.0


# Directory holding serialized benchmark artefacts. Overridable via the
# ``NEOGEN_BENCHMARKS_DIR`` environment variable so tests can point at a
# scratch directory without touching the production layout.
_DEFAULT_BENCHMARKS_DIR = (
    Path(__file__).resolve().parents[3] / "reports" / "benchmarks"
)


def _benchmarks_dir() -> Path:
    """Resolve the benchmark directory at request time.

    We read the env var on every call instead of caching it so tests can
    use ``monkeypatch.setenv`` between cases without resetting module
    state.
    """
    override = os.environ.get("NEOGEN_BENCHMARKS_DIR")
    if override:
        return Path(override)
    return _DEFAULT_BENCHMARKS_DIR


# ════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════

router = APIRouter()


# ── Helper: patient translation ────────────────────────────────


def _phenotype_for(label: Optional[str]) -> Optional[CYPPhenotype]:
    """Map a free-form phenotype label to the engine enum.

    Returns ``None`` when the label does not match a known enum value.
    Callers silently drop unknown labels — this mirrors the rest of the
    engine, which treats missing metadata as 'use the default'.
    """
    if not label:
        return None
    normalized = label.strip().upper()
    # Accept both the bare enum name and the dotted form CYP2D6_PM.
    if normalized in CYPPhenotype.__members__:
        return CYPPhenotype[normalized]
    # Friendly aliases — PM/IM/EM/UM.
    aliases = {
        "PM": CYPPhenotype.POOR,
        "IM": CYPPhenotype.INTERMEDIATE,
        "EM": CYPPhenotype.NORMAL,
        "NM": CYPPhenotype.NORMAL,
        "UM": CYPPhenotype.ULTRA_RAPID,
    }
    suffix = normalized.rsplit("_", 1)[-1]
    return aliases.get(suffix)


def _patient_spec_to_profile(spec: Optional[PatientSpec]) -> Optional[PatientProfile]:
    """Translate the API patient spec into an engine ``PatientProfile``.

    The spec is intentionally a flat subset — the full PatientProfile
    has ten fields, but realistic API callers only care about weight,
    age, smoker status, renal function (via GFR), and maybe a CYP
    phenotype label. Everything else stays at the engine defaults.

    Returns ``None`` when the spec itself is ``None`` — that lets the
    orchestrator fast-path the patient adapter away.
    """
    if spec is None:
        return None

    defaults = PatientProfile()

    weight = spec.weight_kg if spec.weight_kg is not None else defaults.weight_kg
    age = spec.age_years if spec.age_years is not None else defaults.age_years
    smoker = spec.is_smoker if spec.is_smoker is not None else defaults.is_smoker
    fasted = spec.is_fasted if spec.is_fasted is not None else defaults.is_fasted

    # Renal scaling: the engine consumes OrganFunction.renal as a
    # fraction of normal (1.0 = normal). GFR is reported in mL/min and
    # we scale linearly against the reference 120 mL/min. Clamp to
    # [0, 1.5] so that absurd values can't produce negative CL.
    if spec.gfr_ml_per_min is not None:
        renal_fraction = spec.gfr_ml_per_min / _REFERENCE_GFR_ML_PER_MIN
        renal_fraction = max(0.0, min(renal_fraction, 1.5))
    else:
        renal_fraction = defaults.organ_function.renal

    organ = OrganFunction(
        hepatic=defaults.organ_function.hepatic,
        renal=renal_fraction,
        cardiac=defaults.organ_function.cardiac,
        pulmonary=defaults.organ_function.pulmonary,
    )

    # CYP phenotype: we accept a single label like "CYP2D6_PM" and only
    # apply it to the matching isoform. Anything else stays normal.
    genetics = defaults.genetics
    if spec.cyp_phenotype:
        normalized = spec.cyp_phenotype.strip().upper()
        target_iso = None
        for iso in ("CYP2D6", "CYP2C19", "CYP2C9", "CYP2B6", "CYP1A2", "CYP3A4"):
            if normalized.startswith(iso):
                target_iso = iso.lower()
                break
        phenotype = _phenotype_for(spec.cyp_phenotype)
        if target_iso is not None and phenotype is not None:
            genetics = GeneticProfile(
                **{
                    **{
                        "cyp2d6": defaults.genetics.cyp2d6,
                        "cyp2c19": defaults.genetics.cyp2c19,
                        "cyp2c9": defaults.genetics.cyp2c9,
                        "cyp2b6": defaults.genetics.cyp2b6,
                        "cyp1a2": defaults.genetics.cyp1a2,
                        "cyp3a4": defaults.genetics.cyp3a4,
                    },
                    target_iso: phenotype,
                }
            )

    return PatientProfile(
        weight_kg=weight,
        height_cm=defaults.height_cm,
        age_years=age,
        sex=Sex.MALE,
        organ_function=organ,
        genetics=genetics,
        is_smoker=bool(smoker),
        is_fasted=bool(fasted),
        measured_albumin_g_per_dl=defaults.measured_albumin_g_per_dl,
        measured_aag_mg_per_dl=defaults.measured_aag_mg_per_dl,
    )


# ── Helper: engine → response translation ─────────────────────


def _pk_metrics_to_response(pk: PKMetrics) -> PKMetricsResponse:
    return PKMetricsResponse(
        cmax_mg_per_L=pk.cmax_mg_per_L,
        tmax_h=pk.tmax_h,
        auc_0_inf_mg_h_per_L=pk.auc_0_inf_mg_h_per_L,
        auc_0_last_mg_h_per_L=pk.auc_0_last_mg_h_per_L,
        half_life_h=pk.half_life_h,
        cl_apparent_L_per_h=pk.cl_apparent_L_per_h,
        vss_apparent_L=pk.vss_apparent_L,
        mrt_h=pk.mrt_h,
    )


def _absorption_to_response(
    profile: AbsorptionProfile,
    *,
    bcs_class_label: Optional[str],
) -> AbsorptionSummary:
    """Translate a single :class:`AbsorptionProfile` into its API mirror.

    We only surface scalar summary fields — the full per-segment
    dissolved/undissolved/portal traces are intentionally not exposed,
    because they balloon the response size and every UI we've drafted
    only needs Fa/Fg/F_portal/ka/tlag. V2 may add an opt-in
    ``include_absorption_trace`` flag if a real use case emerges.
    """
    return AbsorptionSummary(
        drug_name=profile.drug_name,
        dose_mg=profile.dose_mg,
        formulation_type=profile.formulation_type.value,
        fa=profile.fa,
        fg=profile.fg,
        f_portal=profile.f_portal,
        effective_ka_per_h=profile.effective_ka_per_h,
        tlag_h=profile.tlag_h,
        bcs_class=bcs_class_label,
        solver_success=profile.solver_success,
    )


def _classify_perpetrator(exposure: PerpetratorExposure) -> str:
    """Label a perpetrator's dominant mechanism for UI consumption.

    This is a pragmatic single-label classifier: it checks the drug's
    inhibition and induction profiles in the canonical order MBI →
    induction → reversible inhibition. A perpetrator that is both a
    reversible inhibitor *and* an inducer is labelled ``"mixed"`` —
    that's rare enough in clinical practice that we want it to be
    visibly flagged instead of silently picking one.
    """
    drug = exposure.perpetrator
    inhibition = drug.inhibition
    induction = drug.induction

    is_mbi = bool(getattr(inhibition, "is_mbi", False))
    is_inducer = bool(getattr(induction, "is_inducer", False))
    is_inhibitor = bool(getattr(inhibition, "is_inhibitor", False))

    if is_mbi:
        if is_inducer:
            return "mixed"
        return "mbi"
    if is_inducer and is_inhibitor:
        return "mixed"
    if is_inducer:
        return "induction"
    if is_inhibitor:
        return "inhibition"
    return "unknown"


def _ddi_to_response(
    exposures: tuple[PerpetratorExposure, ...]
) -> Optional[DDISummaryResponse]:
    """Translate the orchestrator's perpetrator tuple into the API summary.

    Returns ``None`` for monotherapy (empty tuple) so the response
    field stays omitted from the JSON envelope instead of materialising
    as an empty object.
    """
    if not exposures:
        return None

    perpetrator_rows = tuple(
        PerpetratorSummary(
            drug_name=e.perpetrator.name,
            steady_state_conc_um=e.steady_state_conc_um,
            duration_on_h=e.duration_on_h,
            pretreatment_h=e.pretreatment_h,
            mechanism=_classify_perpetrator(e),
        )
        for e in exposures
    )

    mechanisms = {row.mechanism for row in perpetrator_rows}
    if len(mechanisms) == 1:
        (dominant,) = mechanisms
    elif "mbi" in mechanisms:
        dominant = "mbi"
    elif mechanisms == {"inhibition", "induction"}:
        dominant = "mixed"
    else:
        dominant = "mixed"

    return DDISummaryResponse(
        n_perpetrators=len(perpetrator_rows),
        perpetrators=perpetrator_rows,
        dominant_mechanism=dominant,
    )


def _pd_outcome_to_response(
    outcome: Optional[ClinicalOutcome],
) -> Optional[PDOutcomeResponse]:
    """Translate a P4 :class:`ClinicalOutcome` into the API mirror.

    Returns ``None`` when the drug had no PDProfile attached so that the
    response omits the ``pd_outcome`` field entirely. That keeps the
    monotherapy-IV midazolam response small for callers that only care
    about PK — the most common path in the benchmark suite.
    """
    if outcome is None:
        return None

    effect_site_cmax: Optional[float] = None
    if outcome.effect_course is not None:
        effect_site_cmax = outcome.effect_course.summary.effect_site_cmax_mg_per_L

    receptor_time_above: Optional[float] = None
    receptor_irreversible: Optional[bool] = None
    if outcome.receptor_trace is not None:
        receptor_time_above = (
            outcome.receptor_trace.summary.time_above_therapeutic_h
        )
        receptor_irreversible = outcome.receptor_trace.irreversible

    triggered: tuple[ClinicalEffectResponse, ...] = ()
    n_effects = 0
    if outcome.toxicity_report is not None:
        triggered = tuple(
            ClinicalEffectResponse(
                organ=effect.organ.value,
                severity=effect.severity.value,
                symptom=effect.symptom,
                triggered_at_h=effect.triggered_at_h,
                triggered_concentration_mg_per_L=effect.triggered_concentration_mg_per_L,
            )
            for effect in outcome.toxicity_report.triggered_effects
        )
        n_effects = outcome.toxicity_report.n_effects

    severity_label: Optional[str] = None
    if outcome.overall_severity is not None:
        severity_label = outcome.overall_severity.value

    return PDOutcomeResponse(
        verdict=outcome.verdict,
        peak_effect_fraction=outcome.peak_effect_fraction,
        time_to_peak_effect_h=outcome.time_to_peak_effect_h,
        hysteresis_lag_h=outcome.hysteresis_lag_h,
        effect_site_cmax_mg_per_L=effect_site_cmax,
        receptor_peak_occupancy=outcome.receptor_peak_occupancy,
        receptor_time_above_therapeutic_h=receptor_time_above,
        receptor_irreversible=receptor_irreversible,
        overall_severity=severity_label,
        therapeutic_index=outcome.therapeutic_index,
        n_toxicity_effects=n_effects,
        triggered_effects=triggered,
    )


def _result_to_response(
    result: SimulationResultV2,
    *,
    include_time_series: bool,
    route: str,
    drug: Optional[object] = None,
    include_clinical_report: bool = False,
    perpetrator_exposures: Optional[tuple] = None,
) -> SimulationResponse:
    """Map the engine result onto the API envelope.

    The ``route`` parameter echoes the normalized request value
    (``iv`` / ``oral``). We do *not* read the engine's stored route
    string because internally it is an enum-ish label like ``IV_BOLUS``
    that does not match the public contract.

    ``drug`` is the victim :class:`Drug` entity from the request handler.
    We use it to pull the ``BCSClass`` label for the absorption summary —
    that lives on the drug, not on the absorption profile, because the
    profile itself is formulation-specific while BCS is intrinsic to the
    molecule.
    """
    pk_inner: SimulationResult = result.pk_result
    time_series_payload: Optional[TimeSeriesPayload] = None
    if include_time_series:
        time_series_payload = TimeSeriesPayload(
            time_h=tuple(pk_inner.time_h),
            plasma_conc_mg_per_L=tuple(pk_inner.plasma_conc_mg_per_L),
        )

    # Absorption: use the first profile (V1 repeat-dose schedules share
    # the same parameters across doses, so one snapshot is faithful).
    absorption_payload: Optional[AbsorptionSummary] = None
    if result.absorption:
        bcs_label: Optional[str] = None
        if drug is not None:
            bcs_attr = getattr(drug, "bcs_class", None)
            if bcs_attr is not None:
                bcs_label = getattr(bcs_attr, "value", str(bcs_attr))
        absorption_payload = _absorption_to_response(
            result.absorption[0],
            bcs_class_label=bcs_label,
        )

    clinical_report_payload: Optional[tuple] = None
    if include_clinical_report:
        # ``to_clinical_report`` already returns an immutable tuple of
        # lines, which is exactly what Pydantic wants for a frozen
        # ``Tuple[str, ...]`` field — no copy necessary.
        clinical_report_payload = result.to_clinical_report()

    return SimulationResponse(
        engine_version=result.engine_version,
        drug_name=pk_inner.drug_name,
        dose_mg=pk_inner.dose_mg,
        route=route,
        pk_metrics=_pk_metrics_to_response(pk_inner.pk_metrics),
        clearance=ClearanceBreakdown(
            hepatic_L_per_h=pk_inner.cl_hepatic_L_per_h,
            renal_L_per_h=pk_inner.cl_renal_L_per_h,
            total_L_per_h=pk_inner.cl_total_L_per_h,
        ),
        ddi_applied=result.has_ddi,
        n_perpetrators=result.ddi.n_perpetrators if result.ddi is not None else 0,
        time_series=time_series_payload,
        absorption=absorption_payload,
        ddi_summary=_ddi_to_response(
            tuple(perpetrator_exposures) if perpetrator_exposures else ()
        ),
        pd_outcome=_pd_outcome_to_response(result.clinical_outcome),
        clinical_report=clinical_report_payload,
    )


# ── Helper: request → schedule/drug/perpetrator construction ───


def _build_schedule(drug_name: str, dose_mg: float, route: str):
    if route == "iv":
        return _scheduler.iv_bolus(drug_name, dose_mg=dose_mg)
    return _scheduler.single_oral(drug_name, dose_mg=dose_mg)


def _require_drug(drug_name: str):
    """Look up a literature drug or raise HTTP 400.

    We check the literature database first so callers get a crisp error
    instead of an opaque engine exception — ``DrugFactory.create`` itself
    will happily try ML fallbacks if RDKit is present, which we don't
    want to expose through the V1 API.
    """
    if drug_name not in LITERATURE_DRUGS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown drug {drug_name!r}. "
                f"Call GET /v1/drugs for the supported list."
            ),
        )
    return _drug_factory.create(drug_name)


def _load_latest_benchmark() -> BenchmarkLatestResponse:
    """Return a summary of the newest ``v1_benchmark_*.json`` artefact.

    Raises ``HTTPException(404)`` when no benchmark artefacts are
    present (fresh checkout, empty directory) and ``HTTPException(500)``
    when the newest file is unreadable or structurally invalid.
    """
    directory = _benchmarks_dir()
    if not directory.is_dir():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Benchmarks directory {directory} does not exist. "
                "Run the V1 benchmark suite to populate it."
            ),
        )

    candidates = sorted(directory.glob("v1_benchmark_*.json"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No benchmark artefacts found in {directory}. "
                "Run the V1 benchmark suite to populate it."
            ),
        )

    # Filenames are ISO-8601 timestamps, so lexicographic sort == newest
    # last. This avoids a filesystem-stat call per candidate and stays
    # deterministic across machines with drifting mtimes (common when
    # restoring a git worktree).
    latest = candidates[-1]

    try:
        with latest.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("failed to load benchmark %s", latest)
        raise HTTPException(
            status_code=500,
            detail=f"Could not read benchmark artefact: {type(exc).__name__}: {exc}",
        ) from exc

    try:
        metadata = BenchmarkMetadata(**payload["metadata"])
        headline = BenchmarkHeadline(**payload["headline"])
        pk_gate = BenchmarkGate(
            engine_version=payload["pk"]["engine_version"],
            n_cases=payload["pk"]["n_cases"],
            n_successful=payload["pk"]["n_successful"],
            passes_v1_gate=payload["pk"]["passes_v1_gate"],
            total_scored=payload["pk"].get("total_scored"),
            total_within_tolerance=payload["pk"].get("total_within_tolerance"),
            overall_pct_within_tolerance=payload["pk"].get(
                "overall_pct_within_tolerance"
            ),
            overall_gmfe=payload["pk"].get("overall_gmfe"),
        )
        ddi_gate = BenchmarkGate(
            engine_version=payload["ddi"]["engine_version"],
            n_cases=payload["ddi"]["n_cases"],
            n_successful=payload["ddi"]["n_successful"],
            passes_v1_gate=payload["ddi"]["passes_v1_gate"],
            total_scored=payload["ddi"].get("total_scored"),
            total_within_tolerance=payload["ddi"].get("total_within_tolerance"),
            overall_pct_within_tolerance=payload["ddi"].get(
                "overall_pct_within_tolerance"
            ),
            overall_gmfe=payload["ddi"].get("overall_gmfe"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.exception("benchmark artefact %s has unexpected shape", latest)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Benchmark artefact {latest.name} is missing expected "
                f"fields: {type(exc).__name__}: {exc}"
            ),
        ) from exc

    return BenchmarkLatestResponse(
        filename=latest.name,
        metadata=metadata,
        headline=headline,
        pk=pk_gate,
        ddi=ddi_gate,
    )


def _build_perpetrator_exposure(spec: PerpetratorSpec) -> PerpetratorExposure:
    drug = _require_drug(spec.drug_name.strip().lower())
    return _ddi.build_exposure(
        drug,
        steady_state_conc_um=spec.steady_state_conc_um,
        duration_on_h=spec.duration_on_h,
        pretreatment_h=spec.pretreatment_h,
        include_enzyme_pool=spec.include_enzyme_pool,
    )


# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Liveness probe + engine version",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", engine_version=ENGINE_VERSION)


@router.get(
    "/drugs",
    response_model=DrugListResponse,
    tags=["meta"],
    summary="List literature drugs supported by the V1 engine",
)
def list_drugs() -> DrugListResponse:
    drugs = tuple(sorted(LITERATURE_DRUGS.keys()))
    return DrugListResponse(count=len(drugs), drugs=drugs)


@router.get(
    "/benchmarks/latest",
    response_model=BenchmarkLatestResponse,
    tags=["meta"],
    summary="Most recent V1 benchmark headline + PK/DDI gate summaries",
    responses={
        404: {"model": ErrorResponse, "description": "No benchmark artefacts found"},
        500: {"model": ErrorResponse, "description": "Benchmark file unreadable"},
    },
)
def latest_benchmark() -> BenchmarkLatestResponse:
    """Surface the newest ``v1_benchmark_*.json`` artefact as JSON.

    The response is a summary: metadata, headline gate, and the PK + DDI
    gate snapshots. Per-case tables are intentionally omitted — callers
    that need them should read the artefact directly from disk.
    """
    return _load_latest_benchmark()


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    tags=["simulation"],
    summary="Run a monotherapy PBPK + PD simulation",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Engine failure"},
    },
)
def simulate(request: SimulationRequest) -> SimulationResponse:
    drug = _require_drug(request.drug_name)
    schedule = _build_schedule(request.drug_name, request.dose_mg, request.route)
    physiology = build_reference_male()
    patient = _patient_spec_to_profile(request.patient)

    try:
        result = _engine.run(drug, schedule, physiology, patient=patient)
    except Exception as exc:
        logger.exception("simulate failed for %r", request.drug_name)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    return _result_to_response(
        result,
        include_time_series=request.include_time_series,
        route=request.route,
        drug=drug,
        include_clinical_report=request.include_clinical_report,
    )


@router.post(
    "/simulate_ddi",
    response_model=SimulationResponse,
    tags=["simulation"],
    summary="Run a simulation with one or more DDI perpetrators",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Engine failure"},
    },
)
def simulate_ddi(request: DDISimulationRequest) -> SimulationResponse:
    victim = _require_drug(request.victim_drug_name)
    schedule = _build_schedule(
        request.victim_drug_name,
        request.victim_dose_mg,
        request.victim_route,
    )
    physiology = build_reference_male()
    patient = _patient_spec_to_profile(request.patient)

    exposures: List[PerpetratorExposure] = [
        _build_perpetrator_exposure(p) for p in request.perpetrators
    ]

    try:
        result = _engine.run(
            victim,
            schedule,
            physiology,
            perpetrator_exposures=exposures,
            patient=patient,
        )
    except Exception as exc:
        logger.exception(
            "simulate_ddi failed for victim=%r perps=%r",
            request.victim_drug_name,
            [p.drug_name for p in request.perpetrators],
        )
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    return _result_to_response(
        result,
        include_time_series=request.include_time_series,
        route=request.victim_route,
        drug=victim,
        include_clinical_report=request.include_clinical_report,
        perpetrator_exposures=tuple(exposures),
    )
