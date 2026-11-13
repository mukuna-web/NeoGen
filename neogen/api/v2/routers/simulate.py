"""Single-drug simulation endpoint.

POST /v2/simulate — runs the V1 OrchestratorV2 end-to-end for a
single drug + dose + route, with optional patient profile scaling
via the population module.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from neogen.api.v2.schemas import (
    PKMetricsPayload,
    PatientProfilePayload,
    SimulateRequest,
    SimulateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulate", tags=["simulate"])


@router.post(
    "",
    response_model=SimulateResponse,
    summary="Run a single-drug PK simulation",
)
async def simulate(request: SimulateRequest) -> SimulateResponse:
    """Execute the V1 engine for one drug, one schedule, optional patient."""
    from neogen.core.constants.reference_physiology import build_reference_male
    from neogen.engine.drug_factory import DrugFactory
    from neogen.engine.dose_scheduler import DoseScheduler
    from neogen.engine.orchestrator import SimulationOrchestrator
    from neogen.engine.orchestrator_v2 import ENGINE_VERSION

    # ── Build drug ─────────────────────────────────────────────
    try:
        drug = DrugFactory().create(request.drug_name)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Drug '{request.drug_name}' not available. "
                f"Use /v2/drugs to list supported drugs."
            ),
        ) from exc

    # ── Build schedule ─────────────────────────────────────────
    scheduler = DoseScheduler()
    try:
        if request.route == "iv_bolus":
            schedule = scheduler.iv_bolus(request.drug_name, request.dose_mg)
        elif request.route == "oral":
            schedule = scheduler.single_oral(request.drug_name, request.dose_mg)
        elif request.route == "iv_infusion":
            schedule = scheduler.iv_infusion(request.drug_name, request.dose_mg)
        else:  # pragma: no cover — guarded by the Literal type
            raise ValueError(f"Unsupported route: {request.route}")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"schedule build failed: {exc}"
        ) from exc

    # ── Build physiology (+ optional patient scaling) ──────────
    physiology = build_reference_male()
    if request.patient is not None:
        drug, physiology = _apply_patient_scaling(
            drug, physiology, request.patient
        )

    # ── Run engine ─────────────────────────────────────────────
    try:
        result = SimulationOrchestrator().run(drug, schedule, physiology)
    except Exception as exc:
        logger.exception("Simulation failed for %s", request.drug_name)
        raise HTTPException(
            status_code=500, detail=f"engine failure: {exc}"
        ) from exc

    pk = PKMetricsPayload(
        cmax_mg_per_L=result.pk_metrics.cmax_mg_per_L,
        tmax_h=result.pk_metrics.tmax_h,
        auc_0_inf_mg_h_per_L=result.pk_metrics.auc_0_inf_mg_h_per_L,
        auc_0_last_mg_h_per_L=result.pk_metrics.auc_0_last_mg_h_per_L,
        half_life_h=result.pk_metrics.half_life_h,
        cl_apparent_L_per_h=result.pk_metrics.cl_apparent_L_per_h,
        vss_apparent_L=result.pk_metrics.vss_apparent_L,
        mrt_h=result.pk_metrics.mrt_h,
    )

    response = SimulateResponse(
        drug_name=result.drug_name,
        dose_mg=result.dose_mg,
        route=result.route,
        engine_version=ENGINE_VERSION,
        pk_metrics=pk,
        cl_hepatic_L_per_h=result.cl_hepatic_L_per_h,
        cl_renal_L_per_h=result.cl_renal_L_per_h,
        cl_total_L_per_h=result.cl_total_L_per_h,
        solver_success=result.solver_success,
    )

    if request.return_curves:
        response = response.model_copy(
            update={
                "time_h": list(result.time_h),
                "plasma_conc_mg_per_L": list(result.plasma_conc_mg_per_L),
                "organ_profiles": _organ_profiles_to_dict(result.organ_profiles),
            }
        )

    return response


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _apply_patient_scaling(drug, physiology, payload: PatientProfilePayload):
    """Convert a :class:`PatientProfilePayload` into a scaled (drug, physiology).

    Uses the population module's scaling helpers so this endpoint
    stays aligned with the Monte Carlo path.
    """
    from neogen.core.entities.patient import (
        CYPPhenotype,
        GeneticProfile,
        OrganFunction,
        PatientProfile,
        Sex,
    )
    from neogen.population.scaling import scale_for_patient

    organ_function = (
        OrganFunction(
            hepatic=payload.organ_function.hepatic,
            renal=payload.organ_function.renal,
            cardiac=payload.organ_function.cardiac,
            pulmonary=payload.organ_function.pulmonary,
        )
        if payload.organ_function is not None
        else OrganFunction()
    )

    if payload.genetics is not None:
        genetics = GeneticProfile(
            cyp3a4=CYPPhenotype(payload.genetics.cyp3a4),
            cyp2d6=CYPPhenotype(payload.genetics.cyp2d6),
            cyp2c9=CYPPhenotype(payload.genetics.cyp2c9),
            cyp2c19=CYPPhenotype(payload.genetics.cyp2c19),
            cyp1a2=CYPPhenotype(payload.genetics.cyp1a2),
            cyp2b6=CYPPhenotype(payload.genetics.cyp2b6),
        )
    else:
        genetics = GeneticProfile()

    patient = PatientProfile(
        weight_kg=payload.weight_kg,
        height_cm=payload.height_cm,
        age_years=payload.age_years,
        sex=Sex(payload.sex),
        organ_function=organ_function,
        genetics=genetics,
        is_smoker=payload.is_smoker,
    )
    return scale_for_patient(drug, physiology, patient)


def _organ_profiles_to_dict(profiles: Dict[str, Any]) -> Dict[str, list]:
    """Convert OrganTimeCourse instances into plain JSON-friendly lists."""
    out: Dict[str, list] = {}
    for name, oc in profiles.items():
        out[name] = list(oc.concentration_mg_per_L)
    return out
