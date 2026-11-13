"""DDI pair AUCR endpoint.

POST /v2/ddi — runs the victim alone and under the perpetrator's
steady-state CLint modulation, returning the predicted AUCR.

Uses the low-level ``DDIOrchestrator + PBPKSolver`` path (same as
the P3 acceptance gate and :class:`Tier2Harness`) because
OrchestratorV2 does not yet wire DDI end-to-end.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException

from neogen.api.v2.schemas import DDIRequest, DDIResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ddi", tags=["ddi"])


@router.post(
    "",
    response_model=DDIResponse,
    summary="Predict DDI AUC ratio for a victim/perpetrator pair",
)
async def predict_ddi(request: DDIRequest) -> DDIResponse:
    """Run the V1 DDI engine for a single pair.

    Victim is dosed via its declared route at steady-state perpetrator
    exposure. For induction and MBI, the enzyme pool is pre-equilibrated
    to its plateau before the victim dose is applied.
    """
    from neogen.engine.ddi.ddi_orchestrator import DDIOrchestrator
    from neogen.engine.drug_factory import DrugFactory
    from neogen.engine.dose_scheduler import DoseScheduler
    from neogen.engine.hepatic_clearance import HepaticClearanceEngine
    from neogen.engine.orchestrator_v2 import ENGINE_VERSION
    from neogen.engine.pbpk_solver import PBPKSolver, SolverConfig
    from neogen.engine.simulation_result import PKCalculator
    from neogen.core.constants.reference_physiology import build_reference_male

    # ── Build entities ─────────────────────────────────────────
    try:
        factory = DrugFactory()
        victim = factory.create(request.victim_name)
        perpetrator = factory.create(request.perpetrator_name)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Drug lookup failed: {exc}. Use /v2/drugs to list "
                f"supported drugs."
            ),
        ) from exc

    physiology = build_reference_male()
    scheduler = DoseScheduler()
    if request.victim_route == "iv_bolus":
        schedule = scheduler.iv_bolus(
            request.victim_name, request.victim_dose_mg
        )
    elif request.victim_route == "oral":
        schedule = scheduler.single_oral(
            request.victim_name, request.victim_dose_mg
        )
    else:
        schedule = scheduler.iv_infusion(
            request.victim_name, request.victim_dose_mg
        )

    solver = PBPKSolver(
        SolverConfig(
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            n_timepoints=2000,
            t_max_multiplier=80.0,
        )
    )

    # ── Baseline AUC (victim alone) ────────────────────────────
    try:
        baseline_out = solver.solve(victim, schedule, physiology)
        auc_baseline = PKCalculator.compute(
            baseline_out.time_h,
            baseline_out.concentrations[1],
            dose_mg=request.victim_dose_mg,
        ).auc_0_inf_mg_h_per_L
    except Exception as exc:
        logger.exception("Baseline DDI solve failed")
        raise HTTPException(
            status_code=500, detail=f"baseline solve: {exc}"
        ) from exc

    if auc_baseline <= 0:
        raise HTTPException(
            status_code=500,
            detail="Baseline AUC non-positive — engine solver returned zero",
        )

    # ── Build CLint provider under perpetrator ─────────────────
    ddi_orch = DDIOrchestrator()
    hepatic = HepaticClearanceEngine()
    base_clint_L_per_h = hepatic.compute(victim, physiology).clint_scaled_L_per_h

    try:
        provider = _build_clint_provider(
            ddi_orch=ddi_orch,
            request=request,
            victim=victim,
            perpetrator=perpetrator,
            base_clint_L_per_h=base_clint_L_per_h,
        )
    except Exception as exc:
        logger.exception("DDI provider build failed")
        raise HTTPException(
            status_code=500, detail=f"ddi provider: {exc}"
        ) from exc

    # ── Victim under perpetrator ───────────────────────────────
    try:
        ddi_out = solver.solve(
            victim, schedule, physiology, clint_provider=provider
        )
        auc_ddi = PKCalculator.compute(
            ddi_out.time_h,
            ddi_out.concentrations[1],
            dose_mg=request.victim_dose_mg,
        ).auc_0_inf_mg_h_per_L
    except Exception as exc:
        logger.exception("DDI solve failed")
        raise HTTPException(
            status_code=500, detail=f"ddi solve: {exc}"
        ) from exc

    aucr_predicted = auc_ddi / auc_baseline if auc_baseline > 0 else 0.0

    # ── Optional: match against reference cohort ───────────────
    ref = _lookup_reference_pair(
        request.victim_name, request.perpetrator_name, request.mode
    )
    within_2_fold: Optional[bool] = None
    clinical_aucr: Optional[float] = None
    citation: Optional[str] = None
    if ref is not None:
        clinical_aucr = ref["clinical_aucr"]
        citation = ref.get("citation")
        if clinical_aucr > 0 and aucr_predicted > 0:
            fe = aucr_predicted / clinical_aucr
            within_2_fold = 0.5 <= fe <= 2.0

    return DDIResponse(
        victim_name=request.victim_name,
        perpetrator_name=request.perpetrator_name,
        mode=request.mode,
        auc_baseline_mg_h_per_L=round(auc_baseline, 6),
        auc_ddi_mg_h_per_L=round(auc_ddi, 6),
        aucr_predicted=round(aucr_predicted, 4),
        aucr_predicted_within_2_fold_of_clinical=within_2_fold,
        clinical_aucr=clinical_aucr,
        citation=citation,
        engine_version=ENGINE_VERSION,
    )


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _build_clint_provider(
    *,
    ddi_orch,
    request: DDIRequest,
    victim,
    perpetrator,
    base_clint_L_per_h: float,
) -> Callable[[float], float]:
    """Build the dynamic CLint provider for one DDI run."""
    if request.mode == "reversible":
        exposure = ddi_orch.build_exposure(
            perpetrator,
            steady_state_conc_um=request.perpetrator_iu_um,
            duration_on_h=request.duration_on_h,
            include_enzyme_pool=False,
        )
        return ddi_orch.build_clint_provider(
            victim=victim,
            base_clint_scaled_L_per_h=base_clint_L_per_h,
            exposures=[exposure],
        )

    # Induction or MBI — pre-equilibrate to plateau.
    exposure = ddi_orch.build_exposure(
        perpetrator,
        steady_state_conc_um=request.perpetrator_iu_um,
        duration_on_h=max(request.duration_on_h, 14 * 24),
        include_enzyme_pool=True,
    )
    raw_provider = ddi_orch.build_clint_provider(
        victim=victim,
        base_clint_scaled_L_per_h=base_clint_L_per_h,
        exposures=[exposure],
    )
    plateau = raw_provider(13 * 24)

    def steady(_t: float) -> float:
        return plateau

    return steady


def _lookup_reference_pair(
    victim_name: str, perpetrator_name: str, mode: str
) -> Optional[dict]:
    """Look up a pair in data/validation/ddi_reference_pairs.json if present."""
    path = (
        Path(__file__).resolve().parents[4]
        / "data"
        / "validation"
        / "ddi_reference_pairs.json"
    )
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception:
        return None
    v = victim_name.lower()
    p = perpetrator_name.lower()
    for entry in data.get("pairs", []):
        if (
            entry.get("victim_name", "").lower() == v
            and entry.get("perpetrator_name", "").lower() == p
            and entry.get("mode", "").lower() == mode.lower()
        ):
            return entry
    return None
