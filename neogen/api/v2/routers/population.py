"""Population Monte Carlo endpoint.

POST /v2/population — wraps :class:`PopulationRunner` for HTTP clients.
Default cohort size is 50 patients; capped at 500 to keep request
latency under ~60 seconds.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from neogen.api.v2.schemas import (
    CohortDemographics,
    PercentileSummaryPayload,
    PopulationRequest,
    PopulationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/population", tags=["population"])


@router.post(
    "",
    response_model=PopulationResponse,
    summary="Run a virtual-patient Monte Carlo simulation",
)
async def simulate_population(
    request: PopulationRequest,
) -> PopulationResponse:
    """Execute :class:`PopulationRunner` for the given drug + cohort size."""
    from neogen.engine.orchestrator_v2 import ENGINE_VERSION
    from neogen.population.runner import PopulationRunner
    from neogen.population.sampler import PopulationSampler

    try:
        sampler = PopulationSampler()
        population = sampler.sample(
            n=request.n_patients,
            seed=request.seed,
            label=request.label,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"sampling failed: {exc}"
        ) from exc

    try:
        runner = PopulationRunner()
        result = runner.simulate(
            drug_name=request.drug_name,
            dose_mg=request.dose_mg,
            route=request.route,
            population=population,
            label=request.label,
        )
    except Exception as exc:
        logger.exception("Population simulation failed")
        raise HTTPException(
            status_code=500, detail=f"engine failure: {exc}"
        ) from exc

    demographics_dict = population.demographic_summary()
    demographics = CohortDemographics(
        n=demographics_dict["n"],
        male_fraction=demographics_dict["male_fraction"],
        smoker_fraction=demographics_dict["smoker_fraction"],
        mean_weight_kg=demographics_dict["mean_weight_kg"],
        mean_age_years=demographics_dict["mean_age_years"],
        cyp2d6_pm_fraction=demographics_dict["cyp2d6_pm_fraction"],
        cyp2d6_um_fraction=demographics_dict["cyp2d6_um_fraction"],
    )

    summaries: dict[str, dict] = {}
    for key, summary in result.headline_summaries().items():
        payload = PercentileSummaryPayload(
            n=summary.n,
            median=_safe(summary.median),
            p05=_safe(summary.p05),
            p25=_safe(summary.p25),
            p75=_safe(summary.p75),
            p95=_safe(summary.p95),
            geomean=_safe(summary.geomean),
            geomean_cv_pct=_safe(summary.geomean_cv_pct),
            min_value=_safe(summary.min_value),
            max_value=_safe(summary.max_value),
        )
        summaries[key] = payload.model_dump()

    return PopulationResponse(
        drug_name=result.drug_name,
        dose_mg=result.dose_mg,
        route=result.route,
        n_requested=result.n_requested,
        n_completed=result.n_completed,
        n_errored=result.n_errored,
        seed=result.seed,
        label=result.label,
        runtime_s=round(result.runtime_s, 2),
        demographics=demographics,
        summaries=summaries,
        engine_version=ENGINE_VERSION,
    )


def _safe(v):
    """Collapse NaN / inf to None for JSON safety."""
    import math

    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v
