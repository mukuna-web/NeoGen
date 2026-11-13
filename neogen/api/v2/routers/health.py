"""Health and info endpoints for the V2 API."""
from __future__ import annotations

from fastapi import APIRouter

from neogen.api.v2.schemas import HealthResponse, ServiceInfo

router = APIRouter(tags=["info"])


@router.get("/", response_model=ServiceInfo, summary="Service information")
async def service_info() -> ServiceInfo:
    """Return top-level service metadata.

    Useful for quick CLI smoke checks (``curl /v2/``) and for the
    auto-generated OpenAPI landing page.
    """
    # Imports are inside the handler so an import-time failure in
    # the engine layer does not break the /v2/ route — we still want
    # the health endpoints to respond with a visible error.
    from neogen.core.constants.literature_params import list_available_drugs
    from neogen.engine.orchestrator_v2 import ENGINE_VERSION

    return ServiceInfo(
        engine_version=ENGINE_VERSION,
        available_drugs_count=len(list_available_drugs()),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Liveness probe — exercised by Railway / Docker healthchecks.

    Returns ``status="ok"`` only when every critical engine import
    resolves successfully.
    """
    try:
        from neogen.engine.orchestrator_v2 import ENGINE_VERSION
        from neogen.engine.drug_factory import DrugFactory  # noqa: F401
        from neogen.engine.orchestrator import (  # noqa: F401
            SimulationOrchestrator,
        )
        from neogen.engine.ddi.ddi_orchestrator import (  # noqa: F401
            DDIOrchestrator,
        )
        from neogen.population.runner import PopulationRunner  # noqa: F401
        from neogen.validation.runner import ValidationRunner  # noqa: F401
    except Exception as exc:
        return HealthResponse(
            status="failing",
            engine_version="unknown",
            imports_ok=False,
            message=f"engine import failed: {exc}",
        )
    return HealthResponse(
        status="ok",
        engine_version=ENGINE_VERSION,
        imports_ok=True,
    )
