"""NeoGen V2 API subpackage.

Two ways to use this package:

1. **Standalone app** (local dev, tests, independent deployment)::

       from neogen.api.v2 import v2_app
       # or:
       uvicorn neogen.api.v2.app:v2_app --port 8001

2. **Mount on the existing production app** (single deployment)::

       from neogen.api.v2 import v2_router
       app.include_router(v2_router, prefix="/v2")

Both paths expose the identical route set under ``/v2/*``.
"""

from neogen.api.v2.app import (
    build_v2_router,
    create_v2_app,
    v2_app,
    v2_router,
)
from neogen.api.v2.schemas import (
    CohortDemographics,
    DDIRequest,
    DDIResponse,
    DrugListResponse,
    ErrorResponse,
    GeneticProfilePayload,
    HealthResponse,
    OrganFunctionPayload,
    PKMetricsPayload,
    PatientProfilePayload,
    PercentileSummaryPayload,
    PopulationRequest,
    PopulationResponse,
    ServiceInfo,
    SimulateRequest,
    SimulateResponse,
    ValidationFullRequest,
    ValidationResponse,
    ValidationTierRequest,
)

__all__ = [
    # App + router
    "v2_app",
    "v2_router",
    "build_v2_router",
    "create_v2_app",
    # Schemas
    "ServiceInfo",
    "HealthResponse",
    "DrugListResponse",
    "PKMetricsPayload",
    "OrganFunctionPayload",
    "GeneticProfilePayload",
    "PatientProfilePayload",
    "SimulateRequest",
    "SimulateResponse",
    "DDIRequest",
    "DDIResponse",
    "PercentileSummaryPayload",
    "CohortDemographics",
    "PopulationRequest",
    "PopulationResponse",
    "ValidationTierRequest",
    "ValidationFullRequest",
    "ValidationResponse",
    "ErrorResponse",
]
