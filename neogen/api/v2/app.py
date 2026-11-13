"""NeoGen V2 FastAPI application.

This module exposes two symbols for downstream use:

    v2_app    — a standalone :class:`fastapi.FastAPI` instance.
                Run it directly via:
                    uvicorn neogen.api.v2.app:v2_app --port 8001

    v2_router — an :class:`fastapi.APIRouter` that bundles every
                V2 route and can be mounted on any existing FastAPI
                app:
                    from neogen.api.v2 import v2_router
                    app.include_router(v2_router, prefix="/v2")

The standalone app is the recommended local-dev path and isolates
the new engine from the production predict.py deployment. The
router variant is the minimum-invasive production integration —
a one-line ``include_router`` call on the existing production app
is all that is required when you are ready to ship v2 to clients.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neogen.api.v2.routers import (
    ddi as ddi_router,
    drugs as drugs_router,
    health as health_router,
    population as population_router,
    simulate as simulate_router,
    validation as validation_router,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# ROUTER ASSEMBLY
# ════════════════════════════════════════════════════════════════


def build_v2_router() -> APIRouter:
    """Assemble the V2 router from every sub-module.

    Mountable on any external FastAPI app::

        app.include_router(build_v2_router(), prefix="/v2")

    The returned router has no prefix itself — the caller decides
    where it lives. For the standalone app the prefix is ``/v2`` so
    endpoints match the external production contract.
    """
    router = APIRouter()
    router.include_router(health_router.router)
    router.include_router(drugs_router.router)
    router.include_router(simulate_router.router)
    router.include_router(ddi_router.router)
    router.include_router(population_router.router)
    router.include_router(validation_router.router)
    return router


# Module-level router for one-line mount on existing production apps.
v2_router: APIRouter = build_v2_router()


# ════════════════════════════════════════════════════════════════
# STANDALONE APP
# ════════════════════════════════════════════════════════════════


def create_v2_app(
    *,
    title: str = "NeoGen V2 API",
    description: Optional[str] = None,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Construct a fresh standalone V2 FastAPI app.

    Tests use this factory to build a clean app per session. The
    module-level :data:`v2_app` is an instance built with default
    arguments for use with ``uvicorn`` / ``python -m``.
    """
    from neogen.engine.orchestrator_v2 import ENGINE_VERSION

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):  # pragma: no cover - startup log
        logger.info(
            "NeoGen V2 API ready — engine %s — docs at /v2/docs",
            ENGINE_VERSION,
        )
        yield
        logger.info("NeoGen V2 API shutting down")

    app = FastAPI(
        title=title,
        description=description
        or (
            "NeoGen V1 mechanistic PBPK engine — REST API. "
            "Exposes single-drug PK, DDI pair AUCR, population "
            "Monte Carlo, and the validation harness."
        ),
        version=ENGINE_VERSION,
        docs_url="/v2/docs",
        redoc_url="/v2/redoc",
        openapi_url="/v2/openapi.json",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(v2_router, prefix="/v2")

    return app


# Default instance for ``uvicorn neogen.api.v2.app:v2_app``.
v2_app: FastAPI = create_v2_app()
