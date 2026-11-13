"""FastAPI app factory for the V1 simulation API.

The factory pattern (``build_app(**overrides)``) keeps the default
instance small for ``uvicorn neogen.api.v1:app`` and still lets tests
and alternate deployments tweak CORS origins, API title, or prefix.

Production deployments should always pass an explicit ``cors_origins``
list — the default ``("*",)`` is permissive for local development and
must not be used when the service is reachable from the public
internet. Callers that skip the override get a log warning so mistakes
are visible in the server log.
"""
from __future__ import annotations

import logging
from typing import Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from neogen.api.routes_conformers import router as conformers_router
from neogen.api.routes_features import router as features_router
from neogen.api.routes_molecules import router as molecules_router
from neogen.api.v1.endpoints import router
from neogen.api.v1.schemas import ErrorResponse
from neogen.engine.orchestrator_v2 import ENGINE_VERSION

logger = logging.getLogger(__name__)


API_PREFIX = "/v1"
"""Path prefix mounted on top of the simulation routes.

Keeping the prefix explicit makes it painless to re-expose the routes
under a different root later (e.g. ``/api/v1``) without touching the
handlers themselves.
"""


_DEFAULT_CORS_ORIGINS: tuple[str, ...] = ("*",)


def build_app(
    *,
    cors_origins: Sequence[str] = _DEFAULT_CORS_ORIGINS,
    title: str = "NeoGen V1 Simulation API",
    prefix: str = API_PREFIX,
) -> FastAPI:
    """Construct a configured FastAPI application.

    Args:
        cors_origins: Allow-list for browser CORS. Pass an explicit
            list in production; the ``("*",)`` default is only safe for
            local development.
        title: OpenAPI title surfaced at ``/docs``.
        prefix: URL prefix mounted on the simulation router.

    Returns:
        A FastAPI instance ready to be served by any ASGI server.
    """
    if tuple(cors_origins) == _DEFAULT_CORS_ORIGINS:
        logger.warning(
            "V1 API using permissive CORS (*) — override cors_origins "
            "before deploying publicly."
        )

    app = FastAPI(
        title=title,
        version=ENGINE_VERSION,
        description=(
            "Mechanistic PBPK simulation API for NeoGen V1. Wraps the "
            "14-compartment engine, P2 absorption, P3 DDI coupling, "
            "and P4 PD layer in a small stateless HTTP surface."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=prefix)
    app.include_router(molecules_router)
    app.include_router(features_router)
    app.include_router(conformers_router)

    # ── Uniform error envelope ────────────────────────────────
    # Without these handlers FastAPI returns its default `{"detail": ...}`
    # shape, which is fine but hides the error type. Wrapping everything
    # in :class:`ErrorResponse` makes the API self-describing and keeps
    # the OpenAPI ``responses`` declarations on the routes honest.

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        envelope = ErrorResponse(
            error=f"HTTP{exc.status_code}",
            detail=str(exc.detail),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope.model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        envelope = ErrorResponse(
            error="ValidationError",
            detail=str(exc.errors()),
        )
        return JSONResponse(
            status_code=422,
            content=envelope.model_dump(),
        )

    return app
