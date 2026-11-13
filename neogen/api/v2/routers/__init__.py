"""V2 API router subpackage.

Each router module exports a module-level ``router: APIRouter``
which :mod:`neogen.api.v2.app` mounts under the appropriate
prefix.
"""

from neogen.api.v2.routers import (
    ddi,
    drugs,
    health,
    population,
    simulate,
    validation,
)

__all__ = [
    "ddi",
    "drugs",
    "health",
    "population",
    "simulate",
    "validation",
]
