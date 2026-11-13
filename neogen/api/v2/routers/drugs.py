"""Drug catalog endpoints for the V2 API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from neogen.api.v2.schemas import DrugListResponse

router = APIRouter(prefix="/drugs", tags=["drugs"])


@router.get(
    "",
    response_model=DrugListResponse,
    summary="List drugs with literature overrides",
)
async def list_drugs() -> DrugListResponse:
    """Return the set of drug names the V1 engine can build directly.

    Every drug in this list is usable as ``drug_name`` in
    :func:`POST /v2/simulate`, :func:`POST /v2/ddi`, and
    :func:`POST /v2/population`. Drugs not in this list require
    the ML property layer (V1.1 scope).
    """
    from neogen.core.constants.literature_params import list_available_drugs

    drugs = list_available_drugs()
    return DrugListResponse(count=len(drugs), drugs=drugs)


@router.get(
    "/{drug_name}",
    summary="Inspect a single drug's literature parameters",
)
async def get_drug(drug_name: str) -> dict:
    """Return the raw literature parameter dict for a single drug.

    Returns 404 if the drug is not in the literature database.
    The response is intentionally a plain dict (not a typed
    response_model) because the literature entry contains mixed
    value types including frozen dataclasses for metabolism and
    transporter profiles.
    """
    from neogen.core.constants.literature_params import get_literature_params

    params = get_literature_params(drug_name)
    if params is None:
        raise HTTPException(
            status_code=404,
            detail=f"Drug '{drug_name}' not in literature database",
        )
    # Convert dataclasses + enums → primitives for JSON serialization.
    return _to_primitive(params)


def _to_primitive(obj):
    """Recursively coerce dataclasses and enums to JSON-friendly primitives."""
    import dataclasses
    from enum import Enum

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_primitive(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_primitive(x) for x in obj]
    return obj
