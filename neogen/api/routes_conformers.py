"""Week 3 conformer generation and descriptor API routes."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from neogen.chem.conformers import ConformerConfig, conformer_set_to_dict, generate_conformer_set
from neogen.db.repositories.conformer_repository import ConformerRepository
from neogen.db.repositories.molecule_repository import MoleculeRepository
from neogen.db.session import get_connection
from neogen.events.conformer_events import conformer_event
from neogen.events.molecule_events import EventRef
from neogen.events.store import append_event
from neogen.ml.feature_schemas.conformer_3d_v1 import RAW_3D_FEATURE_NAMES, compute_3d_tail

router = APIRouter()


class GenerateConformerRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stereoisomer_id: str
    config: dict[str, Any] | None = None


class DescriptorRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conformer_set_id: str
    descriptor_set: str = Field(default="default")


def get_repositories():
    conn = get_connection()
    try:
        yield MoleculeRepository(conn), ConformerRepository(conn)
    finally:
        conn.close()


@router.post("/v1/conformers/generate")
def generate_conformers_endpoint(request: GenerateConformerRequest, repos=Depends(get_repositories)) -> dict:
    molecules, conformers = repos
    isomer = molecules.get_stereoisomer_by_id(request.stereoisomer_id)
    if isomer is None:
        raise HTTPException(status_code=404, detail=f"unknown stereoisomer_id: {request.stereoisomer_id}")
    try:
        conformer_set = generate_conformer_set(isomer, ConformerConfig.from_mapping(request.config))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    conformers.upsert_set(conformer_set)
    return conformer_set_to_dict(conformer_set)


@router.get("/v1/conformer-sets/{conformer_set_id}")
def get_conformer_set(conformer_set_id: str, repos=Depends(get_repositories)) -> dict:
    _molecules, conformers = repos
    conformer_set = conformers.get_set_by_id(conformer_set_id)
    if conformer_set is None:
        raise HTTPException(status_code=404, detail=f"unknown conformer_set_id: {conformer_set_id}")
    return conformer_set_to_dict(conformer_set)


@router.get("/v1/conformers/{conformer_id}")
def get_conformer(conformer_id: str, repos=Depends(get_repositories)) -> dict:
    _molecules, conformers = repos
    conformer = conformers.get_conformer_by_id(conformer_id)
    if conformer is None:
        raise HTTPException(status_code=404, detail=f"unknown conformer_id: {conformer_id}")
    return asdict(conformer)


@router.post("/v1/conformers/descriptors")
def compute_conformer_descriptors(request: DescriptorRequest, repos=Depends(get_repositories)) -> dict:
    molecules, conformers = repos
    conformer_set = conformers.get_set_by_id(request.conformer_set_id)
    if conformer_set is None:
        raise HTTPException(status_code=404, detail=f"unknown conformer_set_id: {request.conformer_set_id}")
    isomer = molecules.get_stereoisomer_by_id(conformer_set.stereoisomer_id)
    if isomer is None:
        raise HTTPException(status_code=404, detail=f"unknown stereoisomer_id: {conformer_set.stereoisomer_id}")
    tail = compute_3d_tail(conformer_set, isomer)
    event = append_event(
        conformer_event(
            event_type="GeometryDescriptorsComputed",
            sequence_number=4,
            input_refs=[EventRef(type="conformer_set", id=conformer_set.conformer_set_id)],
            output_payload={
                "conformer_set_id": conformer_set.conformer_set_id,
                "descriptor_set": request.descriptor_set,
                "values": tail.astype(float).tolist(),
            },
            config_hash_value=conformer_set.config_hash,
            seed=conformer_set.seed,
        )
    )
    return {
        "conformer_set_id": conformer_set.conformer_set_id,
        "descriptor_names": list(RAW_3D_FEATURE_NAMES),
        "values": tail.astype(float).tolist(),
        "events": [event.model_dump()],
    }
