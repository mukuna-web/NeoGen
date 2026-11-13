"""Week 1 molecule identity and stereochemistry API routes."""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from neogen.chem.identity import InputType, MolecularIdentity, MoleculeParent, StereoMixture, Stereoisomer
from neogen.chem.standardize import HeavyAtomLimitError, MoleculeNormalizationError, molecular_identity_from_input
from neogen.chem.stereo import enumerate_stereoisomers, is_default_racemic_warfarin, racemic_warfarin_mixture, resolve_stereoisomer
from neogen.db.repositories.molecule_repository import MoleculeRepository
from neogen.db.session import get_connection
from neogen.events.molecule_events import EventRef, molecule_event
from neogen.events.store import append_event, clear_events
from neogen.ml.feature_registry import reset_model_registry

router = APIRouter()


class NormalizeRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: str = Field(..., min_length=1)
    input_type: InputType


class NormalizeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    molecular_identity: dict
    molecule_parent: dict
    events: list[dict]


class ResolveStereoRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_id: str


class ResolveStereoResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stereoisomer: dict | None = None
    stereo_mixture: dict | None = None
    component_stereoisomers: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)


class EnumerateStereoRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_id: str
    max_isomers: int = Field(default=64, ge=1, le=64)


def reset_molecule_route_state() -> None:
    conn = get_connection()
    try:
        MoleculeRepository(conn).clear()
        clear_events()
        reset_model_registry()
    finally:
        conn.close()


def get_molecule_repository():
    conn = get_connection()
    try:
        yield MoleculeRepository(conn)
    finally:
        conn.close()


def _store_normalized(
    repository: MoleculeRepository,
    parent: MoleculeParent,
    identity: MolecularIdentity,
) -> None:
    repository.upsert_parent(parent)
    repository.upsert_identity(identity)


@router.post("/v1/molecules/normalize")
def normalize_molecule(
    request: NormalizeRequest,
    repository: MoleculeRepository = Depends(get_molecule_repository),
) -> NormalizeResponse:
    try:
        parent, identity, _mol = molecular_identity_from_input(request.input, request.input_type)
    except HeavyAtomLimitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MoleculeNormalizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _store_normalized(repository, parent, identity)
    submitted = append_event(
        molecule_event(
            event_type="MoleculeSubmitted",
            sequence_number=1,
            output_payload={"input": request.input, "input_type": request.input_type},
            output_refs=[EventRef(type="molecular_input", id=identity.identity_id)],
        )
    )
    normalized = append_event(
        molecule_event(
            event_type="MoleculeNormalized",
            sequence_number=2,
            parent_event_id=submitted.event_id,
            input_refs=[EventRef(type="molecular_input", id=identity.identity_id)],
            output_refs=[
                EventRef(type="molecular_identity", id=identity.identity_id),
                EventRef(type="molecule_parent", id=parent.parent_id),
            ],
            output_payload={
                "molecular_identity": asdict(identity),
                "molecule_parent": asdict(parent),
            },
        )
    )
    return NormalizeResponse(
        molecular_identity=asdict(identity),
        molecule_parent=asdict(parent),
        events=[submitted.model_dump(), normalized.model_dump()],
    )


@router.post("/v1/molecules/resolve-stereo")
def resolve_stereo_endpoint(
    request: ResolveStereoRequest,
    repository: MoleculeRepository = Depends(get_molecule_repository),
) -> ResolveStereoResponse:
    identity = repository.get_identity_by_id(request.identity_id)
    if identity is None:
        raise HTTPException(status_code=404, detail=f"unknown identity_id: {request.identity_id}")

    if is_default_racemic_warfarin(identity.input_string, identity.input_type):
        mixture, components = racemic_warfarin_mixture(identity)
        repository.upsert_mixture(mixture)
        for component in components:
            repository.upsert_stereoisomer(component)
        event = append_event(
            molecule_event(
                event_type="StereoDetected",
                sequence_number=3,
                input_refs=[EventRef(type="molecular_identity", id=identity.identity_id)],
                output_refs=[EventRef(type="stereo_mixture", id=mixture.mixture_id)],
                output_payload={"stereo_mixture": asdict(mixture)},
                warnings=["known-drug warfarin defaults to racemic mixture"],
            )
        )
        return ResolveStereoResponse(
            stereo_mixture=asdict(mixture),
            component_stereoisomers=[asdict(component) for component in components],
            warnings=["known-drug warfarin defaults to racemic mixture"],
            events=[event.model_dump()],
        )

    stereoisomer = resolve_stereoisomer(identity)
    repository.upsert_stereoisomer(stereoisomer)
    detected = append_event(
        molecule_event(
            event_type="StereoDetected",
            sequence_number=3,
            input_refs=[EventRef(type="molecular_identity", id=identity.identity_id)],
            output_refs=[EventRef(type="stereoisomer", id=stereoisomer.stereoisomer_id)],
            output_payload={"stereo": asdict(stereoisomer)},
        )
    )
    registered = append_event(
        molecule_event(
            event_type="StereoisomerRegistered",
            sequence_number=4,
            parent_event_id=detected.event_id,
            input_refs=[EventRef(type="molecular_identity", id=identity.identity_id)],
            output_refs=[EventRef(type="stereoisomer", id=stereoisomer.stereoisomer_id)],
            output_payload=asdict(stereoisomer),
        )
    )
    return ResolveStereoResponse(
        stereoisomer=asdict(stereoisomer),
        events=[detected.model_dump(), registered.model_dump()],
    )


@router.post("/v1/molecules/enumerate-stereo")
def enumerate_stereo_endpoint(
    request: EnumerateStereoRequest,
    repository: MoleculeRepository = Depends(get_molecule_repository),
) -> dict[str, list[dict]]:
    identity = repository.get_identity_by_id(request.identity_id)
    if identity is None:
        raise HTTPException(status_code=404, detail=f"unknown identity_id: {request.identity_id}")
    stereoisomers = enumerate_stereoisomers(identity, max_isomers=request.max_isomers)
    for stereoisomer in stereoisomers:
        repository.upsert_stereoisomer(stereoisomer)
    event = append_event(
        molecule_event(
            event_type="StereoEnumerated",
            sequence_number=3,
            input_refs=[EventRef(type="molecular_identity", id=identity.identity_id)],
            output_refs=[
                EventRef(type="stereoisomer", id=stereoisomer.stereoisomer_id)
                for stereoisomer in stereoisomers
            ],
            output_payload=[asdict(stereoisomer) for stereoisomer in stereoisomers],
        )
    )
    return {
        "stereoisomers": [asdict(stereoisomer) for stereoisomer in stereoisomers],
        "events": [event.model_dump()],
    }


@router.get("/v1/molecules/{parent_id}")
def get_molecule_parent(
    parent_id: str,
    repository: MoleculeRepository = Depends(get_molecule_repository),
) -> dict:
    parent = repository.get_parent_by_id(parent_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"unknown parent_id: {parent_id}")
    return asdict(parent)


@router.get("/v1/stereoisomers/{stereoisomer_id}")
def get_stereoisomer(
    stereoisomer_id: str,
    repository: MoleculeRepository = Depends(get_molecule_repository),
) -> dict:
    stereoisomer = repository.get_stereoisomer_by_id(stereoisomer_id)
    if stereoisomer is None:
        raise HTTPException(status_code=404, detail=f"unknown stereoisomer_id: {stereoisomer_id}")
    return asdict(stereoisomer)


@router.get("/v1/molecules/{parent_id}/stereoisomers")
def list_stereoisomers_for_parent(
    parent_id: str,
    repository: MoleculeRepository = Depends(get_molecule_repository),
) -> list[dict]:
    return [asdict(stereoisomer) for stereoisomer in repository.list_stereoisomers_for_parent(parent_id)]
