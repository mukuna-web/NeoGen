"""Week 2 feature pipeline API routes."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from rdkit import __version__ as rdkit_version

from neogen.db.repositories.conformer_repository import ConformerRepository
from neogen.db.repositories.feature_repository import FeatureRepository
from neogen.db.repositories.molecule_repository import MoleculeRepository
from neogen.db.session import get_connection
from neogen.events.feature_events import feature_event
from neogen.events.molecule_events import EventRef
from neogen.events.store import append_event
from neogen.ml._features import FeatureVector, compute_feature_vector
from neogen.ml.feature_registry import FEATURE_SCHEMAS, validate_compatible

router = APIRouter()


class ComputeFeatureRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str
    entity_type: str
    schema_version: str


class ValidateCompatibleRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_vector_id: str
    model_id: str


class RegisterModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    schema_version: str


def get_repositories():
    conn = get_connection()
    try:
        yield MoleculeRepository(conn), FeatureRepository(conn), ConformerRepository(conn)
    finally:
        conn.close()


def _response(feature_vector: FeatureVector, events: list[dict] | None = None) -> dict:
    return {
        "feature_vector_id": feature_vector.feature_vector_id,
        "input_entity_id": feature_vector.input_entity_id,
        "input_entity_type": feature_vector.input_entity_type,
        "schema_version": feature_vector.schema_version,
        "mode": feature_vector.mode,
        "vector_hash": feature_vector.vector_hash,
        "feature_names_hash": feature_vector.feature_names_hash,
        "dimensions": int(feature_vector.values.shape[0]),
        "values": feature_vector.values.astype(float).tolist(),
        "events": events or [],
    }


@router.get("/v1/features/schemas")
def list_feature_schemas() -> list[dict]:
    return [
        {
            "schema_version": schema.schema_version,
            "mode": schema.mode,
            "dimensions": schema.dimensions,
            "required_inputs": {
                "requires_stereo": schema.requires_stereo,
                "requires_conformers": schema.requires_conformers,
            },
            "feature_names": list(schema.feature_names),
        }
        for schema in FEATURE_SCHEMAS.values()
    ]


@router.post("/v1/features/register-model")
def register_model(request: RegisterModelRequest, repos=Depends(get_repositories)) -> dict:
    _molecules, features, _conformers = repos
    try:
        features.register_model_for_schema(request.model_id, request.schema_version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"model_id": request.model_id, "schema_version": request.schema_version}


@router.post("/v1/features/compute")
def compute_features_endpoint(request: ComputeFeatureRequest, repos=Depends(get_repositories)) -> dict:
    molecules, features, conformers = repos
    cached = features.get_by_input_schema(request.entity_id, request.schema_version)
    if cached is not None:
        return _response(cached)
    requested = append_event(
        feature_event(
            event_type="FeatureRequested",
            sequence_number=1,
            input_refs=[EventRef(type=request.entity_type, id=request.entity_id)],
            output_payload=request.model_dump(),
            rdkit_version=rdkit_version,
        )
    )
    try:
        vector = compute_feature_vector(
            entity_id=request.entity_id,
            entity_type=request.entity_type,
            schema_version=request.schema_version,
            repository=molecules,
            conformer_repository=conformers,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    computed = append_event(
        feature_event(
            event_type="FeatureVectorComputed",
            sequence_number=2,
            parent_event_id=requested.event_id,
            input_refs=[EventRef(type=request.entity_type, id=request.entity_id)],
            output_refs=[EventRef(type="feature_vector", id=vector.feature_vector_id)],
            output_payload={
                "entity_id": request.entity_id,
                "schema_version": request.schema_version,
                "vector_hash": vector.vector_hash,
                "feature_names_hash": vector.feature_names_hash,
                "code_hash": "neogen.ml._features.week2",
                "rdkit_version": rdkit_version,
            },
            rdkit_version=rdkit_version,
        )
    )
    stored_vector = features.upsert(vector, computed.event_id)
    stored = append_event(
        feature_event(
            event_type="FeatureVectorStored",
            sequence_number=3,
            parent_event_id=computed.event_id,
            input_refs=[EventRef(type="feature_vector", id=vector.feature_vector_id)],
            output_refs=[EventRef(type="feature_vector", id=vector.feature_vector_id)],
            output_payload=asdict(stored_vector) | {"values": stored_vector.values.astype(float).tolist()},
            rdkit_version=rdkit_version,
        )
    )
    return _response(stored_vector, [requested.model_dump(), computed.model_dump(), stored.model_dump()])


@router.post("/v1/features/validate-compatible")
def validate_compatible_endpoint(request: ValidateCompatibleRequest, repos=Depends(get_repositories)) -> dict:
    _molecules, features, _conformers = repos
    features.load_model_registry()
    vector = features.get_by_id(request.feature_vector_id)
    if vector is None:
        raise HTTPException(status_code=404, detail=f"unknown feature_vector_id: {request.feature_vector_id}")
    check = validate_compatible(request.model_id, vector)
    if not check.compatible:
        event = append_event(
            feature_event(
                event_type="FeatureSchemaMismatch",
                sequence_number=4,
                input_refs=[EventRef(type="feature_vector", id=vector.feature_vector_id)],
                output_payload={
                    "model_id": request.model_id,
                    "feature_vector_id": vector.feature_vector_id,
                    "reason": check.reason,
                },
                model_id=request.model_id,
                rdkit_version=rdkit_version,
            )
        )
        return {"compatible": False, "reason": check.reason, "events": [event.model_dump()]}
    return {"compatible": True}


@router.get("/v1/features/{feature_vector_id}")
def get_feature_vector(feature_vector_id: str, repos=Depends(get_repositories)) -> dict:
    _molecules, features, _conformers = repos
    vector = features.get_by_id(feature_vector_id)
    if vector is None:
        raise HTTPException(status_code=404, detail=f"unknown feature_vector_id: {feature_vector_id}")
    return _response(vector)
