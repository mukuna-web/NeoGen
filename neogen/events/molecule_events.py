"""Pydantic event envelope models for molecule identity workflows."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventStatus = Literal["succeeded", "failed", "retrying"]
MoleculeEventType = Literal[
    "MoleculeSubmitted",
    "MoleculeNormalized",
    "StereoDetected",
    "StereoEnumerated",
    "StereoisomerRegistered",
]


class EventRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    id: str


class ReplayImplications(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    deterministic: bool = True
    depends_on: list[str] = Field(default_factory=list)
    version_pinned: bool = True


class MoleculeEventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    run_id: str
    parent_event_id: str | None = None
    sequence_number: int
    timestamp: str
    tenant_id: str
    event_type: MoleculeEventType
    subsystem: Literal["chemistry"] = "chemistry"
    actor: str = "service:chem-worker"
    input_refs: list[EventRef] = Field(default_factory=list)
    output_refs: list[EventRef] = Field(default_factory=list)
    tool_name: str = "rdkit"
    tool_version: str = "2024.03.5"
    code_hash: str
    config_hash: str
    dataset_snapshot_id: str | None = None
    model_id: str | None = None
    seed: int = 0
    status: EventStatus = "succeeded"
    warnings: list[str] = Field(default_factory=list)
    output_hash: str
    duration_ms: int = 0
    replay_implications: ReplayImplications = Field(
        default_factory=ReplayImplications
    )


def content_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def molecule_event(
    *,
    event_type: MoleculeEventType,
    sequence_number: int,
    output_payload: Any,
    input_refs: list[EventRef] | None = None,
    output_refs: list[EventRef] | None = None,
    parent_event_id: str | None = None,
    run_id: str = "run_week1_identity",
    tenant_id: str = "tenant_default",
    warnings: list[str] | None = None,
) -> MoleculeEventEnvelope:
    output_hash = content_hash(output_payload)
    event_seed = {
        "event_type": event_type,
        "sequence_number": sequence_number,
        "parent_event_id": parent_event_id,
        "input_refs": [ref.model_dump() for ref in input_refs or []],
        "output_refs": [ref.model_dump() for ref in output_refs or []],
        "output_hash": output_hash,
    }
    event_id = "evt_" + hashlib.sha256(
        json.dumps(event_seed, sort_keys=True).encode("utf-8")
    ).hexdigest()[:26]
    return MoleculeEventEnvelope(
        event_id=event_id,
        run_id=run_id,
        parent_event_id=parent_event_id,
        sequence_number=sequence_number,
        timestamp="2026-05-12T00:00:00.000Z",
        tenant_id=tenant_id,
        event_type=event_type,
        input_refs=input_refs or [],
        output_refs=output_refs or [],
        code_hash=content_hash("neogen.chem.identity.week1"),
        config_hash=content_hash({"standardization_version": "stdz_v1_rdkit_2024.03.5"}),
        warnings=warnings or [],
        output_hash=output_hash,
    )
