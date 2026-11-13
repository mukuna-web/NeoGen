"""Pydantic event helpers for conformer generation workflows."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rdkit import rdBase

from neogen.events.molecule_events import EventRef, ReplayImplications, content_hash

ConformerEventType = Literal[
    "ConformerGenerationRequested",
    "ConformerSetGenerated",
    "ConformerSetStored",
    "GeometryDescriptorsComputed",
    "ConformerGenerationFailed",
]


class ConformerEventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    run_id: str
    parent_event_id: str | None = None
    sequence_number: int
    timestamp: str
    tenant_id: str
    event_type: ConformerEventType
    subsystem: Literal["chemistry"] = "chemistry"
    actor: str = "service:chem-worker"
    input_refs: list[EventRef] = Field(default_factory=list)
    output_refs: list[EventRef] = Field(default_factory=list)
    tool_name: str = "rdkit"
    tool_version: str = rdBase.rdkitVersion
    code_hash: str
    config_hash: str
    dataset_snapshot_id: str | None = None
    model_id: str | None = None
    seed: int = 42
    status: Literal["succeeded", "failed", "retrying"] = "succeeded"
    warnings: list[str] = Field(default_factory=list)
    output_hash: str
    duration_ms: int = 0
    replay_implications: ReplayImplications = Field(default_factory=ReplayImplications)


def conformer_event(
    *,
    event_type: ConformerEventType,
    sequence_number: int,
    output_payload: Any,
    input_refs: list[EventRef] | None = None,
    output_refs: list[EventRef] | None = None,
    parent_event_id: str | None = None,
    run_id: str = "run_week3_conformer_cache",
    tenant_id: str = "tenant_default",
    config_hash_value: str = "",
    seed: int = 42,
    status: Literal["succeeded", "failed", "retrying"] = "succeeded",
    warnings: list[str] | tuple[str, ...] | None = None,
    duration_ms: int = 0,
) -> ConformerEventEnvelope:
    output_hash = content_hash(output_payload)
    event_seed = {
        "event_type": event_type,
        "sequence_number": sequence_number,
        "parent_event_id": parent_event_id,
        "input_refs": [ref.model_dump() for ref in input_refs or []],
        "output_refs": [ref.model_dump() for ref in output_refs or []],
        "output_hash": output_hash,
        "config_hash": config_hash_value,
        "seed": seed,
        "status": status,
    }
    event_id = "evt_" + hashlib.sha256(
        json.dumps(event_seed, sort_keys=True).encode("utf-8")
    ).hexdigest()[:26]
    return ConformerEventEnvelope(
        event_id=event_id,
        run_id=run_id,
        parent_event_id=parent_event_id,
        sequence_number=sequence_number,
        timestamp="2026-05-13T00:00:00.000Z",
        tenant_id=tenant_id,
        event_type=event_type,
        input_refs=input_refs or [],
        output_refs=output_refs or [],
        code_hash=content_hash("neogen.chem.conformers.week3"),
        config_hash="sha256:" + config_hash_value if config_hash_value else content_hash({}),
        seed=seed,
        status=status,
        warnings=list(warnings or []),
        output_hash=output_hash,
        duration_ms=duration_ms,
    )
