"""Pydantic event envelope helpers for feature pipeline workflows."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neogen.events.molecule_events import EventRef, ReplayImplications, content_hash

FeatureEventType = Literal[
    "FeatureRequested",
    "FeatureVectorComputed",
    "FeatureVectorStored",
    "FeatureSchemaMismatch",
]


class FeatureEventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    run_id: str
    parent_event_id: str | None = None
    sequence_number: int
    timestamp: str
    tenant_id: str
    event_type: FeatureEventType
    subsystem: Literal["features"] = "features"
    actor: str = "service:feature-worker"
    input_refs: list[EventRef] = Field(default_factory=list)
    output_refs: list[EventRef] = Field(default_factory=list)
    tool_name: str = "rdkit"
    tool_version: str
    code_hash: str
    config_hash: str
    dataset_snapshot_id: str | None = None
    model_id: str | None = None
    seed: int = 0
    status: Literal["succeeded", "failed", "retrying"] = "succeeded"
    warnings: list[str] = Field(default_factory=list)
    output_hash: str
    duration_ms: int = 0
    replay_implications: ReplayImplications = Field(default_factory=ReplayImplications)


def feature_event(
    *,
    event_type: FeatureEventType,
    sequence_number: int,
    output_payload: Any,
    input_refs: list[EventRef] | None = None,
    output_refs: list[EventRef] | None = None,
    parent_event_id: str | None = None,
    model_id: str | None = None,
    run_id: str = "run_week2_feature_pipeline",
    tenant_id: str = "tenant_default",
    rdkit_version: str = "unknown",
    warnings: list[str] | None = None,
) -> FeatureEventEnvelope:
    output_hash = content_hash(output_payload)
    event_seed = {
        "event_type": event_type,
        "sequence_number": sequence_number,
        "parent_event_id": parent_event_id,
        "input_refs": [ref.model_dump() for ref in input_refs or []],
        "output_refs": [ref.model_dump() for ref in output_refs or []],
        "output_hash": output_hash,
        "model_id": model_id,
    }
    event_id = "evt_" + hashlib.sha256(
        json.dumps(event_seed, sort_keys=True).encode("utf-8")
    ).hexdigest()[:26]
    return FeatureEventEnvelope(
        event_id=event_id,
        run_id=run_id,
        parent_event_id=parent_event_id,
        sequence_number=sequence_number,
        timestamp="2026-05-13T00:00:00.000Z",
        tenant_id=tenant_id,
        event_type=event_type,
        input_refs=input_refs or [],
        output_refs=output_refs or [],
        tool_version=rdkit_version,
        code_hash=content_hash("neogen.ml._features.week2"),
        config_hash=content_hash({"feature_pipeline": "week2", "seed": 0}),
        model_id=model_id,
        warnings=warnings or [],
        output_hash=output_hash,
    )
