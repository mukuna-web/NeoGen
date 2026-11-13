"""Versioned feature schema registry and model compatibility checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FeatureMode = Literal["legacy_2d", "stereo_2d", "conformer_3d"]


@dataclass(frozen=True)
class FeatureSchema:
    schema_version: str
    mode: FeatureMode
    dimensions: int
    requires_stereo: bool
    requires_conformers: bool
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class CompatibilityCheck:
    compatible: bool
    reason: str | None = None


def _names(prefix: str, dimensions: int) -> tuple[str, ...]:
    return tuple(sorted(f"{prefix}_{idx:04d}" for idx in range(dimensions)))


FEATURE_SCHEMAS: dict[str, FeatureSchema] = {
    "legacy_2d_v1": FeatureSchema(
        schema_version="legacy_2d_v1",
        mode="legacy_2d",
        dimensions=1044,
        requires_stereo=False,
        requires_conformers=False,
        feature_names=_names("legacy_2d", 1044),
    ),
    "stereo_2d_v1": FeatureSchema(
        schema_version="stereo_2d_v1",
        mode="stereo_2d",
        dimensions=1054,
        requires_stereo=True,
        requires_conformers=False,
        feature_names=_names("stereo_2d", 1054),
    ),
    "conformer_3d_v1": FeatureSchema(
        schema_version="conformer_3d_v1",
        mode="conformer_3d",
        dimensions=1086,
        requires_stereo=True,
        requires_conformers=True,
        feature_names=_names("conformer_3d", 1086),
    ),
}

_MODEL_COMPATIBILITY: dict[str, set[str]] = {}


def get_schema(version: str) -> FeatureSchema:
    try:
        return FEATURE_SCHEMAS[version]
    except KeyError as exc:
        raise KeyError(f"unknown feature schema_version: {version}") from exc


def register_model_for_schema(model_id: str, schema_version: str) -> None:
    get_schema(schema_version)
    _MODEL_COMPATIBILITY.setdefault(model_id, set()).add(schema_version)


def validate_compatible(model_id: str, feature_vector: object) -> CompatibilityCheck:
    schema_version = getattr(feature_vector, "schema_version", None)
    compatible = _MODEL_COMPATIBILITY.get(model_id, set())
    if schema_version in compatible:
        return CompatibilityCheck(compatible=True)
    if not compatible:
        return CompatibilityCheck(
            compatible=False,
            reason=f"model_id {model_id!r} has no registered compatible schemas",
        )
    return CompatibilityCheck(
        compatible=False,
        reason=(
            f"model_id {model_id!r} expects {sorted(compatible)}; "
            f"feature vector uses {schema_version!r}"
        ),
    )


def reset_model_registry() -> None:
    _MODEL_COMPATIBILITY.clear()
