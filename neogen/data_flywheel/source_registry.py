"""Source registry loader + validator for the NeoGen data flywheel.

Loads the registry index (``data/flywheel/source_registry.yaml``), merges the
per-group include files under ``data/flywheel/sources/``, validates every
source entry against the required-field + enum contract, and exposes a small
query API.

Design constraints:
- Fail loudly and clearly on malformed input (raise ``SourceRegistryError``).
- No network access, no mutation of source files.
- Pure read + validate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Repo root = parents[2] of this file (neogen/data_flywheel/source_registry.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = _REPO_ROOT / "data" / "flywheel" / "source_registry.yaml"

REQUIRED_FIELDS: tuple[str, ...] = (
    "source_id",
    "name",
    "category",
    "source_url",
    "access_method",
    "license_status",
    "robots_tos_status",
    "update_frequency",
    "expected_data_types",
    "key_identifiers",
    "joins_to_identity_spine",
    "value_for_moat",
    "ingestion_priority",
    "crawl_allowed",
    "requires_api_key",
    "estimated_size",
    "notes",
    "first_ingestion_status",
)

VALID_ACCESS_METHODS = frozenset(
    {"api", "bulk_download", "scrape", "browser", "manual", "licensed"}
)
VALID_LICENSE_STATUS = frozenset(
    {"open", "restricted", "commercial", "unknown", "requires_review"}
)
VALID_PRIORITY = frozenset({"P0", "P1", "P2", "P3"})
VALID_INGESTION_STATUS = frozenset(
    {"not_started", "tested", "partial", "complete", "blocked"}
)


class SourceRegistryError(ValueError):
    """Raised when the registry is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class Registry:
    """An immutable, validated view of the source registry."""

    registry_version: str
    created_at: str
    task_id: str
    categories: tuple[str, ...]
    sources: tuple[dict[str, Any], ...]

    def get(self, source_id: str) -> dict[str, Any] | None:
        for s in self.sources:
            if s["source_id"] == source_id:
                return s
        return None

    def by_priority(self, priority: str) -> list[dict[str, Any]]:
        return [s for s in self.sources if s["ingestion_priority"] == priority]

    def by_category(self, category: str) -> list[dict[str, Any]]:
        return [s for s in self.sources if s["category"] == category]

    def crawl_allowed_sources(self) -> list[dict[str, Any]]:
        return [s for s in self.sources if s.get("crawl_allowed") is True]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {"total": len(self.sources)}
        for s in self.sources:
            out[s["ingestion_priority"]] = out.get(s["ingestion_priority"], 0) + 1
            key = f"status:{s['first_ingestion_status']}"
            out[key] = out.get(key, 0) + 1
        return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SourceRegistryError(f"registry file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise SourceRegistryError(f"YAML parse error in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SourceRegistryError(f"top-level of {path} must be a mapping")
    return data


def _validate_source(src: Any, where: str) -> None:
    if not isinstance(src, dict):
        raise SourceRegistryError(f"{where}: source entry must be a mapping")
    missing = [f for f in REQUIRED_FIELDS if f not in src]
    if missing:
        raise SourceRegistryError(
            f"{where}: source {src.get('source_id', '<no id>')!r} missing "
            f"required fields: {missing}"
        )
    sid = src["source_id"]
    if src["access_method"] not in VALID_ACCESS_METHODS:
        raise SourceRegistryError(
            f"{where}: source {sid!r} has invalid access_method "
            f"{src['access_method']!r}"
        )
    if src["license_status"] not in VALID_LICENSE_STATUS:
        raise SourceRegistryError(
            f"{where}: source {sid!r} has invalid license_status "
            f"{src['license_status']!r}"
        )
    if src["ingestion_priority"] not in VALID_PRIORITY:
        raise SourceRegistryError(
            f"{where}: source {sid!r} has invalid ingestion_priority "
            f"{src['ingestion_priority']!r}"
        )
    if src["first_ingestion_status"] not in VALID_INGESTION_STATUS:
        raise SourceRegistryError(
            f"{where}: source {sid!r} has invalid first_ingestion_status "
            f"{src['first_ingestion_status']!r}"
        )
    if not isinstance(src["crawl_allowed"], bool):
        raise SourceRegistryError(f"{where}: source {sid!r} crawl_allowed must be bool")
    if not isinstance(src["requires_api_key"], bool):
        raise SourceRegistryError(
            f"{where}: source {sid!r} requires_api_key must be bool"
        )


def load_registry(index_path: Path | str | None = None) -> Registry:
    """Load + validate the full registry (index + includes)."""
    path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
    index = _load_yaml(path)

    for key in ("registry_version", "created_at", "task_id", "categories", "includes"):
        if key not in index:
            raise SourceRegistryError(f"{path}: index missing top-level key {key!r}")

    sources: list[dict[str, Any]] = []
    sources_dir = path.parent / "sources"
    for rel in index["includes"]:
        inc_path = (path.parent / rel).resolve()
        inc = _load_yaml(inc_path)
        group_sources = inc.get("sources")
        if not isinstance(group_sources, list):
            raise SourceRegistryError(f"{inc_path}: 'sources' must be a list")
        for src in group_sources:
            _validate_source(src, where=str(inc_path))
            sources.append(src)

    # Uniqueness of source_id.
    seen: set[str] = set()
    for s in sources:
        sid = s["source_id"]
        if sid in seen:
            raise SourceRegistryError(f"duplicate source_id: {sid!r}")
        seen.add(sid)

    # Every source category must be declared in the index.
    declared = set(index["categories"])
    for s in sources:
        if s["category"] not in declared:
            raise SourceRegistryError(
                f"source {s['source_id']!r} has undeclared category "
                f"{s['category']!r} (declared: {sorted(declared)})"
            )

    if not sources_dir.exists():
        raise SourceRegistryError(f"sources dir not found: {sources_dir}")

    return Registry(
        registry_version=str(index["registry_version"]),
        created_at=str(index["created_at"]),
        task_id=str(index["task_id"]),
        categories=tuple(index["categories"]),
        sources=tuple(sources),
    )
