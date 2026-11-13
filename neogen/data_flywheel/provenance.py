"""Provenance model for every ingested datum in the data flywheel.

Nothing enters the moat without provenance. A ``Provenance`` records *where*
a datum came from, *when*, *how*, under *what license*, with a *content hash*
for replay, and a human-readable citation. ``reviewer_status`` tracks whether
a human has cleared it for use in any claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

ExtractionMethod = Literal[
    "api", "bulk_download", "scrape", "browser", "manual", "derived"
]
ReviewerStatus = Literal[
    "unreviewed", "auto_accepted", "human_reviewed", "rejected"
]
LicenseStatus = Literal[
    "open", "restricted", "commercial", "unknown", "requires_review"
]


def content_hash(payload: str) -> str:
    """Deterministic SHA-256 over a UTF-8 payload. 64-char lowercase hex."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Provenance:
    """Immutable provenance record."""

    source_id: str
    source_url: str
    retrieved_at: str  # ISO-8601 UTC
    license_status: LicenseStatus
    extraction_method: ExtractionMethod
    content_hash: str
    citation_text: str | None = None
    citation_span: str | None = None  # e.g. "label set_id=X section=DDI"
    reviewer_status: ReviewerStatus = "unreviewed"
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("Provenance.source_id is required")
        if not self.source_url:
            raise ValueError("Provenance.source_url is required")
        if not self.retrieved_at:
            raise ValueError("Provenance.retrieved_at is required")
        if len(self.content_hash) != 64 or any(
            c not in "0123456789abcdef" for c in self.content_hash
        ):
            raise ValueError("Provenance.content_hash must be 64-char hex sha256")

    @property
    def is_citable(self) -> bool:
        """True only if there is at least one citation field present."""
        return bool(self.citation_text) or bool(self.citation_span)

    @property
    def is_usable_for_claims(self) -> bool:
        """A datum may back a claim only if human-reviewed AND citable."""
        return self.reviewer_status == "human_reviewed" and self.is_citable

    @classmethod
    def for_payload(
        cls,
        *,
        source_id: str,
        source_url: str,
        retrieved_at: str,
        license_status: LicenseStatus,
        extraction_method: ExtractionMethod,
        payload: str,
        citation_text: str | None = None,
        citation_span: str | None = None,
        reviewer_status: ReviewerStatus = "unreviewed",
    ) -> "Provenance":
        """Build a Provenance by hashing the payload for the caller."""
        return cls(
            source_id=source_id,
            source_url=source_url,
            retrieved_at=retrieved_at,
            license_status=license_status,
            extraction_method=extraction_method,
            content_hash=content_hash(payload),
            citation_text=citation_text,
            citation_span=citation_span,
            reviewer_status=reviewer_status,
        )
