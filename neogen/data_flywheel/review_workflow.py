"""Append-only reviewer governance workflow (GAP-005).

Enforces how flywheel records move out of ``needs_review``:
- Only CITED records may advance beyond needs_review.
- Severity-bearing records may ONLY reach ``human_reviewed`` (never auto).
- A "machine:" reviewer id may NOT perform a human_reviewed promotion (no fake
  human review).
- Ambiguity/license gaps are terminal-blocked, not promotable.
- Every transition emits an append-only ``ReviewEvent`` with prev/new hashes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from neogen.data_flywheel.provenance import content_hash

STATUSES = {
    "needs_review", "machine_checked", "human_reviewed", "rejected",
    "superseded", "blocked_license", "blocked_ambiguity",
}

# (from, to) allowed transitions.
_ALLOWED = {
    ("needs_review", "machine_checked"),
    ("needs_review", "human_reviewed"),
    ("needs_review", "rejected"),
    ("needs_review", "blocked_license"),
    ("needs_review", "blocked_ambiguity"),
    ("machine_checked", "human_reviewed"),
    ("machine_checked", "rejected"),
    ("machine_checked", "superseded"),
    ("human_reviewed", "superseded"),
    ("human_reviewed", "rejected"),
}


class ReviewError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewEvent:
    record_id: str
    from_status: str
    to_status: str
    reviewer_id: str
    timestamp: str
    reason: str
    previous_hash: str
    new_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_cited(record: dict) -> bool:
    return bool(
        record.get("citation")
        or record.get("citation_span")
        or record.get("evidence_refs")
        or record.get("evidence_text")
    )


def promote(
    record: dict,
    *,
    record_id: str,
    from_status: str,
    to_status: str,
    reviewer_id: str,
    reason: str,
    timestamp: str | None = None,
) -> ReviewEvent:
    """Validate + emit a ReviewEvent. Raises ReviewError on any rule violation."""
    timestamp = timestamp or _now()
    if to_status not in STATUSES or from_status not in STATUSES:
        raise ReviewError(f"unknown status in transition {from_status}->{to_status}")
    if (from_status, to_status) not in _ALLOWED:
        raise ReviewError(f"transition not allowed: {from_status} -> {to_status}")

    advancing = to_status in ("machine_checked", "human_reviewed")
    if advancing and not is_cited(record):
        raise ReviewError("cannot advance an uncited record beyond needs_review")

    has_severity = record.get("severity") is not None
    if has_severity and to_status == "machine_checked":
        raise ReviewError("severity-bearing record requires human_reviewed, not machine_checked")

    if to_status == "human_reviewed" and reviewer_id.startswith("machine:"):
        raise ReviewError("human_reviewed requires a human reviewer id (no 'machine:' prefix)")

    prev_hash = str(record.get("content_hash") or record.get("provenance_hash") or "")
    new_hash = content_hash(f"{prev_hash}|{to_status}|{reviewer_id}|{timestamp}")
    return ReviewEvent(
        record_id=record_id, from_status=from_status, to_status=to_status,
        reviewer_id=reviewer_id, timestamp=timestamp, reason=reason,
        previous_hash=prev_hash, new_hash=new_hash,
    )


def append_event(event: ReviewEvent, path: Path | str) -> None:
    """Append a review event as one JSONL line (append-only; never rewrites)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
