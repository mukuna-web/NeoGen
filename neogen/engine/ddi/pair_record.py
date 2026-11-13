"""DDIPairRecord v0 — typed, frozen, replayable.

A `DDIPairRecord` is the smallest record that links a drug pair to a mechanism,
the feature-schema version the pair was built under, and the evidence pointers
that justify any downstream claim.

Design constraints (V35 §1):
- No clinical claims without citations. Every `severity` value MUST be paired
  with at least one `EvidenceRef`; otherwise `severity` is `None`.
- Deterministic content hash — same inputs always produce the same hash.
- Frozen dataclasses — records are immutable after construction.
- No external I/O — pure data + hashing only.

This is the foundation slice for the Week-2 DDIRecordV2 backlog item in
specs/3d_ddi/15_MASTER_ROADMAP_DR_STACK.md. A future cycle will:
- Promote this into a Postgres-backed table (Phase 1).
- Add CYP/transporter mechanism enrichment via the existing engine modules.
- Wire `feature_schema_version` against the Feature Pipeline contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

# ── Vocabulary (extend in future cycles, do not silently mutate) ────────────

MechanismCategory = Literal[
    "cyp_inhibition",
    "cyp_induction",
    "transporter_inhibition",
    "transporter_induction",
    "absorption",
    "protein_binding_displacement",
    "renal_clearance",
    "pharmacodynamic_synergy",
    "pharmacodynamic_antagonism",
    "unknown",
]

Role = Literal["perpetrator", "victim", "bidirectional", "unknown"]

EvidenceType = Literal[
    "fda_label",
    "ema_label",
    "rxnorm_crosswalk",
    "drugbank_interaction",
    "peer_reviewed",
    "textbook",
    "hypothetical_illustrative",
    "other",
]

StereoRelevance = Literal[
    "none",
    "enantiomer_sensitive",
    "diastereomer_sensitive",
    "ez_isomer_sensitive",
    "unknown",
]

# Pinned set of feature-schema versions accepted by this record. Bumping this
# set is itself a contract change and must follow specs/contracts/feature_pipeline.md.
ACCEPTED_FEATURE_SCHEMAS: frozenset[str] = frozenset(
    {"legacy_2d_v1", "stereo_2d_v1", "conformer_3d_v1"}
)


# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvidenceRef:
    """Single citation backing a claim in a DDIPairRecord."""

    evidence_type: EvidenceType
    citation: str  # e.g. "FDA-Label:NDA-021411-Escitalopram-2024-08-rev"
    url: Optional[str] = None
    extracted_at_utc: Optional[str] = None  # ISO 8601, UTC
    confidence: Optional[float] = None  # 0..1, source-derived if available


@dataclass(frozen=True)
class DDIPairRecord:
    """Typed record for one drug-drug interaction pair."""

    # Identity ──────────────────────────────────────────────────────────────
    pair_id: str  # canonical "drug_a_id|drug_b_id" with ids sorted
    drug_a_id: str  # rxnorm/drugbank/chembl id — see citation
    drug_b_id: str
    drug_a_name: str  # human-readable for UI; not used for hashing semantics
    drug_b_name: str

    # Mechanism ─────────────────────────────────────────────────────────────
    role: Role
    mechanism_category: MechanismCategory
    mechanism_note: str = ""  # short prose; never a clinical instruction
    enzyme_or_transporter: Optional[str] = None  # e.g. "CYP3A4", "OATP1B1"

    # Stereo + feature-schema linkage ───────────────────────────────────────
    stereo_relevance: StereoRelevance = "unknown"
    feature_schema_version: str = "legacy_2d_v1"

    # Evidence ──────────────────────────────────────────────────────────────
    evidence_refs: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    severity: Optional[str] = None  # None unless backed by EvidenceRef
    uncertainty_note: str = ""

    # Provenance ────────────────────────────────────────────────────────────
    schema_version: str = "ddi_pair_record_v0"
    created_at_utc: Optional[str] = None

    # ── Invariants enforced at construction ─────────────────────────────────
    def __post_init__(self) -> None:
        if self.feature_schema_version not in ACCEPTED_FEATURE_SCHEMAS:
            raise ValueError(
                f"feature_schema_version={self.feature_schema_version!r} "
                f"not in ACCEPTED_FEATURE_SCHEMAS={sorted(ACCEPTED_FEATURE_SCHEMAS)}"
            )
        if self.severity is not None and not self.evidence_refs:
            raise ValueError(
                "severity is set but no evidence_refs were provided. "
                "V35 §1: no clinical claims without citations."
            )
        expected_pair_id = _canonical_pair_id(self.drug_a_id, self.drug_b_id)
        if self.pair_id != expected_pair_id:
            raise ValueError(
                f"pair_id={self.pair_id!r} must equal canonical "
                f"{expected_pair_id!r} (sorted, pipe-joined)"
            )

    # ── Content hash for replay ────────────────────────────────────────────
    def content_hash(self) -> str:
        """Deterministic SHA-256 over canonical-JSON of all fields.

        Same inputs always produce the same hex digest. Used by replay tests
        and downstream cache keys.
        """
        payload = _canonical_json(asdict(self))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _canonical_pair_id(drug_a_id: str, drug_b_id: str) -> str:
    a, b = sorted([drug_a_id, drug_b_id])
    return f"{a}|{b}"


def make_pair_id(drug_a_id: str, drug_b_id: str) -> str:
    """Public helper — build a canonical pair_id from two raw ids."""
    return _canonical_pair_id(drug_a_id, drug_b_id)


def _canonical_json(obj: object) -> str:
    """JSON with sorted keys, no whitespace, ensure_ascii=False — stable bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def from_dict(d: dict) -> DDIPairRecord:
    """Build a DDIPairRecord from a plain dict (e.g. JSON fixture)."""
    evidence_refs = tuple(
        EvidenceRef(**ev) for ev in d.get("evidence_refs", [])
    )
    kwargs = {**d, "evidence_refs": evidence_refs}
    return DDIPairRecord(**kwargs)
