"""DDI evidence schema for the data flywheel.

This is the *ingestion-side* evidence record: a single cited assertion that a
drug pair interacts, normalized into a typed, replayable shape. It is the
bridge between raw crawled sources and the DDI science layer's
``DDIPairRecord``.

Hard safety rules enforced here (NOT optional):
- ``severity`` MUST be ``None`` unless a non-empty ``citation`` is present.
- A stereochemistry-sensitive record (``stereo_relevance`` other than ``none``
  / ``unknown``) MUST require a stereo-aware feature schema, never the
  chirality-blind ``legacy_2d_v1``.
- ``canonical_pair_id`` is computed with the SAME function the science engine
  uses (`neogen.engine.ddi.pair_record.make_pair_id`) so records line up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Reuse the engine's canonical pair-id + accepted schema set so the flywheel
# and the science engine never disagree on identity. (Read-only import; the
# engine module is not modified.)
from neogen.engine.ddi.pair_record import (  # noqa: E402
    ACCEPTED_FEATURE_SCHEMAS,
    make_pair_id,
)

Mechanism = Literal[
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

EvidenceLevel = Literal[
    "fda_label",
    "regulatory_review",
    "peer_reviewed",
    "curated_db",
    "pharmacovigilance_statistical",
    "preclinical",
    "hypothesis",
]

StereoRelevance = Literal[
    "none",
    "enantiomer_sensitive",
    "diastereomer_sensitive",
    "ez_isomer_sensitive",
    "unknown",
]

_STEREO_SENSITIVE = frozenset(
    {"enantiomer_sensitive", "diastereomer_sensitive", "ez_isomer_sensitive"}
)
_CHIRALITY_BLIND_SCHEMAS = frozenset({"legacy_2d_v1"})


@dataclass(frozen=True)
class DDIEvidenceRecord:
    """One cited DDI assertion, ingestion-normalized."""

    drug_a: str
    drug_b: str
    canonical_pair_id: str
    perpetrator: str | None
    victim: str | None
    mechanism: Mechanism
    source: str  # source_id from the registry
    citation: str | None
    severity: str | None
    evidence_level: EvidenceLevel
    stereo_relevance: StereoRelevance
    feature_schema_required: str
    provenance_hash: str

    def __post_init__(self) -> None:
        # --- canonical pair id must match the engine's convention ---
        expected = make_pair_id(self.drug_a, self.drug_b)
        if self.canonical_pair_id != expected:
            raise ValueError(
                f"canonical_pair_id {self.canonical_pair_id!r} must equal "
                f"{expected!r} (engine make_pair_id of drug_a/drug_b)"
            )
        # --- no clinical claim (severity) without a citation ---
        if self.severity is not None and not (self.citation and self.citation.strip()):
            raise ValueError(
                "severity is set but no citation provided. Safety rule: no DDI "
                "severity without a source/citation."
            )
        # --- feature schema must be a known one ---
        if self.feature_schema_required not in ACCEPTED_FEATURE_SCHEMAS:
            raise ValueError(
                f"feature_schema_required {self.feature_schema_required!r} not "
                f"in accepted set {sorted(ACCEPTED_FEATURE_SCHEMAS)}"
            )
        # --- stereo-sensitive records may not use a chirality-blind schema ---
        if (
            self.stereo_relevance in _STEREO_SENSITIVE
            and self.feature_schema_required in _CHIRALITY_BLIND_SCHEMAS
        ):
            raise ValueError(
                f"stereo_relevance={self.stereo_relevance!r} requires a "
                "stereo-aware feature schema (stereo_2d_v1 or conformer_3d_v1); "
                f"got chirality-blind {self.feature_schema_required!r}."
            )
        # --- perpetrator/victim, if named, must be one of the two drugs ---
        for role_name, role_val in (("perpetrator", self.perpetrator), ("victim", self.victim)):
            if role_val is not None and role_val not in (self.drug_a, self.drug_b):
                raise ValueError(
                    f"{role_name} {role_val!r} must be drug_a or drug_b"
                )

    @property
    def is_clinical_claim(self) -> bool:
        """True if this record asserts a severity (hence requires a citation)."""
        return self.severity is not None

    @classmethod
    def build(
        cls,
        *,
        drug_a: str,
        drug_b: str,
        mechanism: Mechanism,
        source: str,
        evidence_level: EvidenceLevel,
        provenance_hash: str,
        perpetrator: str | None = None,
        victim: str | None = None,
        citation: str | None = None,
        severity: str | None = None,
        stereo_relevance: StereoRelevance = "unknown",
        feature_schema_required: str = "legacy_2d_v1",
    ) -> "DDIEvidenceRecord":
        """Convenience builder that computes ``canonical_pair_id`` for you."""
        return cls(
            drug_a=drug_a,
            drug_b=drug_b,
            canonical_pair_id=make_pair_id(drug_a, drug_b),
            perpetrator=perpetrator,
            victim=victim,
            mechanism=mechanism,
            source=source,
            citation=citation,
            severity=severity,
            evidence_level=evidence_level,
            stereo_relevance=stereo_relevance,
            feature_schema_required=feature_schema_required,
            provenance_hash=provenance_hash,
        )
