"""Mechanism assertions for DDI pairs (GAP-004).

A ``MechanismAssertion`` records perpetrator/victim + mechanism + enzyme ONLY
when the source text explicitly supports it (with a citation span). Ambiguous
cases produce a ``directionality_gap`` instead of a forced claim. No severity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from neogen.data_flywheel.directionality_extractor import extract_direction
from neogen.data_flywheel.provenance import content_hash
from neogen.engine.ddi.pair_record import make_pair_id

_VALID_MECH = {
    "cyp_inhibition", "cyp_induction", "transporter_inhibition",
    "transporter_induction", "unknown",
}
_VALID_CONFIDENCE = {
    "explicit_label", "fda_table", "inferred_needs_review", "blocked",
    "bidirectional_needs_review",
}


@dataclass(frozen=True)
class MechanismAssertion:
    drug_a: str
    drug_b: str
    canonical_pair_id: str
    perpetrator: str | None
    victim: str | None
    mechanism: str
    enzyme_or_transporter: str | None
    evidence_text: str
    source_url: str
    confidence: str
    provenance_hash: str

    def __post_init__(self) -> None:
        if not (self.evidence_text and self.evidence_text.strip()):
            raise ValueError("MechanismAssertion requires evidence_text (citation span)")
        if self.mechanism not in _VALID_MECH:
            raise ValueError(f"invalid mechanism {self.mechanism!r}")
        if self.confidence not in _VALID_CONFIDENCE:
            raise ValueError(f"invalid confidence {self.confidence!r}")
        if self.canonical_pair_id != make_pair_id(self.drug_a, self.drug_b):
            raise ValueError("canonical_pair_id must equal make_pair_id(drug_a, drug_b)")
        for role in (self.perpetrator, self.victim):
            if role is not None and role not in (self.drug_a, self.drug_b):
                raise ValueError("perpetrator/victim must be drug_a or drug_b")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalityGap:
    drug_a: str
    drug_b: str
    canonical_pair_id: str
    reason: str
    evidence_text: str
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


def assert_from_evidence(
    *,
    drug_a: str,
    drug_b: str,
    citation_span: str,
    mechanism: str,
    enzyme_or_transporter: str | None,
    source_url: str,
) -> MechanismAssertion | DirectionalityGap:
    """Build a MechanismAssertion if direction is explicit; else a gap.

    ``mechanism`` and ``enzyme`` must already be source-supported (passed from
    the evidence record). Direction is extracted explicitly from the span.
    """
    if not (citation_span and citation_span.strip()):
        return DirectionalityGap(
            drug_a=drug_a, drug_b=drug_b, canonical_pair_id=make_pair_id(drug_a, drug_b),
            reason="no citation span", evidence_text="", source_url=source_url,
        )
    d = extract_direction(citation_span, drug_a, drug_b)
    pid = make_pair_id(drug_a, drug_b)
    if d.confidence == "explicit_label" and d.perpetrator and d.victim:
        return MechanismAssertion(
            drug_a=drug_a, drug_b=drug_b, canonical_pair_id=pid,
            perpetrator=d.perpetrator, victim=d.victim,
            mechanism=mechanism if mechanism in _VALID_MECH else "unknown",
            enzyme_or_transporter=enzyme_or_transporter,
            evidence_text=d.matched_span or citation_span,
            source_url=source_url, confidence="explicit_label",
            provenance_hash=content_hash(f"{pid}|{d.perpetrator}|{d.victim}|{mechanism}|{citation_span}"),
        )
    reason = (
        "both directions present" if d.confidence == "bidirectional_needs_review"
        else "no explicit directional verb linking the two drugs"
    )
    return DirectionalityGap(
        drug_a=drug_a, drug_b=drug_b, canonical_pair_id=pid,
        reason=reason, evidence_text=citation_span[:300], source_url=source_url,
    )
