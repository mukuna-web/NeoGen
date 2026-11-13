"""PK effect records (GAP-008): cited quantitative exposure changes.

A ``PKEffectRecord`` captures an AUC/Cmax change for a victim drug when a
perpetrator is coadministered — only when the label states a number. No
severity is ever inferred from magnitude. extraction_status tracks confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from neogen.data_flywheel.pk_extractor import extract_pk_effects
from neogen.data_flywheel.provenance import content_hash
from neogen.engine.ddi.pair_record import make_pair_id

_VALID_STATUS = {"extracted", "ambiguous", "absent", "needs_review"}


@dataclass(frozen=True)
class PKEffectRecord:
    drug_a: str
    drug_b: str
    canonical_pair_id: str
    victim: str | None
    perpetrator: str | None
    auc_ratio: float | None
    cmax_ratio: float | None
    clearance_change: str | None
    dose_context: str | None
    route: str | None
    population: str | None
    citation_span: str
    source_url: str
    extraction_status: str
    provenance_hash: str

    def __post_init__(self) -> None:
        if self.extraction_status not in _VALID_STATUS:
            raise ValueError(f"invalid extraction_status {self.extraction_status!r}")
        if self.extraction_status == "extracted" and not (
            self.citation_span and self.citation_span.strip()
        ):
            raise ValueError("extracted PK record requires a citation span")
        if self.extraction_status == "extracted" and self.auc_ratio is None and self.cmax_ratio is None:
            raise ValueError("extracted PK record requires an AUC or Cmax value")
        if self.canonical_pair_id != make_pair_id(self.drug_a, self.drug_b):
            raise ValueError("canonical_pair_id must equal make_pair_id(drug_a, drug_b)")

    def to_dict(self) -> dict:
        return asdict(self)


def build_pk_records(
    *,
    drug_a: str,
    drug_b: str,
    text: str,
    source_url: str,
    victim: str | None = None,
    perpetrator: str | None = None,
    route: str | None = None,
    population: str | None = None,
    dose_context: str | None = None,
) -> list[PKEffectRecord]:
    """Extract PK effect records from text. Empty if no quantitative AUC/Cmax."""
    effects = extract_pk_effects(text)
    pid = make_pair_id(drug_a, drug_b)
    records: list[PKEffectRecord] = []
    for eff in effects:
        auc = eff.ratio if eff.metric == "AUC" else None
        cmax = eff.ratio if eff.metric == "Cmax" else None
        records.append(
            PKEffectRecord(
                drug_a=drug_a, drug_b=drug_b, canonical_pair_id=pid,
                victim=victim, perpetrator=perpetrator,
                auc_ratio=auc, cmax_ratio=cmax, clearance_change=None,
                dose_context=dose_context, route=route, population=population,
                citation_span=eff.span, source_url=source_url,
                extraction_status="extracted",
                provenance_hash=content_hash(
                    f"{pid}|{eff.metric}|{eff.kind}|{eff.magnitude}|{eff.direction}|{eff.span}"
                ),
            )
        )
    return records
