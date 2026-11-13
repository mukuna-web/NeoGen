"""Schema for FDA CYP/transporter DDI table records (GAP-014).

One ``FDADDITableRecord`` = one (drug, entity, role) classification extracted
from an official FDA table. These are mechanistic *classifications*, NOT clinical
severity and NOT pair-level claims. No severity field exists by design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ROLE_CLASSES = {"substrate", "inhibitor", "inducer", "marker_reaction", "unknown"}
EVIDENCE_CONTEXTS = {
    "in_vitro", "clinical_index", "clinical_example",
    "healthcare_professional_example", "unknown",
}
STRENGTHS = {"strong", "moderate", "weak", "sensitive", "moderate_sensitive", "index", "unknown"}


def classify_system(entity: str) -> str:
    """CYP vs transporter from an entity token."""
    e = entity.upper().replace(" ", "")
    if e.startswith("CYP"):
        return "CYP"
    transporters = ("P-GP", "PGP", "BCRP", "OATP", "OAT", "OCT", "MATE", "BSEP", "ABCB1")
    if any(e.startswith(t) for t in transporters):
        return "transporter"
    return "unknown"


@dataclass(frozen=True)
class FDADDITableRecord:
    source_url: str
    retrieved_at: str
    page_title: str
    table_id: str
    table_name: str
    table_category: str       # cyp | transporter | unknown
    system: str               # CYP | transporter | unknown
    entity: str               # CYP3A | CYP2C9 | P-gp | OATP1B1 | ...
    gene_symbol: str | None
    drug_name: str            # FDA display name (preserved)
    normalized_drug_name: str # lowercase comparison key
    role_class: str           # substrate | inhibitor | inducer | marker_reaction | unknown
    evidence_context: str
    strength: str
    footnotes: tuple[str, ...]
    row_text: str
    provenance_hash: str
    reviewer_status: str = "needs_review"

    def __post_init__(self) -> None:
        if self.role_class not in ROLE_CLASSES:
            raise ValueError(f"invalid role_class {self.role_class!r}")
        if self.evidence_context not in EVIDENCE_CONTEXTS:
            raise ValueError(f"invalid evidence_context {self.evidence_context!r}")
        if self.strength not in STRENGTHS:
            raise ValueError(f"invalid strength {self.strength!r}")
        if not self.row_text or not self.row_text.strip():
            raise ValueError("FDADDITableRecord requires row_text (never drop source text)")
        if not self.provenance_hash:
            raise ValueError("FDADDITableRecord requires provenance_hash")

    @property
    def is_clinical(self) -> bool:
        return self.evidence_context in (
            "clinical_index", "clinical_example", "healthcare_professional_example"
        )

    @property
    def is_perpetrator_role(self) -> bool:
        return self.role_class in ("inhibitor", "inducer")

    @property
    def is_victim_role(self) -> bool:
        return self.role_class == "substrate"

    def to_dict(self) -> dict:
        return asdict(self)
