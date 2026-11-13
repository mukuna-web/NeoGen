"""Drug identity spine schema.

The identity spine is the canonical join key for the whole flywheel. Every
external identifier (RxNorm, PubChem, ChEMBL, UNII, NDC, DrugCentral, ...)
resolves to one ``IdentitySpineRecord`` with one ``internal_drug_id``. This is
what lets DDI evidence from five different sources line up on the same drug.

Stereochemistry is first-class: ``stereochemistry_status`` and the
``salt_parent_relationship`` capture the distinctions that make two otherwise
"same" drugs behave differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StereochemistryStatus = Literal[
    "achiral",
    "single_enantiomer",
    "racemate",
    "diastereomer_mixture",
    "ez_defined",
    "unspecified",
    "unknown",
]


@dataclass(frozen=True)
class IdentitySpineRecord:
    """One canonical drug identity with cross-source identifiers."""

    internal_drug_id: str
    normalized_name: str
    rxnorm_rxcui: str | None = None
    pubchem_cid: str | None = None
    inchikey: str | None = None
    drugcentral_id: str | None = None
    chembl_id: str | None = None
    unii: str | None = None
    ndc_examples: tuple[str, ...] = field(default_factory=tuple)
    synonyms: tuple[str, ...] = field(default_factory=tuple)
    salt_parent_relationship: str | None = None  # e.g. "salt_of:<parent_id>"
    stereochemistry_status: StereochemistryStatus = "unknown"

    def __post_init__(self) -> None:
        if not self.internal_drug_id:
            raise ValueError("IdentitySpineRecord.internal_drug_id is required")
        if not self.normalized_name:
            raise ValueError("IdentitySpineRecord.normalized_name is required")
        if self.inchikey is not None and not _looks_like_inchikey(self.inchikey):
            raise ValueError(
                f"inchikey {self.inchikey!r} is not a valid InChIKey shape "
                "(expected 14-block-10-block-1, hyphen-separated)"
            )

    @property
    def has_structure(self) -> bool:
        return self.inchikey is not None or self.pubchem_cid is not None

    @property
    def is_stereo_resolved(self) -> bool:
        """True when chirality is explicitly known (not unspecified/unknown)."""
        return self.stereochemistry_status not in ("unspecified", "unknown")

    def crosswalk(self) -> dict[str, str]:
        """Return the non-null external identifier crosswalk."""
        out: dict[str, str] = {}
        for k in (
            "rxnorm_rxcui",
            "pubchem_cid",
            "inchikey",
            "drugcentral_id",
            "chembl_id",
            "unii",
        ):
            v = getattr(self, k)
            if v:
                out[k] = v
        return out


def _looks_like_inchikey(s: str) -> bool:
    parts = s.split("-")
    return (
        len(parts) == 3
        and len(parts[0]) == 14
        and len(parts[1]) == 10
        and len(parts[2]) == 1
        and parts[0].isalpha()
        and parts[1].isalpha()
    )
