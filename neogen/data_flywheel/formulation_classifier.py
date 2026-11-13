"""Classify an openFDA label's formulation/route as systemic vs topical (GAP-012).

The Cycle-2 ingestion picked a ketoconazole *shampoo* label, which carries no
systemic DDI evidence. This module classifies a label so the selector can prefer
systemic/oral labels for DDI work. Pure function over a label result dict.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEMIC_ROUTES = {
    "ORAL", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS", "INTRAVENOUS DRIP",
    "INTRA-ARTERIAL", "PARENTERAL",
}
TOPICAL_ROUTES = {
    "TOPICAL", "CUTANEOUS", "OPHTHALMIC", "OTIC", "DENTAL", "VAGINAL",
    "TRANSDERMAL", "RECTAL", "NASAL",
}
_TOPICAL_KEYWORDS = (
    "shampoo", "cream", "ointment", "lotion", "gel", "foam", "ophthalmic",
    "otic", "topical", "transdermal patch", "suppository",
)


@dataclass(frozen=True)
class FormulationClass:
    klass: str  # systemic | topical | unknown
    routes: tuple[str, ...]
    dosage_forms: tuple[str, ...]
    reasons: tuple[str, ...]


def _listfield(result: dict, *path: str) -> list[str]:
    cur: object = result
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = None
            break
    if isinstance(cur, list):
        return [str(x) for x in cur]
    if isinstance(cur, str):
        return [cur]
    return []


def classify_formulation(result: dict) -> FormulationClass:
    routes = [r.upper() for r in _listfield(result, "openfda", "route")]
    dosage = [d.upper() for d in _listfield(result, "openfda", "dosage_form")]
    brand = " ".join(_listfield(result, "openfda", "brand_name")).lower()
    generic = " ".join(_listfield(result, "openfda", "generic_name")).lower()
    blob = f"{brand} {generic} {' '.join(dosage).lower()}"

    reasons: list[str] = []
    topical_kw = [k for k in _TOPICAL_KEYWORDS if k in blob]
    if topical_kw:
        reasons.append(f"topical keyword(s): {topical_kw}")

    has_systemic = any(r in SYSTEMIC_ROUTES for r in routes)
    has_topical = any(r in TOPICAL_ROUTES for r in routes)
    if routes:
        reasons.append(f"route(s): {routes}")

    if has_systemic and not topical_kw:
        klass = "systemic"
    elif has_topical or topical_kw:
        klass = "topical"
    elif has_systemic:
        klass = "systemic"
    else:
        klass = "unknown"
        reasons.append("no route/formulation signal")

    return FormulationClass(
        klass=klass,
        routes=tuple(routes),
        dosage_forms=tuple(dosage),
        reasons=tuple(reasons),
    )
