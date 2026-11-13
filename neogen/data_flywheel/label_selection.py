"""Deterministic label ranking for DDI ingestion (GAP-012).

Given several candidate openFDA labels for a drug, pick the one best suited to
systemic DDI evidence: prefer systemic/oral formulation, presence of a DRUG
INTERACTIONS section, CLINICAL PHARMACOLOGY section, and explicit interacting
partners; penalize topical/local labels. Ties broken deterministically.

Pure functions over label result dicts; no network here.
"""

from __future__ import annotations

from dataclasses import dataclass

from neogen.data_flywheel.formulation_classifier import classify_formulation
from neogen.data_flywheel.label_section_extractor import (
    extract_sections,
    find_named_partners,
)
from neogen.data_flywheel.ddi_label_ingestion import DEFAULT_PARTNER_LEXICON

# Score weights (higher = more preferred). Formulation dominates.
W_SYSTEMIC = 100
W_TOPICAL_PENALTY = -100
W_HAS_DDI = 30
W_HAS_CLINPHARM = 15
W_PARTNER = 2  # per distinct named partner (capped)
PARTNER_CAP = 10


@dataclass(frozen=True)
class LabelCandidate:
    set_id: str | None
    title: str
    klass: str
    has_drug_interactions: bool
    has_clinical_pharmacology: bool
    n_named_partners: int
    score: int
    reasons: tuple[str, ...]


def _title(result: dict) -> str:
    of = result.get("openfda") or {}
    for k in ("brand_name", "generic_name"):
        v = of.get(k)
        if isinstance(v, list) and v:
            return str(v[0])
    return result.get("set_id") or "<no title>"


def score_label(result: dict, *, drug: str | None = None) -> LabelCandidate:
    fc = classify_formulation(result)
    sections = extract_sections(result)
    has_ddi = "DRUG INTERACTIONS" in sections
    has_cp = "CLINICAL PHARMACOLOGY" in sections

    lexicon = [p for p in DEFAULT_PARTNER_LEXICON if not drug or p.lower() != drug.lower()]
    partners: set[str] = set()
    for name in ("DRUG INTERACTIONS", "CLINICAL PHARMACOLOGY"):
        if name in sections:
            for p, _ in find_named_partners(sections[name], lexicon):
                partners.add(p)
    n_partners = len(partners)

    score = 0
    reasons: list[str] = []
    if fc.klass == "systemic":
        score += W_SYSTEMIC
        reasons.append("systemic (+100)")
    elif fc.klass == "topical":
        score += W_TOPICAL_PENALTY
        reasons.append("topical (-100)")
    if has_ddi:
        score += W_HAS_DDI
        reasons.append("has DRUG INTERACTIONS (+30)")
    if has_cp:
        score += W_HAS_CLINPHARM
        reasons.append("has CLINICAL PHARMACOLOGY (+15)")
    capped = min(n_partners, PARTNER_CAP)
    if capped:
        score += capped * W_PARTNER
        reasons.append(f"{n_partners} named partners (+{capped * W_PARTNER})")

    return LabelCandidate(
        set_id=result.get("set_id"),
        title=_title(result),
        klass=fc.klass,
        has_drug_interactions=has_ddi,
        has_clinical_pharmacology=has_cp,
        n_named_partners=n_partners,
        score=score,
        reasons=tuple(reasons),
    )


def rank_labels(results: list[dict], *, drug: str | None = None) -> list[LabelCandidate]:
    """Rank candidates by score desc, then set_id asc (deterministic)."""
    cands = [score_label(r, drug=drug) for r in results]
    return sorted(cands, key=lambda c: (-c.score, c.set_id or ""))


def select_best_label(
    results: list[dict], *, drug: str | None = None
) -> tuple[LabelCandidate | None, list[LabelCandidate]]:
    """Return (best, ranked). best is None if no systemic label with a DDI section.

    Adversarial guard: refuses to 'select' a topical-only or DDI-less label as
    the systemic DDI source — returns best=None so the caller opens a gap.
    """
    ranked = rank_labels(results, drug=drug)
    for c in ranked:
        if c.klass == "systemic" and c.has_drug_interactions:
            return c, ranked
    return None, ranked
