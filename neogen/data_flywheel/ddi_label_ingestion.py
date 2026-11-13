"""Normalize openFDA label sections into cited DDI evidence.

Pipeline: label JSON -> interaction sections -> ``LabelDDIEvidence`` (cited,
needs_review) -> optionally ``DDIEvidenceRecord`` (engine-aligned) and a safe
``DDIPairRecord`` ONLY when a partner drug is explicitly named.

Safety rules enforced here:
- No severity without a citation span (and this cycle keeps severity=None;
  labels do not state a severity grade, so we do not invent one).
- A pair (drug_b) is set ONLY when explicitly named in the label text.
- Ambiguous sections (no named partner) become needs_review evidence with
  ``drug_b=None`` and never produce a pair record.
- Citation spans are short quotes; full label bodies are never stored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from neogen.data_flywheel.label_section_extractor import (
    INTERACTION_SECTION_NAMES,
    extract_sections,
    find_enzymes,
    find_named_partners,
    infer_mechanism,
    split_sentences,
)
from neogen.data_flywheel.provenance import content_hash
from neogen.engine.ddi.pair_record import (
    DDIPairRecord,
    EvidenceRef,
    make_pair_id,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
NORM_DIR = _REPO_ROOT / "data" / "flywheel" / "normalized"
MANIFEST_DIR = _REPO_ROOT / "data" / "flywheel" / "manifests"

CITATION_MAX_CHARS = 300
SOURCE_ID = "openfda_labels"

# Curated partner lexicon (pilot drugs + common, well-known interacting drugs).
# A label's own drug is removed from its partner search.
DEFAULT_PARTNER_LEXICON = [
    "warfarin", "amiodarone", "ketoconazole", "rifampin", "rifampicin",
    "clarithromycin", "simvastatin", "midazolam", "digoxin", "cyclosporine",
    "fluconazole", "itraconazole", "phenytoin", "carbamazepine", "erythromycin",
    "atorvastatin", "tacrolimus", "verapamil", "diltiazem", "grapefruit",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(s: str, n: int = CITATION_MAX_CHARS) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


@dataclass(frozen=True)
class LabelDDIEvidence:
    """One cited interaction evidence item extracted from a drug label."""

    drug_a: str
    drug_b: str | None  # None/pending when no partner is explicitly named
    source_id: str
    source_url: str
    label_section: str
    citation_span: str
    retrieved_at: str
    content_hash: str
    mechanism: str | None
    severity: str | None
    evidence_level: str  # "label"
    reviewer_status: str  # "needs_review"
    enzymes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.drug_a:
            raise ValueError("LabelDDIEvidence.drug_a is required")
        if not self.citation_span or not self.citation_span.strip():
            raise ValueError("LabelDDIEvidence.citation_span is required (no claim without citation)")
        if self.severity is not None and not self.citation_span.strip():
            raise ValueError("severity requires a citation span (safety rule)")
        if self.evidence_level != "label":
            raise ValueError("evidence_level for label ingestion must be 'label'")

    @property
    def is_pairable(self) -> bool:
        """A safe pair can be formed only when a partner is explicitly named."""
        return bool(self.drug_b)

    def to_dict(self) -> dict:
        return asdict(self)


def ingest_label_result(
    drug_a: str,
    result: dict,
    *,
    source_url: str,
    retrieved_at: str | None = None,
    partner_lexicon: list[str] | None = None,
) -> list[LabelDDIEvidence]:
    """Turn one label result into a list of cited evidence items."""
    drug_a = drug_a.strip().lower()
    retrieved_at = retrieved_at or _now()
    lexicon = [p for p in (partner_lexicon or DEFAULT_PARTNER_LEXICON) if p.lower() != drug_a]

    out: list[LabelDDIEvidence] = []
    sections = extract_sections(result)
    for section_name, text in sections.items():
        if section_name not in INTERACTION_SECTION_NAMES:
            continue
        partners = find_named_partners(text, lexicon)
        if partners:
            for partner, sentence in partners:
                span = _truncate(sentence)
                out.append(
                    LabelDDIEvidence(
                        drug_a=drug_a,
                        drug_b=partner,
                        source_id=SOURCE_ID,
                        source_url=source_url,
                        label_section=section_name,
                        citation_span=span,
                        retrieved_at=retrieved_at,
                        content_hash=content_hash(f"{drug_a}|{partner}|{section_name}|{span}"),
                        mechanism=infer_mechanism(sentence),
                        severity=None,  # labels do not state a severity grade
                        evidence_level="label",
                        reviewer_status="needs_review",
                        enzymes=tuple(find_enzymes(sentence)),
                    )
                )
        else:
            # No explicit partner -> pending evidence, never a pair.
            snippet = _truncate(split_sentences(text)[0] if split_sentences(text) else text)
            out.append(
                LabelDDIEvidence(
                    drug_a=drug_a,
                    drug_b=None,
                    source_id=SOURCE_ID,
                    source_url=source_url,
                    label_section=section_name,
                    citation_span=snippet,
                    retrieved_at=retrieved_at,
                    content_hash=content_hash(f"{drug_a}|pending|{section_name}|{snippet}"),
                    mechanism=infer_mechanism(text),
                    severity=None,
                    evidence_level="label",
                    reviewer_status="needs_review",
                    enzymes=tuple(find_enzymes(text)),
                )
            )
    return out


def to_pair_record(ev: LabelDDIEvidence) -> DDIPairRecord | None:
    """Create a safe DDIPairRecord ONLY when a partner is explicitly named.

    Severity stays None (label does not grade severity). An fda_label
    EvidenceRef carries the citation span + source URL. ``role`` is 'unknown'
    because perpetrator/victim direction is not reliably inferable from a
    single label sentence in this cycle.
    """
    if not ev.is_pairable or ev.drug_b is None:
        return None
    a, b = ev.drug_a, ev.drug_b
    return DDIPairRecord(
        pair_id=make_pair_id(a, b),
        drug_a_id=a,
        drug_b_id=b,
        drug_a_name=a,
        drug_b_name=b,
        role="unknown",
        mechanism_category=ev.mechanism or "unknown",
        mechanism_note=f"label section: {ev.label_section}",
        enzyme_or_transporter=ev.enzymes[0] if ev.enzymes else None,
        stereo_relevance="unknown",
        feature_schema_version="legacy_2d_v1",
        evidence_refs=(
            EvidenceRef(
                evidence_type="fda_label",
                citation=ev.citation_span,
                url=ev.source_url,
                extracted_at_utc=ev.retrieved_at,
            ),
        ),
        severity=None,  # never assert severity without explicit label grading
        uncertainty_note="direction (perpetrator/victim) not yet resolved; needs_review",
        created_at_utc=ev.retrieved_at,
    )


def run_cycle(
    drugs: list[str],
    *,
    fixtures: dict[str, dict] | None = None,
    api_key: str | None = None,
    write: bool = True,
) -> dict:
    """Run the label ingestion cycle.

    ``fixtures`` maps drug -> label result dict (for offline/testing). When not
    provided, labels are fetched live from openFDA.
    """
    from neogen.data_flywheel import openfda_labels as ofda

    evidences: list[LabelDDIEvidence] = []
    pairs: list[DDIPairRecord] = []
    per_drug: list[dict] = []
    errors: list[str] = []

    for drug in drugs:
        rec = {"drug": drug}
        try:
            if fixtures and drug in fixtures:
                result = fixtures[drug]
                src_url = ofda.label_source_url(result)
            else:  # pragma: no cover - network
                label_json = ofda.fetch_label(drug, api_key=api_key)
                result = ofda.first_result(label_json)
                src_url = ofda.label_source_url(result)
            evs = ingest_label_result(drug, result, source_url=src_url)
            evidences.extend(evs)
            drug_pairs = [p for p in (to_pair_record(e) for e in evs) if p is not None]
            pairs.extend(drug_pairs)
            rec.update(
                status="ok",
                evidence=len(evs),
                pairs=len(drug_pairs),
                pending=sum(1 for e in evs if e.drug_b is None),
            )
        except Exception as exc:  # pragma: no cover - network
            rec.update(status="error", reason=str(exc)[:200])
            errors.append(f"{drug}: {exc}")
        per_drug.append(rec)

    summary = {
        "cycle_id": "openfda-ddi-label-cycle-2026-05-28",
        "retrieved_at": _now(),
        "source_id": SOURCE_ID,
        "drugs_attempted": len(drugs),
        "evidence_records": len(evidences),
        "pending_pairs": sum(1 for e in evidences if e.drug_b is None),
        "named_pairs": sum(1 for e in evidences if e.drug_b is not None),
        "pair_records": len(pairs),
        "per_drug": per_drug,
        "errors": errors,
    }

    if write:
        NORM_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        examples = {
            "cycle_id": summary["cycle_id"],
            "source_id": SOURCE_ID,
            "generated_at": summary["retrieved_at"],
            "note": "metadata-only; citation spans are short quotes, full labels not stored",
            "evidence": [e.to_dict() for e in evidences],
            "pair_records": [
                {
                    "pair_id": p.pair_id,
                    "drug_a_id": p.drug_a_id,
                    "drug_b_id": p.drug_b_id,
                    "mechanism_category": p.mechanism_category,
                    "enzyme_or_transporter": p.enzyme_or_transporter,
                    "severity": p.severity,
                    "content_hash": p.content_hash(),
                    "evidence_refs": [asdict(r) for r in p.evidence_refs],
                    "uncertainty_note": p.uncertainty_note,
                }
                for p in pairs
            ],
        }
        (NORM_DIR / "openfda_ddi_label_examples.json").write_text(
            json.dumps(examples, indent=2, sort_keys=True)
        )
        (MANIFEST_DIR / "openfda_ddi_label_cycle_2026-05-28.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )

    return summary
