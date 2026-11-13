"""Tests for openFDA label DDI evidence ingestion (offline, fixture-based)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.ddi_label_ingestion import (
    LabelDDIEvidence,
    ingest_label_result,
    to_pair_record,
)
from neogen.data_flywheel.evidence_schema import DDIEvidenceRecord
from neogen.data_flywheel.label_section_extractor import (
    extract_sections,
    find_named_partners,
    infer_mechanism,
)
from neogen.data_flywheel.provenance import content_hash

# Offline fixture: an amiodarone-like label naming warfarin (a real interaction)
# plus a generic clinical-pharmacology section with NO named partner.
FIXTURE = {
    "set_id": "test-set-123",
    "drug_interactions": [
        "7 DRUG INTERACTIONS. Coadministration of warfarin with amiodarone "
        "may increase the anticoagulant effect of warfarin. Amiodarone is an "
        "inhibitor of CYP2C9. Monitor INR closely."
    ],
    "clinical_pharmacology": [
        "12 CLINICAL PHARMACOLOGY. Amiodarone is metabolized by CYP3A4. "
        "No specific interacting drug is named in this paragraph."
    ],
    "openfda": {"generic_name": ["AMIODARONE"], "set_id": ["test-set-123"]},
}
SRC_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=test-set-123"
RETRIEVED = "2026-05-28T00:00:00Z"


# ── extractor ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_extractor_finds_known_section_names() -> None:
    sections = extract_sections(FIXTURE)
    assert "DRUG INTERACTIONS" in sections
    assert "CLINICAL PHARMACOLOGY" in sections


@pytest.mark.unit
def test_find_named_partner_detects_warfarin() -> None:
    text = FIXTURE["drug_interactions"][0]
    partners = find_named_partners(text, ["warfarin", "digoxin"])
    names = [p for p, _ in partners]
    assert "warfarin" in names
    assert "digoxin" not in names  # not present in text


@pytest.mark.unit
def test_infer_mechanism_only_when_explicit() -> None:
    assert infer_mechanism("Amiodarone is an inhibitor of CYP2C9.") == "cyp_inhibition"
    assert infer_mechanism("This drug is well tolerated.") is None


# ── provenance determinism ─────────────────────────────────────────────────


@pytest.mark.unit
def test_provenance_hash_is_deterministic() -> None:
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


# ── evidence requires citation ──────────────────────────────────────────────


@pytest.mark.unit
def test_label_evidence_requires_citation_span() -> None:
    with pytest.raises(ValueError, match="citation_span is required"):
        LabelDDIEvidence(
            drug_a="amiodarone",
            drug_b="warfarin",
            source_id="openfda_labels",
            source_url=SRC_URL,
            label_section="DRUG INTERACTIONS",
            citation_span="   ",  # blank -> reject
            retrieved_at=RETRIEVED,
            content_hash=content_hash("x"),
            mechanism="cyp_inhibition",
            severity=None,
            evidence_level="label",
            reviewer_status="needs_review",
        )


@pytest.mark.unit
def test_severity_cannot_be_set_without_citation_engine_record() -> None:
    # The engine-aligned DDIEvidenceRecord enforces no severity w/o citation.
    with pytest.raises(ValueError, match="no DDI severity without"):
        DDIEvidenceRecord.build(
            drug_a="amiodarone",
            drug_b="warfarin",
            mechanism="cyp_inhibition",
            source="openfda_labels",
            evidence_level="fda_label",
            provenance_hash="0" * 64,
            severity="major",  # no citation -> must fail
        )


# ── ambiguous extraction must not fabricate pairs ──────────────────────────


@pytest.mark.unit
def test_ambiguous_section_creates_no_pair() -> None:
    evs = ingest_label_result(
        "amiodarone", FIXTURE, source_url=SRC_URL, retrieved_at=RETRIEVED
    )
    # The clinical-pharmacology section names no partner -> pending, no pair.
    pending = [e for e in evs if e.label_section == "CLINICAL PHARMACOLOGY"]
    assert pending, "expected a pending clinical-pharmacology evidence"
    for e in pending:
        assert e.drug_b is None
        assert to_pair_record(e) is None


@pytest.mark.unit
def test_named_partner_creates_pairable_evidence() -> None:
    evs = ingest_label_result(
        "amiodarone", FIXTURE, source_url=SRC_URL, retrieved_at=RETRIEVED
    )
    named = [e for e in evs if e.drug_b == "warfarin"]
    assert named, "expected an amiodarone-warfarin evidence"
    pr = to_pair_record(named[0])
    assert pr is not None
    assert pr.pair_id == "amiodarone|warfarin"
    assert pr.severity is None
    assert pr.evidence_refs and pr.evidence_refs[0].citation


# ── repeated ingestion is stable ───────────────────────────────────────────


@pytest.mark.unit
def test_repeated_ingestion_is_stable() -> None:
    a = ingest_label_result("amiodarone", FIXTURE, source_url=SRC_URL, retrieved_at=RETRIEVED)
    b = ingest_label_result("amiodarone", FIXTURE, source_url=SRC_URL, retrieved_at=RETRIEVED)
    assert len(a) == len(b)
    assert [e.content_hash for e in a] == [e.content_hash for e in b]
    assert [e.drug_b for e in a] == [e.drug_b for e in b]


@pytest.mark.unit
def test_all_evidence_is_needs_review_and_label_level() -> None:
    evs = ingest_label_result("amiodarone", FIXTURE, source_url=SRC_URL, retrieved_at=RETRIEVED)
    assert evs
    for e in evs:
        assert e.evidence_level == "label"
        assert e.reviewer_status == "needs_review"
        assert e.severity is None
        assert e.citation_span.strip()
