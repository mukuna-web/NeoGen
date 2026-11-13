"""Tests for the data-flywheel DDI evidence schema (safety-critical)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.evidence_schema import DDIEvidenceRecord
from neogen.engine.ddi.pair_record import make_pair_id


@pytest.mark.unit
def test_build_computes_canonical_pair_id() -> None:
    rec = DDIEvidenceRecord.build(
        drug_a="warfarin",
        drug_b="amiodarone",
        mechanism="cyp_inhibition",
        source="fda_cyp_transporter_tables",
        evidence_level="fda_label",
        provenance_hash="0" * 64,
    )
    assert rec.canonical_pair_id == make_pair_id("warfarin", "amiodarone")


@pytest.mark.unit
def test_severity_without_citation_raises() -> None:
    with pytest.raises(ValueError, match="no DDI severity without"):
        DDIEvidenceRecord.build(
            drug_a="a",
            drug_b="b",
            mechanism="cyp_inhibition",
            source="ddinter",
            evidence_level="curated_db",
            provenance_hash="0" * 64,
            severity="major",  # no citation -> must fail
        )


@pytest.mark.unit
def test_severity_with_citation_allowed() -> None:
    rec = DDIEvidenceRecord.build(
        drug_a="a",
        drug_b="b",
        mechanism="cyp_inhibition",
        source="ddinter",
        evidence_level="curated_db",
        provenance_hash="0" * 64,
        severity="moderate",
        citation="DDInter row id=123; retrieved 2026-05-28",
    )
    assert rec.severity == "moderate"
    assert rec.is_clinical_claim is True


@pytest.mark.unit
def test_stereo_sensitive_rejects_chirality_blind_schema() -> None:
    with pytest.raises(ValueError, match="stereo-aware feature schema"):
        DDIEvidenceRecord.build(
            drug_a="a",
            drug_b="b",
            mechanism="cyp_inhibition",
            source="pubmed",
            evidence_level="peer_reviewed",
            provenance_hash="0" * 64,
            stereo_relevance="enantiomer_sensitive",
            feature_schema_required="legacy_2d_v1",  # blind -> must fail
        )


@pytest.mark.unit
def test_stereo_sensitive_accepts_stereo_schema() -> None:
    rec = DDIEvidenceRecord.build(
        drug_a="a",
        drug_b="b",
        mechanism="cyp_inhibition",
        source="pubmed",
        evidence_level="peer_reviewed",
        provenance_hash="0" * 64,
        stereo_relevance="enantiomer_sensitive",
        feature_schema_required="stereo_2d_v1",
    )
    assert rec.feature_schema_required == "stereo_2d_v1"


@pytest.mark.unit
def test_unknown_feature_schema_rejected() -> None:
    with pytest.raises(ValueError, match="not in accepted set"):
        DDIEvidenceRecord.build(
            drug_a="a",
            drug_b="b",
            mechanism="unknown",
            source="x",
            evidence_level="hypothesis",
            provenance_hash="0" * 64,
            feature_schema_required="quantum_5d_v9",
        )


@pytest.mark.unit
def test_perpetrator_must_be_one_of_the_pair() -> None:
    with pytest.raises(ValueError, match="perpetrator"):
        DDIEvidenceRecord.build(
            drug_a="a",
            drug_b="b",
            mechanism="cyp_inhibition",
            source="x",
            evidence_level="hypothesis",
            provenance_hash="0" * 64,
            perpetrator="c",  # not a or b -> fail
        )


@pytest.mark.unit
def test_canonical_pair_id_mismatch_raises_direct_construction() -> None:
    with pytest.raises(ValueError, match="canonical_pair_id"):
        DDIEvidenceRecord(
            drug_a="a",
            drug_b="b",
            canonical_pair_id="b|a",  # wrong order
            perpetrator=None,
            victim=None,
            mechanism="unknown",
            source="x",
            citation=None,
            severity=None,
            evidence_level="hypothesis",
            stereo_relevance="unknown",
            feature_schema_required="legacy_2d_v1",
            provenance_hash="0" * 64,
        )
