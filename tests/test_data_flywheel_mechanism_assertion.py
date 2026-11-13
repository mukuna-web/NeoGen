"""Tests for mechanism assertions (explicit-only, cited)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.mechanism_assertion import (
    DirectionalityGap,
    MechanismAssertion,
    assert_from_evidence,
)
from neogen.engine.ddi.pair_record import make_pair_id


@pytest.mark.unit
def test_explicit_direction_yields_assertion() -> None:
    out = assert_from_evidence(
        drug_a="amiodarone", drug_b="warfarin",
        citation_span="Amiodarone inhibits the metabolism of warfarin via CYP2C9.",
        mechanism="cyp_inhibition", enzyme_or_transporter="CYP2C9",
        source_url="https://dailymed/x",
    )
    assert isinstance(out, MechanismAssertion)
    assert out.perpetrator == "amiodarone" and out.victim == "warfarin"
    assert out.confidence == "explicit_label"


@pytest.mark.unit
def test_ambiguous_direction_yields_gap_not_claim() -> None:
    out = assert_from_evidence(
        drug_a="amiodarone", drug_b="warfarin",
        citation_span="Amiodarone and warfarin were both administered to patients.",
        mechanism="unknown", enzyme_or_transporter=None,
        source_url="https://dailymed/x",
    )
    assert isinstance(out, DirectionalityGap)
    assert "no explicit" in out.reason


@pytest.mark.unit
def test_no_citation_yields_gap() -> None:
    out = assert_from_evidence(
        drug_a="a", drug_b="b", citation_span="   ", mechanism="unknown",
        enzyme_or_transporter=None, source_url="u",
    )
    assert isinstance(out, DirectionalityGap)
    assert out.reason == "no citation span"


@pytest.mark.unit
def test_assertion_requires_evidence_text() -> None:
    with pytest.raises(ValueError, match="evidence_text"):
        MechanismAssertion(
            drug_a="a", drug_b="b", canonical_pair_id=make_pair_id("a", "b"),
            perpetrator="a", victim="b", mechanism="cyp_inhibition",
            enzyme_or_transporter=None, evidence_text="", source_url="u",
            confidence="explicit_label", provenance_hash="0" * 64,
        )


@pytest.mark.unit
def test_invalid_mechanism_rejected() -> None:
    with pytest.raises(ValueError, match="invalid mechanism"):
        MechanismAssertion(
            drug_a="a", drug_b="b", canonical_pair_id=make_pair_id("a", "b"),
            perpetrator=None, victim=None, mechanism="telepathy",
            enzyme_or_transporter=None, evidence_text="cited", source_url="u",
            confidence="explicit_label", provenance_hash="0" * 64,
        )


@pytest.mark.unit
def test_no_severity_field() -> None:
    out = assert_from_evidence(
        drug_a="amiodarone", drug_b="warfarin",
        citation_span="Amiodarone inhibits warfarin metabolism.",
        mechanism="cyp_inhibition", enzyme_or_transporter="CYP2C9", source_url="u",
    )
    assert not hasattr(out, "severity")
