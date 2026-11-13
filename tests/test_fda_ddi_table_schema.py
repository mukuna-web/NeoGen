"""Tests for the FDA DDI table record schema (GAP-014).

These records are mechanistic *classifications*, not clinical severity. The
schema must reject invalid enums, demand row_text + provenance_hash, and expose
no severity field by design.
"""

from __future__ import annotations

import pytest

from neogen.data_flywheel.fda_ddi_table_schema import (
    FDADDITableRecord,
    classify_system,
)


def _rec(**over) -> FDADDITableRecord:
    base = dict(
        source_url="https://www.fda.gov/x", retrieved_at="2026-05-29T00:00:00Z",
        page_title="FDA clinical examples", table_id="clinical_table_1",
        table_name="CYP/Transporter Clinical Substrates, Inhibitors, or Inducers",
        table_category="cyp", system="CYP", entity="CYP3A", gene_symbol=None,
        drug_name="clarithromycin", normalized_drug_name="clarithromycin",
        role_class="inhibitor", evidence_context="clinical_example",
        strength="strong", footnotes=tuple(), row_text="clarithromycin | CYP3A strong inhibitor",
        provenance_hash="a" * 64, reviewer_status="needs_review",
    )
    base.update(over)
    return FDADDITableRecord(**base)


@pytest.mark.unit
def test_classify_system_cyp_transporter_unknown() -> None:
    assert classify_system("CYP3A") == "CYP"
    assert classify_system("CYP2C9") == "CYP"
    assert classify_system("P-gp") == "transporter"
    assert classify_system("OATP1B1") == "transporter"
    assert classify_system("BCRP") == "transporter"
    assert classify_system("aspirin") == "unknown"


@pytest.mark.unit
def test_valid_record_constructs() -> None:
    r = _rec()
    assert r.entity == "CYP3A"
    assert r.is_clinical is True
    assert r.is_perpetrator_role is True
    assert r.is_victim_role is False


@pytest.mark.unit
def test_substrate_is_victim_role() -> None:
    r = _rec(role_class="substrate", strength="sensitive")
    assert r.is_victim_role is True
    assert r.is_perpetrator_role is False


@pytest.mark.unit
def test_inducer_is_perpetrator_role() -> None:
    r = _rec(role_class="inducer", strength="strong")
    assert r.is_perpetrator_role is True


@pytest.mark.unit
def test_invalid_role_class_rejected() -> None:
    with pytest.raises(ValueError, match="invalid role_class"):
        _rec(role_class="telepath")


@pytest.mark.unit
def test_invalid_evidence_context_rejected() -> None:
    with pytest.raises(ValueError, match="invalid evidence_context"):
        _rec(evidence_context="rumor")


@pytest.mark.unit
def test_invalid_strength_rejected() -> None:
    with pytest.raises(ValueError, match="invalid strength"):
        _rec(strength="catastrophic")


@pytest.mark.unit
def test_row_text_required() -> None:
    with pytest.raises(ValueError, match="row_text"):
        _rec(row_text="   ")


@pytest.mark.unit
def test_provenance_hash_required() -> None:
    with pytest.raises(ValueError, match="provenance_hash"):
        _rec(provenance_hash="")


@pytest.mark.unit
def test_no_severity_field_by_design() -> None:
    r = _rec()
    assert not hasattr(r, "severity")
    # to_dict must not smuggle a severity key either
    assert "severity" not in r.to_dict()
