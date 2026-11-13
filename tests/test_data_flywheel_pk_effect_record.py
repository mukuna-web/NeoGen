"""Tests for PK effect records (offline)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.pk_effect_record import PKEffectRecord, build_pk_records
from neogen.engine.ddi.pair_record import make_pair_id


@pytest.mark.unit
def test_build_from_auc_fold_text() -> None:
    recs = build_pk_records(
        drug_a="clarithromycin", drug_b="midazolam",
        text="Midazolam AUC increased 3.5-fold with clarithromycin.",
        source_url="https://dailymed/x", victim="midazolam", perpetrator="clarithromycin",
    )
    assert len(recs) == 1
    assert recs[0].auc_ratio == 3.5
    assert recs[0].extraction_status == "extracted"
    assert recs[0].canonical_pair_id == make_pair_id("clarithromycin", "midazolam")


@pytest.mark.unit
def test_no_number_yields_no_record() -> None:
    recs = build_pk_records(
        drug_a="a", drug_b="b",
        text="Coadministration may increase exposure (no value given).",
        source_url="u",
    )
    assert recs == []


@pytest.mark.unit
def test_extracted_record_requires_value() -> None:
    with pytest.raises(ValueError, match="AUC or Cmax value"):
        PKEffectRecord(
            drug_a="a", drug_b="b", canonical_pair_id=make_pair_id("a", "b"),
            victim=None, perpetrator=None, auc_ratio=None, cmax_ratio=None,
            clearance_change=None, dose_context=None, route=None, population=None,
            citation_span="cited", source_url="u", extraction_status="extracted",
            provenance_hash="0" * 64,
        )


@pytest.mark.unit
def test_invalid_status_rejected() -> None:
    with pytest.raises(ValueError, match="extraction_status"):
        PKEffectRecord(
            drug_a="a", drug_b="b", canonical_pair_id=make_pair_id("a", "b"),
            victim=None, perpetrator=None, auc_ratio=1.5, cmax_ratio=None,
            clearance_change=None, dose_context=None, route=None, population=None,
            citation_span="cited", source_url="u", extraction_status="guessed",
            provenance_hash="0" * 64,
        )


@pytest.mark.unit
def test_no_severity_field() -> None:
    recs = build_pk_records(
        drug_a="clarithromycin", drug_b="midazolam",
        text="Midazolam AUC increased 3.5-fold.", source_url="u",
    )
    assert recs and not hasattr(recs[0], "severity")
