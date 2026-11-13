"""Tests for the data-flywheel identity spine schema."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.identity_spine import IdentitySpineRecord


@pytest.mark.unit
def test_minimal_record_valid() -> None:
    r = IdentitySpineRecord(
        internal_drug_id="neogen-drug-0001",
        normalized_name="escitalopram",
    )
    assert r.internal_drug_id == "neogen-drug-0001"
    assert r.stereochemistry_status == "unknown"
    assert r.has_structure is False


@pytest.mark.unit
def test_required_fields_enforced() -> None:
    with pytest.raises(ValueError, match="internal_drug_id"):
        IdentitySpineRecord(internal_drug_id="", normalized_name="x")
    with pytest.raises(ValueError, match="normalized_name"):
        IdentitySpineRecord(internal_drug_id="x", normalized_name="")


@pytest.mark.unit
def test_valid_inchikey_accepted() -> None:
    # WHWDNIPEYRBRBM-UHFFFAOYSA-N shape: 14-10-1 blocks.
    r = IdentitySpineRecord(
        internal_drug_id="d1",
        normalized_name="caffeine",
        inchikey="RYYVLZVUVIJVGH-UHFFFAOYSA-N",
        pubchem_cid="2519",
    )
    assert r.has_structure is True
    assert "inchikey" in r.crosswalk()


@pytest.mark.unit
def test_bad_inchikey_rejected() -> None:
    with pytest.raises(ValueError, match="InChIKey"):
        IdentitySpineRecord(
            internal_drug_id="d1",
            normalized_name="x",
            inchikey="NOT-A-KEY",
        )


@pytest.mark.unit
def test_stereo_resolution_flag() -> None:
    unknown = IdentitySpineRecord(internal_drug_id="d", normalized_name="x")
    assert unknown.is_stereo_resolved is False
    single = IdentitySpineRecord(
        internal_drug_id="d",
        normalized_name="escitalopram",
        stereochemistry_status="single_enantiomer",
    )
    assert single.is_stereo_resolved is True


@pytest.mark.unit
def test_crosswalk_only_includes_non_null() -> None:
    r = IdentitySpineRecord(
        internal_drug_id="d",
        normalized_name="x",
        rxnorm_rxcui="123",
        chembl_id="CHEMBL25",
    )
    cw = r.crosswalk()
    assert cw == {"rxnorm_rxcui": "123", "chembl_id": "CHEMBL25"}
