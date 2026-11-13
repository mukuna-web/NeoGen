"""Tests for the structure/isomerism cache (local RDKit + injectable fetcher)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.structure_cache import (
    analyze_stereo,
    build_structure_record,
    fetch_pubchem_props,
)

# Injectable PubChem response (PUG-REST shape) for an achiral molecule (ethanol).
_FAKE_PUBCHEM = {
    "PropertyTable": {
        "Properties": [
            {
                "CID": 702,
                "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "CanonicalSMILES": "CCO",
                "IsomericSMILES": "CCO",
            }
        ]
    }
}


def _fetcher(name: str) -> dict | None:
    return _FAKE_PUBCHEM


# ── stereo analysis (local RDKit) ───────────────────────────────────────────


@pytest.mark.unit
def test_achiral_molecule_is_legacy_2d() -> None:
    r = analyze_stereo("CCO")  # ethanol
    assert r["valid"] is True
    assert r["n_stereocenters_total"] == 0
    assert r["stereochemistry_status"] == "achiral"
    assert r["stereo_relevance"] == "none"
    assert r["required_feature_schema"] == "legacy_2d_v1"


@pytest.mark.unit
def test_single_enantiomer_requires_stereo_schema() -> None:
    r = analyze_stereo("C[C@@H](N)C(=O)O")  # L-alanine, one specified center
    assert r["n_specified"] == 1
    assert r["stereochemistry_status"] == "single_enantiomer"
    assert r["stereo_relevance"] == "enantiomer_sensitive"
    assert r["required_feature_schema"] == "stereo_2d_v1"


@pytest.mark.unit
def test_unspecified_center_flagged() -> None:
    r = analyze_stereo("CC(N)C(=O)O")  # alanine w/o stereo -> unassigned center
    assert r["n_stereocenters_total"] >= 1
    assert r["n_unspecified"] >= 1
    assert r["stereochemistry_status"] == "unspecified"
    assert r["required_feature_schema"] == "stereo_2d_v1"


# ── PubChem prop parsing ────────────────────────────────────────────────────


@pytest.mark.unit
def test_fetch_pubchem_props_parses_shape() -> None:
    props = fetch_pubchem_props("ethanol", fetcher=_fetcher)
    assert props is not None
    assert props["cid"] == "702"
    assert props["inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert props["isomeric_smiles"] == "CCO"


@pytest.mark.unit
def test_fetch_returns_none_on_empty() -> None:
    assert fetch_pubchem_props("x", fetcher=lambda n: None) is None
    assert fetch_pubchem_props("x", fetcher=lambda n: {"PropertyTable": {"Properties": []}}) is None


# ── record build + determinism ──────────────────────────────────────────────


@pytest.mark.unit
def test_build_record_is_deterministic_and_typed() -> None:
    props = fetch_pubchem_props("ethanol", fetcher=_fetcher)
    a = build_structure_record("ethanol", props, rxcui="123", retrieved_at="2026-05-29T00:00:00Z")
    b = build_structure_record("ethanol", props, rxcui="123", retrieved_at="2026-05-29T00:00:00Z")
    assert a.content_hash == b.content_hash
    assert len(a.content_hash) == 64
    assert a.pubchem_cid == "702"
    assert a.required_feature_schema == "legacy_2d_v1"  # ethanol is achiral
    assert a.rxnorm_rxcui == "123"


@pytest.mark.unit
def test_structure_record_has_no_severity() -> None:
    props = fetch_pubchem_props("ethanol", fetcher=_fetcher)
    rec = build_structure_record("ethanol", props)
    assert not hasattr(rec, "severity")
