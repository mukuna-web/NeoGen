"""Tests for enantiomer-aware structure resolution (offline)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.isomer_resolution import (
    inchikey_blocks,
    looks_enantiomeric,
    resolve_isomer,
    same_cid,
)


def _pubchem(cid, inchikey, smiles):
    return {
        "PropertyTable": {
            "Properties": [
                {"CID": cid, "InChIKey": inchikey, "SMILES": smiles, "ConnectivitySMILES": smiles}
            ]
        }
    }


# A specified single enantiomer (1 chiral center).
_SPEC = _pubchem(100, "AAAAAAAAAAAAAA-HNNXBMFYSA-N", "C[C@@H](N)C(=O)O")
# A stereo-unspecified parent (skeleton same, UHFFFAOYSA block, flat SMILES).
_UNSPEC = _pubchem(54678486, "PJVWKTKQMONHTI-UHFFFAOYSA-N", "CC(N)C(=O)O")


@pytest.mark.unit
def test_looks_enantiomeric() -> None:
    assert looks_enantiomeric("(S)-warfarin")
    assert looks_enantiomeric("esomeprazole")
    assert looks_enantiomeric("dexibuprofen")
    assert not looks_enantiomeric("warfarin")
    assert not looks_enantiomeric("amiodarone")


@pytest.mark.unit
def test_inchikey_blocks() -> None:
    skel, stereo = inchikey_blocks("PJVWKTKQMONHTI-UHFFFAOYSA-N")
    assert skel == "PJVWKTKQMONHTI"
    assert stereo == "UHFFFAOYSA"
    assert inchikey_blocks(None) == (None, None)


@pytest.mark.unit
def test_specified_enantiomer_resolves() -> None:
    r = resolve_isomer("(S)-alanine-like", fetcher=lambda n: _SPEC)
    assert r.status == "resolved"
    assert r.n_specified == 1
    assert r.required_feature_schema == "stereo_2d_v1"
    assert r.enantiomer_query_collapsed is False


@pytest.mark.unit
def test_enantiomeric_query_collapse_is_flagged() -> None:
    # An explicitly enantiomeric query that returns the unspecified parent.
    r = resolve_isomer("(S)-warfarin", fetcher=lambda n: _UNSPEC)
    assert r.enantiomeric_query is True
    assert r.stereo_unspecified is True
    assert r.enantiomer_query_collapsed is True
    assert r.status == "collapsed_needs_gap"


@pytest.mark.unit
def test_unresolved_when_no_props() -> None:
    r = resolve_isomer("nope", fetcher=lambda n: None)
    assert r.status == "unresolved"
    assert r.pubchem_cid is None


@pytest.mark.unit
def test_same_cid_detects_collapse() -> None:
    a = resolve_isomer("(S)-warfarin", fetcher=lambda n: _UNSPEC)
    b = resolve_isomer("(R)-warfarin", fetcher=lambda n: _UNSPEC)
    assert same_cid(a, b) is True  # both collapsed to the same parent CID


@pytest.mark.unit
def test_resolution_is_deterministic() -> None:
    a = resolve_isomer("(S)-x", fetcher=lambda n: _SPEC, retrieved_at="2026-05-29T00:00:00Z")
    b = resolve_isomer("(S)-x", fetcher=lambda n: _SPEC, retrieved_at="2026-05-29T00:00:00Z")
    assert a.content_hash == b.content_hash
