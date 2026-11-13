"""Tests for isomer counterfactual records (offline)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.isomer_counterfactuals import build_counterfactual
from neogen.data_flywheel.isomer_resolution import resolve_isomer


def _pubchem(cid, inchikey, smiles):
    return {"PropertyTable": {"Properties": [
        {"CID": cid, "InChIKey": inchikey, "SMILES": smiles, "ConnectivitySMILES": smiles}]}}


# Two true enantiomers: SAME skeleton, DIFFERENT stereo block, both specified.
_S = _pubchem(54678487, "PJVWKTKQMONHTI-HNNXBMFYSA-N", "C[C@@H](N)C(=O)O")
_R = _pubchem(54678488, "PJVWKTKQMONHTI-SECBINFHSA-N", "C[C@H](N)C(=O)O")
# The stereo-unspecified parent (both queries collapse to it).
_PARENT = _pubchem(54678486, "PJVWKTKQMONHTI-UHFFFAOYSA-N", "CC(N)C(=O)O")


@pytest.mark.unit
def test_true_enantiomer_pair_collides_2d_separates_stereo() -> None:
    a = resolve_isomer("(S)-x", fetcher=lambda n: _S, retrieved_at="t")
    b = resolve_isomer("(R)-x", fetcher=lambda n: _R, retrieved_at="t")
    cf = build_counterfactual("x", a, b, retrieved_at="t")
    assert cf.same_connectivity is True
    assert cf.different_stereochemistry is True
    assert cf.legacy_2d_expected_collision is True
    assert cf.stereo_2d_expected_separation is True


@pytest.mark.unit
def test_collapsed_pair_cannot_separate() -> None:
    a = resolve_isomer("(S)-warfarin", fetcher=lambda n: _PARENT, retrieved_at="t")
    b = resolve_isomer("(R)-warfarin", fetcher=lambda n: _PARENT, retrieved_at="t")
    cf = build_counterfactual("warfarin", a, b, retrieved_at="t")
    assert cf.same_connectivity is True
    assert cf.different_stereochemistry is False  # same InChIKey -> collapsed
    assert cf.stereo_2d_expected_separation is False
    assert "gap" in cf.note.lower()


@pytest.mark.unit
def test_counterfactual_is_deterministic() -> None:
    a = resolve_isomer("(S)-x", fetcher=lambda n: _S, retrieved_at="t")
    b = resolve_isomer("(R)-x", fetcher=lambda n: _R, retrieved_at="t")
    cf1 = build_counterfactual("x", a, b, retrieved_at="t")
    cf2 = build_counterfactual("x", a, b, retrieved_at="t")
    assert cf1.content_hash == cf2.content_hash


@pytest.mark.unit
def test_no_severity_or_clinical_field() -> None:
    a = resolve_isomer("(S)-x", fetcher=lambda n: _S, retrieved_at="t")
    b = resolve_isomer("(R)-x", fetcher=lambda n: _R, retrieved_at="t")
    cf = build_counterfactual("x", a, b, retrieved_at="t")
    assert not hasattr(cf, "severity")
