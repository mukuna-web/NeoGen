"""Tests for RxNorm identity resolution (offline, injectable fetcher)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.identity_resolution import (
    build_identity_spine,
    resolve_names,
    resolve_rxcui,
    unresolved_names,
)
from neogen.data_flywheel.identity_spine import IdentitySpineRecord

_DB = {
    "warfarin": {"idGroup": {"name": "warfarin", "rxnormId": ["11289"]}},
    "amiodarone": {"idGroup": {"name": "amiodarone", "rxnormId": ["703"]}},
    "notadrug": {"idGroup": {"name": "notadrug"}},  # no rxnormId
}


def _fetcher(name: str) -> dict | None:
    return _DB.get(name.lower())


@pytest.mark.unit
def test_resolve_known_drug() -> None:
    assert resolve_rxcui("warfarin", fetcher=_fetcher) == "11289"
    assert resolve_rxcui("amiodarone", fetcher=_fetcher) == "703"


@pytest.mark.unit
def test_resolve_unknown_returns_none() -> None:
    assert resolve_rxcui("notadrug", fetcher=_fetcher) is None
    assert resolve_rxcui("missing", fetcher=_fetcher) is None


@pytest.mark.unit
def test_resolve_names_dedups_case_insensitive() -> None:
    out = resolve_names(["Warfarin", "warfarin", "AMIODARONE"], fetcher=_fetcher)
    assert set(out.keys()) == {"warfarin", "amiodarone"}
    assert out["warfarin"] == "11289"


@pytest.mark.unit
def test_build_identity_spine_sets_rxcui() -> None:
    mapping = {"warfarin": "11289", "notadrug": None}
    recs = build_identity_spine(mapping)
    by_name = {r.normalized_name: r for r in recs}
    assert isinstance(by_name["warfarin"], IdentitySpineRecord)
    assert by_name["warfarin"].rxnorm_rxcui == "11289"
    assert by_name["warfarin"].internal_drug_id == "neogen-drug-warfarin"
    # Unresolved -> provisional id, no rxcui (a gap, not a fabricated id).
    assert by_name["notadrug"].rxnorm_rxcui is None
    assert by_name["notadrug"].internal_drug_id.endswith("-provisional")


@pytest.mark.unit
def test_unresolved_names_listed() -> None:
    mapping = {"warfarin": "11289", "notadrug": None}
    assert unresolved_names(mapping) == ["notadrug"]


@pytest.mark.unit
def test_identity_record_has_no_severity() -> None:
    rec = build_identity_spine({"warfarin": "11289"})[0]
    assert not hasattr(rec, "severity")
