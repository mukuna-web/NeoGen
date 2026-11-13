"""Tests for DailyMed SPL verification (offline, injectable fetcher)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.dailymed_spl import verify_set_id, verify_set_ids

# Mimics the DailyMed v2 shape: {"data": [ {...} ]}.
_FAKE_META = {
    "data": [
        {
            "spl_version": 20,
            "published_date": "Jun 23, 2025",
            "title": "WARFARIN SODIUM TABLET [TEVA]",
            "setid": "set-abc",
        }
    ]
}


def _ok_fetcher(set_id: str) -> dict | None:
    return _FAKE_META


def _none_fetcher(set_id: str) -> dict | None:
    return None


def _empty_fetcher(set_id: str) -> dict | None:
    return {"data": []}


@pytest.mark.unit
def test_verified_when_metadata_present() -> None:
    v = verify_set_id("set-abc", fetcher=_ok_fetcher)
    assert v.verified_against_spl is True
    assert v.spl_version == "20"
    assert "WARFARIN" in (v.title or "")
    assert v.source_url.endswith("setid=set-abc")
    assert v.content_hash and len(v.content_hash) == 64


@pytest.mark.unit
def test_not_verified_when_fetch_fails() -> None:
    v = verify_set_id("set-x", fetcher=_none_fetcher)
    assert v.verified_against_spl is False
    assert v.content_hash is None


@pytest.mark.unit
def test_not_verified_when_empty_data() -> None:
    v = verify_set_id("set-y", fetcher=_empty_fetcher)
    assert v.verified_against_spl is False


@pytest.mark.unit
def test_content_hash_is_deterministic() -> None:
    a = verify_set_id("set-abc", fetcher=_ok_fetcher)
    b = verify_set_id("set-abc", fetcher=_ok_fetcher)
    assert a.content_hash == b.content_hash


@pytest.mark.unit
def test_verify_set_ids_dedups() -> None:
    out = verify_set_ids(["set-abc", "set-abc"], fetcher=_ok_fetcher)
    assert list(out.keys()) == ["set-abc"]
    assert out["set-abc"].verified_against_spl is True


@pytest.mark.unit
def test_verification_has_no_severity_field() -> None:
    # SPL verification must not introduce any clinical severity claim.
    v = verify_set_id("set-abc", fetcher=_ok_fetcher)
    assert not hasattr(v, "severity")
