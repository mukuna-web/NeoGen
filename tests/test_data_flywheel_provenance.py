"""Tests for the data-flywheel provenance model."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.provenance import Provenance, content_hash


@pytest.mark.unit
def test_content_hash_is_64_hex_and_stable() -> None:
    h1 = content_hash("hello")
    h2 = content_hash("hello")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)
    assert content_hash("hello") != content_hash("world")


@pytest.mark.unit
def test_for_payload_builds_valid_provenance() -> None:
    p = Provenance.for_payload(
        source_id="unichem",
        source_url="https://www.ebi.ac.uk/unichem/",
        retrieved_at="2026-05-28T00:00:00Z",
        license_status="open",
        extraction_method="api",
        payload="some content",
    )
    assert p.content_hash == content_hash("some content")
    assert p.reviewer_status == "unreviewed"


@pytest.mark.unit
def test_is_citable_requires_a_citation_field() -> None:
    base = dict(
        source_id="s",
        source_url="https://x",
        retrieved_at="2026-05-28T00:00:00Z",
        license_status="open",
        extraction_method="api",
        content_hash=content_hash("x"),
    )
    assert Provenance(**base).is_citable is False
    assert Provenance(**base, citation_text="FDA label X").is_citable is True
    assert Provenance(**base, citation_span="set_id=Y sec=DDI").is_citable is True


@pytest.mark.unit
def test_usable_for_claims_requires_review_and_citation() -> None:
    base = dict(
        source_id="s",
        source_url="https://x",
        retrieved_at="2026-05-28T00:00:00Z",
        license_status="open",
        extraction_method="api",
        content_hash=content_hash("x"),
    )
    # citation but unreviewed -> not usable
    p1 = Provenance(**base, citation_text="cite", reviewer_status="unreviewed")
    assert p1.is_usable_for_claims is False
    # reviewed but no citation -> not usable
    p2 = Provenance(**base, reviewer_status="human_reviewed")
    assert p2.is_usable_for_claims is False
    # reviewed AND cited -> usable
    p3 = Provenance(**base, citation_text="cite", reviewer_status="human_reviewed")
    assert p3.is_usable_for_claims is True


@pytest.mark.unit
def test_invalid_content_hash_raises() -> None:
    with pytest.raises(ValueError, match="content_hash"):
        Provenance(
            source_id="s",
            source_url="https://x",
            retrieved_at="2026-05-28T00:00:00Z",
            license_status="open",
            extraction_method="api",
            content_hash="not-a-hash",
        )


@pytest.mark.unit
def test_required_fields_enforced() -> None:
    with pytest.raises(ValueError):
        Provenance(
            source_id="",  # empty -> invalid
            source_url="https://x",
            retrieved_at="2026-05-28T00:00:00Z",
            license_status="open",
            extraction_method="api",
            content_hash=content_hash("x"),
        )
