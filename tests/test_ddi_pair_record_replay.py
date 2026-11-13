"""Replay + determinism tests for DDIPairRecord v0.

Covers V35 backlog items #2, #3, #5:
- Deterministic content_hash (same input → same hex digest, twice).
- Replay sensitivity (tampering any identity field changes the hash).
- V35 §1 invariant: severity is None unless evidence_refs is non-empty.
- pair_id is canonical (sorted, pipe-joined drug ids).
- feature_schema_version is restricted to the accepted set.

All fixture data is illustrative and source-free; per V35 every record has
``severity=None`` until a source-backed cycle adds evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neogen.engine.ddi.pair_record import (
    ACCEPTED_FEATURE_SCHEMAS,
    DDIPairRecord,
    EvidenceRef,
    from_dict,
    make_pair_id,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "ddi_pair_fixture_v0.json"
)


# ── fixture-shape sanity ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_fixture_is_valid_json_list() -> None:
    data = json.loads(FIXTURE.read_text())
    assert isinstance(data, list)
    assert len(data) == 3
    assert all(isinstance(item, dict) for item in data)


@pytest.mark.unit
def test_fixture_loads_into_records() -> None:
    data = json.loads(FIXTURE.read_text())
    records = [from_dict(d) for d in data]
    assert len(records) == 3
    assert all(isinstance(r, DDIPairRecord) for r in records)


# ── canonical pair_id + accepted schemas ─────────────────────────────────────


@pytest.mark.unit
def test_pair_ids_are_canonical() -> None:
    data = json.loads(FIXTURE.read_text())
    for d in data:
        expected = make_pair_id(d["drug_a_id"], d["drug_b_id"])
        assert d["pair_id"] == expected, (
            f"fixture pair_id {d['pair_id']!r} != canonical {expected!r}"
        )


@pytest.mark.unit
def test_feature_schema_versions_are_accepted() -> None:
    data = json.loads(FIXTURE.read_text())
    for d in data:
        assert d["feature_schema_version"] in ACCEPTED_FEATURE_SCHEMAS


# ── V35 §1 severity-evidence invariant ───────────────────────────────────────


@pytest.mark.unit
def test_fixture_records_all_have_null_severity() -> None:
    """No record may declare severity without evidence (V35 §1)."""
    data = json.loads(FIXTURE.read_text())
    for d in data:
        assert d["severity"] is None
        assert d["evidence_refs"] == []


@pytest.mark.unit
def test_severity_without_evidence_raises() -> None:
    """Constructing a record with severity but no evidence must fail."""
    with pytest.raises(ValueError, match="no clinical claims without citations"):
        DDIPairRecord(
            pair_id=make_pair_id("a", "b"),
            drug_a_id="a",
            drug_b_id="b",
            drug_a_name="A",
            drug_b_name="B",
            role="perpetrator",
            mechanism_category="cyp_inhibition",
            severity="major",  # ← forbidden without evidence
        )


@pytest.mark.unit
def test_severity_with_evidence_is_allowed() -> None:
    """A record with at least one EvidenceRef may carry severity."""
    rec = DDIPairRecord(
        pair_id=make_pair_id("a", "b"),
        drug_a_id="a",
        drug_b_id="b",
        drug_a_name="A",
        drug_b_name="B",
        role="perpetrator",
        mechanism_category="cyp_inhibition",
        evidence_refs=(
            EvidenceRef(
                evidence_type="hypothetical_illustrative",
                citation="illustrative-source-001",
            ),
        ),
        severity="moderate",
    )
    assert rec.severity == "moderate"
    assert len(rec.evidence_refs) == 1


# ── determinism / replay ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_content_hash_is_stable_across_reloads() -> None:
    """Loading the same fixture twice must yield identical content hashes."""
    data = json.loads(FIXTURE.read_text())
    hashes_a = [from_dict(d).content_hash() for d in data]
    hashes_b = [from_dict(d).content_hash() for d in data]
    assert hashes_a == hashes_b


@pytest.mark.unit
def test_content_hash_is_deterministic_hex_format() -> None:
    data = json.loads(FIXTURE.read_text())
    for d in data:
        h = from_dict(d).content_hash()
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


@pytest.mark.unit
def test_content_hashes_are_pair_unique() -> None:
    data = json.loads(FIXTURE.read_text())
    hashes = [from_dict(d).content_hash() for d in data]
    assert len(set(hashes)) == len(hashes), "hashes must be pair-unique"


@pytest.mark.unit
def test_content_hash_changes_when_drug_id_changes() -> None:
    data = json.loads(FIXTURE.read_text())
    base = from_dict(data[0])
    perturbed_dict = dict(data[0])
    perturbed_dict["drug_a_id"] = base.drug_a_id + "-tampered"
    perturbed_dict["pair_id"] = make_pair_id(
        perturbed_dict["drug_a_id"], perturbed_dict["drug_b_id"]
    )
    perturbed = from_dict(perturbed_dict)
    assert base.content_hash() != perturbed.content_hash()


# ── pair_id canonical enforcement at construction ────────────────────────────


@pytest.mark.unit
def test_pair_id_must_match_canonical_order() -> None:
    with pytest.raises(ValueError, match="canonical"):
        DDIPairRecord(
            pair_id="b|a",  # wrong order — canonical would be "a|b"
            drug_a_id="a",
            drug_b_id="b",
            drug_a_name="A",
            drug_b_name="B",
            role="perpetrator",
            mechanism_category="cyp_inhibition",
        )


@pytest.mark.unit
def test_make_pair_id_sorts_ids() -> None:
    assert make_pair_id("b", "a") == "a|b"
    assert make_pair_id("a", "b") == "a|b"


@pytest.mark.unit
def test_unknown_feature_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="feature_schema_version"):
        DDIPairRecord(
            pair_id=make_pair_id("a", "b"),
            drug_a_id="a",
            drug_b_id="b",
            drug_a_name="A",
            drug_b_name="B",
            role="perpetrator",
            mechanism_category="cyp_inhibition",
            feature_schema_version="some_future_schema",  # not in accepted set
        )
