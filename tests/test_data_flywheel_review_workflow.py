"""Tests for the append-only reviewer governance workflow."""

from __future__ import annotations

import json

import pytest

from neogen.data_flywheel.review_workflow import (
    ReviewError,
    append_event,
    promote,
)

_CITED = {"citation_span": "FDA label X", "severity": None, "content_hash": "a" * 64}
_UNCITED = {"citation_span": "", "severity": None, "content_hash": "b" * 64}
_SEVERITY = {"citation_span": "FDA label Y", "severity": "moderate", "content_hash": "c" * 64}


@pytest.mark.unit
def test_cited_record_can_be_machine_checked() -> None:
    ev = promote(_CITED, record_id="r1", from_status="needs_review",
                 to_status="machine_checked", reviewer_id="machine:flywheel",
                 reason="cited + severity-null", timestamp="t")
    assert ev.to_status == "machine_checked"
    assert ev.previous_hash == "a" * 64 and len(ev.new_hash) == 64


@pytest.mark.unit
def test_uncited_record_cannot_advance() -> None:
    with pytest.raises(ReviewError, match="uncited"):
        promote(_UNCITED, record_id="r2", from_status="needs_review",
                to_status="machine_checked", reviewer_id="machine:flywheel",
                reason="x", timestamp="t")


@pytest.mark.unit
def test_severity_record_cannot_be_machine_checked() -> None:
    with pytest.raises(ReviewError, match="human_reviewed"):
        promote(_SEVERITY, record_id="r3", from_status="needs_review",
                to_status="machine_checked", reviewer_id="machine:flywheel",
                reason="x", timestamp="t")


@pytest.mark.unit
def test_machine_cannot_perform_human_review() -> None:
    with pytest.raises(ReviewError, match="human reviewer"):
        promote(_CITED, record_id="r4", from_status="needs_review",
                to_status="human_reviewed", reviewer_id="machine:flywheel",
                reason="x", timestamp="t")


@pytest.mark.unit
def test_human_review_with_human_id_ok() -> None:
    ev = promote(_CITED, record_id="r5", from_status="machine_checked",
                 to_status="human_reviewed", reviewer_id="reviewer:mukund",
                 reason="checked label span", timestamp="t")
    assert ev.to_status == "human_reviewed"


@pytest.mark.unit
def test_illegal_transition_rejected() -> None:
    with pytest.raises(ReviewError, match="not allowed"):
        promote(_CITED, record_id="r6", from_status="rejected",
                to_status="human_reviewed", reviewer_id="reviewer:x",
                reason="x", timestamp="t")


@pytest.mark.unit
def test_append_event_is_append_only(tmp_path) -> None:
    p = tmp_path / "events.jsonl"
    e1 = promote(_CITED, record_id="r1", from_status="needs_review",
                 to_status="machine_checked", reviewer_id="machine:f", reason="x", timestamp="t")
    e2 = promote(_CITED, record_id="r1", from_status="machine_checked",
                 to_status="human_reviewed", reviewer_id="reviewer:m", reason="y", timestamp="t2")
    append_event(e1, p)
    append_event(e2, p)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["to_status"] == "machine_checked"
    assert json.loads(lines[1])["to_status"] == "human_reviewed"
