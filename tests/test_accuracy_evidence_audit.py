"""Tests for the evidence/provenance audit (offline tmp fixtures)."""

from __future__ import annotations

import json

import pytest

from neogen.accuracy.evidence_audit import (
    _audit_review_events,
    run_evidence_audit,
)


def _write(norm, evidence, pair_records=None):
    norm.mkdir(parents=True, exist_ok=True)
    (norm / "openfda_ddi_label_examples.json").write_text(json.dumps({
        "evidence": evidence, "pair_records": pair_records or [],
    }))


@pytest.mark.unit
def test_missing_source_url_flagged(tmp_path):
    norm = tmp_path / "norm"
    _write(norm, [{"drug_a": "a", "citation_span": "x", "retrieved_at": "t"}])
    res = run_evidence_audit(norm=norm, review_dir=tmp_path / "rev")
    findings = [f for r in res for f in r.findings if f.is_problem]
    assert any("source_url" in f.check_id for f in findings)


@pytest.mark.unit
def test_uncited_severity_flagged_high(tmp_path):
    norm = tmp_path / "norm"
    _write(norm, [{
        "drug_a": "a", "drug_b": "b", "source_url": "http://x", "retrieved_at": "t",
        "citation_span": "", "severity": "major",  # severity with no citation
    }])
    res = run_evidence_audit(norm=norm, review_dir=tmp_path / "rev")
    hi = [f for r in res for f in r.findings if f.severity == "high" and "uncited_severity" in f.check_id]
    assert hi, "uncited severity must be a high finding"


@pytest.mark.unit
def test_complete_record_clean(tmp_path):
    norm = tmp_path / "norm"
    _write(norm, [{
        "drug_a": "a", "drug_b": "b", "source_url": "http://x", "retrieved_at": "t",
        "citation_span": "quoted span", "severity": None,
    }])
    res = run_evidence_audit(norm=norm, review_dir=tmp_path / "rev")
    ev = next(r for r in res if r.artifact.endswith(":evidence"))
    assert ev.incomplete == 0 and ev.complete == 1


@pytest.mark.unit
def test_pair_record_not_flagged_for_missing_source_url(tmp_path):
    # pair records carry provenance via evidence_refs, not their own url/ts
    norm = tmp_path / "norm"
    _write(norm, [], pair_records=[
        {"pair_id": "a|b", "severity": None, "evidence_refs": ["e1"]},
    ])
    res = run_evidence_audit(norm=norm, review_dir=tmp_path / "rev")
    pr = next(r for r in res if r.artifact.endswith(":pair_records"))
    assert pr.incomplete == 0


@pytest.mark.unit
def test_fake_human_review_flagged_critical(tmp_path):
    rev = tmp_path / "rev"
    rev.mkdir(parents=True)
    (rev / "review_events_2026.jsonl").write_text(json.dumps({
        "reviewer_id": "machine:auditor", "to_status": "human_reviewed",
        "reason": "looks fine", "timestamp": "2026-05-30T00:00:00Z",
    }) + "\n")
    res = _audit_review_events(rev)
    crit = [f for f in res.findings if f.severity == "critical"]
    assert crit and "fake_human" in crit[0].check_id


@pytest.mark.unit
def test_machine_checked_review_is_fine(tmp_path):
    rev = tmp_path / "rev"
    rev.mkdir(parents=True)
    (rev / "ev.jsonl").write_text(json.dumps({
        "reviewer_id": "machine:auditor", "to_status": "machine_checked",
        "reason": "cited + severity-null", "timestamp": "2026-05-30T00:00:00Z",
    }) + "\n")
    res = _audit_review_events(rev)
    assert not [f for f in res.findings if f.is_problem]
