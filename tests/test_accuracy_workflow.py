"""Tests for the Opus-class workflow orchestration + release gate wiring."""

from __future__ import annotations

import pytest

from neogen.accuracy.contracts import GATES
from neogen.accuracy.workflow import run_workflow


@pytest.fixture(scope="module")
def result():
    # One real audit over the repo (no test results injected).
    return run_workflow("ddi-flywheel")


@pytest.mark.unit
def test_all_eight_roles_present(result):
    for role in ("planner", "code_auditor", "biological_auditor", "evidence_auditor",
                 "hallucination_auditor", "replay_runner", "ui_auditor",
                 "validation_runner", "final_arbiter"):
        assert role in result.role_summaries


@pytest.mark.unit
def test_scorecards_have_valid_status(result):
    titles = {c.title for c in result.run.checks}
    assert {"Code Sweep", "Biological Benchmark", "Evidence Provenance",
            "Hallucination Risk", "Replay Determinism", "VORTEXT / API Integrity",
            "Test Suite"} <= titles
    for c in result.run.checks:
        assert c.status in ("pass", "fail", "warning", "blocked", "not_applicable")
        assert 0 <= c.score <= 100


@pytest.mark.unit
def test_biological_matrix_present(result):
    dims = {d.dimension for d in result.biological_dimensions}
    assert "cyp_directionality" in dims and "marker_reaction_excluded" in dims


@pytest.mark.unit
def test_release_gate_valid(result):
    g = result.run.release_gate
    assert g.status in GATES
    # invariant: PASS iff there are no critical/high/medium problems and nothing
    # blocked. Low/info notes never block PASS.
    no_blockers = g.critical == 0 and g.high == 0 and g.medium == 0 and g.blocked == 0
    assert (g.status == "PASS") == no_blockers
    # a low-only finding set must never produce FAIL/BLOCKED
    if g.critical == 0 and g.high == 0 and g.medium == 0:
        assert g.status not in ("FAIL", "BLOCKED")


@pytest.mark.unit
def test_no_medical_claim_note(result):
    assert "not clinical validation" in result.run.note.lower()
    assert "no severity" in result.run.note.lower()


@pytest.mark.unit
def test_failing_tests_fail_the_gate():
    res = run_workflow("ddi-flywheel", test_results={
        "passed": 1, "failed": 3, "errors": 0, "summary": "3 failed"})
    gate = res.run.release_gate
    val = next(c for c in res.run.checks if c.title == "Test Suite")
    assert val.status == "fail"
    assert gate.status == "FAIL"  # a high validation finding fails the gate
