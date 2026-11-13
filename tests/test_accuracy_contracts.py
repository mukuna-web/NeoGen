"""Tests for accuracy contracts + release-gate derivation."""

from __future__ import annotations

import pytest

from neogen.accuracy.contracts import (
    AccuracyFinding,
    gate_from_findings,
    worst_severity,
)


def _f(sev="info", status="pass", expl="ok"):
    return AccuracyFinding(check_id="c", check_type="t", severity=sev,
                           status=status, explanation=expl)


@pytest.mark.unit
def test_invalid_severity_rejected():
    with pytest.raises(ValueError, match="severity"):
        _f(sev="apocalyptic")


@pytest.mark.unit
def test_invalid_status_rejected():
    with pytest.raises(ValueError, match="status"):
        _f(status="maybe")


@pytest.mark.unit
def test_empty_explanation_rejected():
    with pytest.raises(ValueError, match="explanation"):
        _f(expl="   ")


@pytest.mark.unit
def test_worst_severity():
    assert worst_severity([]) == "info"
    assert worst_severity(["low", "high", "medium"]) == "high"
    assert worst_severity(["critical", "high"]) == "critical"


@pytest.mark.unit
def test_gate_fails_on_critical():
    gate = gate_from_findings([_f("critical", "fail", "boom")])
    assert gate.status == "FAIL"
    assert gate.critical == 1


@pytest.mark.unit
def test_gate_fails_on_high():
    gate = gate_from_findings([_f("high", "fail", "bad")])
    assert gate.status == "FAIL"


@pytest.mark.unit
def test_gate_blocked_on_blocked_finding():
    gate = gate_from_findings([_f("high", "blocked", "missing input")])
    assert gate.status == "BLOCKED"


@pytest.mark.unit
def test_gate_warns_on_medium():
    gate = gate_from_findings([_f("medium", "warning", "review")])
    assert gate.status == "WARNING"


@pytest.mark.unit
def test_gate_passes_only_when_critical_high_absent():
    # low/info notes do not block PASS
    gate = gate_from_findings([_f("low", "warning", "note"), _f("info", "pass", "ok")])
    assert gate.status == "PASS"
    # any high finding prevents PASS
    assert gate_from_findings([_f("high", "warning", "x")]).status == "FAIL"


@pytest.mark.unit
def test_passing_findings_are_not_problems():
    gate = gate_from_findings([_f("critical", "pass", "satisfied")])
    assert gate.status == "PASS"  # a pass-status finding is not a problem
