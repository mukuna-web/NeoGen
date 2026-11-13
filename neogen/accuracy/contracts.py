"""Typed contracts for the Accuracy Lab.

Frozen dataclasses + small helpers shared by every audit module and the
workflow. Severity/status are closed string sets; the release gate is derived
deterministically from the set of findings.

Severity:  info | low | medium | high | critical
Status:    pass | fail | warning | blocked | not_applicable
Gate:      PASS | WARNING | FAIL | BLOCKED
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SEVERITIES = ("info", "low", "medium", "high", "critical")
STATUSES = ("pass", "fail", "warning", "blocked", "not_applicable")
GATES = ("PASS", "WARNING", "FAIL", "BLOCKED")

# Higher = worse. Used to roll a finding set up to a single severity/gate.
_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def worst_severity(severities: list[str]) -> str:
    """Return the most severe of a list of severities (``info`` if empty)."""
    if not severities:
        return "info"
    return max(severities, key=lambda s: _SEVERITY_RANK.get(s, 0))


@dataclass(frozen=True)
class AccuracyFinding:
    """One atomic issue (or confirmation) produced by an audit.

    A finding with ``status="pass"`` records a satisfied check; ``fail`` /
    ``warning`` / ``blocked`` record problems that feed the release gate.
    """

    check_id: str
    check_type: str                  # code_sweep | evidence | biological | replay | ui | ...
    severity: str                    # info | low | medium | high | critical
    status: str                      # pass | fail | warning | blocked | not_applicable
    explanation: str
    file_path: str | None = None
    line: int | None = None
    source_url: str | None = None
    citation: str | None = None
    evidence_hash: str | None = None
    actual: str | None = None
    expected: str | None = None
    remediation: str | None = None
    reviewer_status: str = "needs_review"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"invalid severity {self.severity!r}")
        if self.status not in STATUSES:
            raise ValueError(f"invalid status {self.status!r}")
        if not (self.explanation and self.explanation.strip()):
            raise ValueError("AccuracyFinding requires an explanation")

    @property
    def is_problem(self) -> bool:
        return self.status in ("fail", "warning", "blocked")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HallucinationFinding:
    """A risky/unsupported-claim match from the code/report sweep."""

    pattern: str                     # the risky phrase class, e.g. "fda approved"
    matched_text: str
    file_path: str
    line: int
    severity: str
    allowed: bool                    # True if in quoted source / disclaimer / cited
    reason: str
    reviewer_status: str = "needs_review"

    def to_finding(self) -> AccuracyFinding:
        return AccuracyFinding(
            check_id=f"halluc::{self.pattern.replace(' ', '_')}",
            check_type="hallucination",
            severity=self.severity,
            status="pass" if self.allowed else "fail",
            explanation=(
                f"Risky phrase {self.pattern!r} — {self.reason}"
            ),
            file_path=self.file_path,
            line=self.line,
            actual=self.matched_text[:200],
            remediation=(
                "Allowed (quoted source / disclaimer / cited)."
                if self.allowed else
                "Remove the claim, attribute it to a cited source span, or mark "
                "the record needs_review. Do not assert it as NeoGen's own claim."
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CodeSweepResult:
    files_scanned: int
    risky_matches: int
    flagged: int                     # not-allowed matches -> findings
    allowed: int                     # quoted/disclaimer/cited matches
    findings: tuple[AccuracyFinding, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d


@dataclass(frozen=True)
class EvidenceAuditResult:
    artifact: str
    records_checked: int
    complete: int                    # records with full provenance
    incomplete: int
    findings: tuple[AccuracyFinding, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d


@dataclass(frozen=True)
class BiologicalBenchmarkResult:
    dimension: str                   # cyp_directionality | transporter_directionality | ...
    checked: int
    passed: int
    failed: int
    status: str                      # pass | warning | fail | not_applicable
    details: str
    findings: tuple[AccuracyFinding, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d


@dataclass(frozen=True)
class AccuracyCheck:
    """A named check (group of findings) shown as one scorecard row."""

    check_id: str
    check_type: str
    title: str
    status: str                      # pass | fail | warning | blocked | not_applicable
    severity: str                    # worst severity among its findings
    score: float                     # 0..100 research-grade score for this card
    summary: str
    findings: tuple[AccuracyFinding, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d


@dataclass(frozen=True)
class ReleaseGateResult:
    status: str                      # PASS | WARNING | FAIL | BLOCKED
    critical: int
    high: int
    medium: int
    low: int
    blocked: int
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AccuracyRun:
    run_id: str
    scope: str
    started_at: str
    completed_at: str
    checks: tuple[AccuracyCheck, ...]
    release_gate: ReleaseGateResult
    overall_score: float
    finding_counts: dict
    test_results: dict = field(default_factory=dict)
    note: str = (
        "Research/prototype accuracy only. NOT clinical validation, NOT FDA "
        "approval/acceptance/clearance, NOT medical advice. No severity asserted."
    )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "scope": self.scope,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "overall_score": self.overall_score,
            "release_gate": self.release_gate.to_dict(),
            "finding_counts": self.finding_counts,
            "test_results": self.test_results,
            "checks": [c.to_dict() for c in self.checks],
            "note": self.note,
        }


def gate_from_findings(findings: list[AccuracyFinding]) -> ReleaseGateResult:
    """Derive the release gate from the full finding set.

    PASS requires zero critical and zero high *problems* (fail/warning/blocked).
    A critical or high problem fails the gate; a medium problem warns; blocked
    inputs block. Low/info notes do not block PASS.
    """
    problems = [f for f in findings if f.is_problem]
    by_sev = {s: sum(1 for f in problems if f.severity == s) for s in SEVERITIES}
    blocked = sum(1 for f in problems if f.status == "blocked")
    reasons: list[str] = []

    if blocked:
        status = "BLOCKED"
        reasons.append(f"{blocked} blocked check(s) — required inputs missing/unverified")
    elif by_sev["critical"]:
        status = "FAIL"
        reasons.append(f"{by_sev['critical']} critical finding(s)")
    elif by_sev["high"]:
        status = "FAIL"
        reasons.append(f"{by_sev['high']} high-severity finding(s)")
    elif by_sev["medium"]:
        status = "WARNING"
        reasons.append(f"{by_sev['medium']} medium-severity finding(s)")
    else:
        status = "PASS"
        reasons.append("No critical/high/medium problems; low/info notes only")

    return ReleaseGateResult(
        status=status,
        critical=by_sev["critical"], high=by_sev["high"],
        medium=by_sev["medium"], low=by_sev["low"], blocked=blocked,
        reasons=tuple(reasons),
    )
