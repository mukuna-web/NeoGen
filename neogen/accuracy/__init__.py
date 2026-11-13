"""NeoGen Accuracy Lab — research/prototype validation, not clinical validation.

This package scores NeoGen's internal consistency, evidence provenance,
biological-rule sanity, hallucination risk, and determinism. It produces a
research-grade accuracy report and a red/amber/green release gate.

It makes **no** claim of FDA approval, clinical validity, or medical advice.
Every assertion it checks must trace to code, source data, or a test fixture;
anything that cannot be verified is marked ``unverified`` / ``blocked`` /
``needs_review`` rather than asserted.
"""

from neogen.accuracy.contracts import (
    AccuracyCheck,
    AccuracyFinding,
    AccuracyRun,
    BiologicalBenchmarkResult,
    CodeSweepResult,
    EvidenceAuditResult,
    HallucinationFinding,
    ReleaseGateResult,
    gate_from_findings,
    worst_severity,
)

__all__ = [
    "AccuracyCheck",
    "AccuracyFinding",
    "AccuracyRun",
    "BiologicalBenchmarkResult",
    "CodeSweepResult",
    "EvidenceAuditResult",
    "HallucinationFinding",
    "ReleaseGateResult",
    "gate_from_findings",
    "worst_severity",
]
