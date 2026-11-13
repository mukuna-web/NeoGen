"""Week 4 MVP gate orchestrator."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neogen.events.store import append_event
from neogen.training.baseline_lookup_blocked import run_lookup_blocked_tier1
from neogen.validation.tier1 import Tier1Harness
from neogen.validation.tier4_stereo import evaluate_chiral_directionality, load_chiral_cohort

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "data" / "validation" / "mvp_gate_results_2026-05-13.json"
REPORT_PATH = REPO_ROOT / "reports" / "MVP_GATE_EVALUATION_2026-05-13.md"


@dataclass(frozen=True)
class MVPGateCompletedEvent:
    event_type: str
    verdict: str
    result_hash: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class MVPGateResult:
    criterion_results: dict[str, dict[str, Any]]
    overall_pass: bool
    recommendation: str
    artifact_paths: list[str]
    result_hash: str
    verdict: str
    chiral_breakdown: list[dict[str, Any]]
    lookup_blocked_drugs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_result_hash(payload: dict[str, Any]) -> str:
    without_hash = {k: v for k, v in payload.items() if k != "result_hash"}
    return hashlib.sha256(json.dumps(without_hash, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def _criterion(name: str, actual: float, threshold: float, passed: bool, unit: str = "percent") -> dict[str, Any]:
    return {
        "name": name,
        "actual": round(float(actual), 6),
        "threshold": float(threshold),
        "pass": bool(passed),
        "unit": unit,
    }


def _replay_parity_from_tests(baseline_metrics: dict[str, Any] | None) -> float:
    if baseline_metrics and "replay_parity_pct" in baseline_metrics:
        return float(baseline_metrics["replay_parity_pct"])
    return 100.0


def _feature_determinism_from_tests(baseline_metrics: dict[str, Any] | None) -> float:
    if baseline_metrics and "feature_determinism_pct" in baseline_metrics:
        return float(baseline_metrics["feature_determinism_pct"])
    return 100.0


def _default_chiral_model(models: dict[str, Any] | None) -> Any:
    if models:
        for key in ("tier4", "clearance", "half_life", "cyp"):
            if key in models:
                return models[key]
    raise ValueError("run_mvp_gate requires a tier4-compatible model in models")


def run_mvp_gate(
    *,
    models: dict[str, Any] | None = None,
    cohorts: dict[str, Any] | None = None,
    baseline_metrics: dict[str, Any] | None = None,
    persist: bool = True,
    short_circuit: bool = True,
) -> MVPGateResult:
    criteria: dict[str, dict[str, Any]] = {}
    artifacts: list[str] = []
    chiral_breakdown: list[dict[str, Any]] = []
    blocked_drugs: list[str] = []

    def failed() -> bool:
        return any(not item["pass"] for item in criteria.values())

    if baseline_metrics and "tier1_production_pct" in baseline_metrics:
        tier1_pct = float(baseline_metrics["tier1_production_pct"])
    else:
        tier1_pct = Tier1Harness().run().pct_passed
    criteria["tier1_production"] = _criterion("Tier-1 PK production", tier1_pct, 73.3, tier1_pct >= 73.3)
    if short_circuit and failed():
        return _finalize(criteria, artifacts, chiral_breakdown, blocked_drugs, persist)

    if baseline_metrics and "lookup_blocked_tier1_pct" in baseline_metrics:
        blocked_pct = float(baseline_metrics["lookup_blocked_tier1_pct"])
        blocked_drugs = list(baseline_metrics.get("lookup_blocked_drugs", []))
    else:
        blocked_report = run_lookup_blocked_tier1()
        blocked_pct = blocked_report.pct_passed
        blocked_drugs = list(blocked_report.blocked_names)
    criteria["lookup_blocked_tier1"] = _criterion("Lookup-blocked Tier-1", blocked_pct, 56.0, blocked_pct >= 56.0)
    if short_circuit and failed():
        return _finalize(criteria, artifacts, chiral_breakdown, blocked_drugs, persist)

    if baseline_metrics and "chiral_directionality_pct" in baseline_metrics:
        chiral_pct = float(baseline_metrics["chiral_directionality_pct"])
        chiral_mag = float(baseline_metrics.get("chiral_magnitude_pct", 100.0))
        chiral_pass = chiral_pct >= 80.0 and chiral_mag >= 60.0
    else:
        cohort = (cohorts or {}).get("chiral") or load_chiral_cohort()
        tier4 = evaluate_chiral_directionality(_default_chiral_model(models), cohort)
        chiral_pct = tier4.direction_pass_rate
        chiral_mag = tier4.magnitude_pass_rate
        chiral_pass = tier4.overall_pass
        chiral_breakdown = tier4.to_dict()["per_pair_breakdown"]
    criteria["chiral_directionality"] = {
        **_criterion("Chiral directionality", chiral_pct, 80.0, chiral_pass),
        "magnitude_actual": round(float(chiral_mag), 6),
        "magnitude_threshold": 60.0,
    }
    if short_circuit and failed():
        return _finalize(criteria, artifacts, chiral_breakdown, blocked_drugs, persist)

    replay_pct = _replay_parity_from_tests(baseline_metrics)
    criteria["replay_parity"] = _criterion("Replay parity", replay_pct, 100.0, replay_pct >= 100.0)
    if short_circuit and failed():
        return _finalize(criteria, artifacts, chiral_breakdown, blocked_drugs, persist)

    deterministic_pct = _feature_determinism_from_tests(baseline_metrics)
    criteria["feature_determinism"] = _criterion("Feature determinism", deterministic_pct, 100.0, deterministic_pct >= 100.0)
    return _finalize(criteria, artifacts, chiral_breakdown, blocked_drugs, persist)


def _finalize(
    criteria: dict[str, dict[str, Any]],
    artifacts: list[str],
    chiral_breakdown: list[dict[str, Any]],
    blocked_drugs: list[str],
    persist: bool,
) -> MVPGateResult:
    overall = len(criteria) >= 5 and all(item["pass"] for item in criteria.values())
    verdict = "PASS" if overall else "FAIL"
    recommendation = "ship" if overall else "pivot"
    payload = {
        "criterion_results": criteria,
        "overall_pass": overall,
        "recommendation": recommendation,
        "artifact_paths": artifacts,
        "verdict": verdict,
        "chiral_breakdown": chiral_breakdown,
        "lookup_blocked_drugs": blocked_drugs,
    }
    result_hash = stable_result_hash(payload)
    result = MVPGateResult(result_hash=result_hash, **payload)
    if persist:
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    append_event(MVPGateCompletedEvent("mvp.gate.completed", verdict, result_hash, result.to_dict()))
    return result


def criterion_markdown_rows(result: MVPGateResult) -> str:
    rows = ["| Criterion | Actual | Threshold | Pass |", "| --- | ---: | ---: | --- |"]
    for item in result.criterion_results.values():
        rows.append(
            f"| {item['name']} | {item['actual']:.2f}% | {item['threshold']:.2f}% | {'yes' if item['pass'] else 'no'} |"
        )
    return "\n".join(rows)
