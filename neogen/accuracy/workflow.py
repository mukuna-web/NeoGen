"""Opus-class adversarial workflow — structured stages with role summaries.

Even though Claude subagents are not invoked here, the audit is organised as the
roles the mission specifies (Planner, Code Auditor, Biological Auditor, Evidence
Auditor, Hallucination Auditor, UI Auditor, Validation Runner, Final Arbiter).
Each role runs its audit module, contributes findings, and emits a short
machine-readable + markdown summary. The Final Arbiter derives the release gate.

Pure/in-memory: ``run_workflow`` does no disk I/O — the CLI persists the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from neogen.accuracy import biological_benchmark as bio
from neogen.accuracy import code_sweep, evidence_audit, replay_audit
from neogen.accuracy.contracts import (
    AccuracyCheck,
    AccuracyFinding,
    AccuracyRun,
    gate_from_findings,
    worst_severity,
)

_ROOT = Path(__file__).resolve().parents[2]
# critical/high/medium drive the score; low/info are informational "notes" whose
# total impact per card is capped so a pile of review notes can't tank a card
# that has zero real problems (the gate already treats low/info as non-blocking).
_SEV_PENALTY = {"critical": 45, "high": 25, "medium": 12}
_LOW_PENALTY_CAP = 10

STAGES = (
    "plan", "code_sweep", "biological_audit", "evidence_audit",
    "hallucination_audit", "replay_audit", "ui_audit", "validation", "final_gate",
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _status_from(findings: list[AccuracyFinding], checked: int) -> str:
    probs = [f for f in findings if f.is_problem]
    if any(f.status == "blocked" for f in probs):
        return "blocked"
    if any(f.status == "fail" for f in probs):
        return "fail"
    if any(f.status == "warning" for f in probs):
        return "warning"
    return "not_applicable" if checked == 0 else "pass"


def _score(findings: list[AccuracyFinding]) -> float:
    probs = [f for f in findings if f.is_problem]
    penalty = sum(_SEV_PENALTY.get(f.severity, 0) for f in probs)
    low_notes = sum(1 for f in probs if f.severity in ("low", "info"))
    penalty += min(low_notes, _LOW_PENALTY_CAP)
    return float(max(0, min(100, 100 - penalty)))


def _check(check_id, ctype, title, findings, checked, summary) -> AccuracyCheck:
    probs = [f for f in findings if f.is_problem]
    return AccuracyCheck(
        check_id=check_id, check_type=ctype, title=title,
        status=_status_from(findings, checked),
        severity=worst_severity([f.severity for f in probs]),
        score=_score(findings), summary=summary, findings=tuple(findings),
    )


@dataclass
class WorkflowResult:
    run: AccuracyRun
    biological_dimensions: list = field(default_factory=list)
    evidence_results: list = field(default_factory=list)
    hallucination: object = None
    code_sweep: object = None
    role_summaries: dict = field(default_factory=dict)
    all_findings: list = field(default_factory=list)


def _ui_audit(root: Path) -> list[AccuracyFinding]:
    findings: list[AccuracyFinding] = []
    routes = root / "neogen" / "agent_runtime" / "routes_vortext.py"
    txt = routes.read_text() if routes.exists() else ""
    # routes use the router prefix + @router.get("/accuracy/...") decorators
    if '"/accuracy"' in txt and '"/accuracy/release-gate"' in txt:
        findings.append(AccuracyFinding(
            check_id="ui::endpoints", check_type="ui", severity="info", status="pass",
            explanation="accuracy API endpoints present in routes_vortext.py"))
    else:
        findings.append(AccuracyFinding(
            check_id="ui::endpoints", check_type="ui", severity="medium", status="warning",
            explanation="accuracy API endpoints not yet wired in routes_vortext.py",
            remediation="Add read-only /api/vortext/accuracy[/latest|/findings|/release-gate]."))
    comp = root / "vortext" / "terminal" / "components" / "vortext"
    needed = ["accuracy-lab-pane.tsx", "accuracy-scorecard.tsx",
              "hallucination-findings-table.tsx", "biological-benchmark-matrix.tsx",
              "release-gate-panel.tsx"]
    missing = [n for n in needed if not (comp / n).exists()]
    if missing:
        findings.append(AccuracyFinding(
            check_id="ui::components", check_type="ui", severity="low", status="warning",
            explanation=f"Accuracy Lab components not yet present: {missing}",
            remediation="Build the listed terminal components."))
    else:
        pane = (comp / "accuracy-lab-pane.tsx").read_text()
        banner_ok = "medical" in pane.lower() and (
            "no " in pane.lower() or "not " in pane.lower() or "research" in pane.lower())
        findings.append(AccuracyFinding(
            check_id="ui::components", check_type="ui", severity="info", status="pass",
            explanation="all Accuracy Lab terminal components present"))
        findings.append(AccuracyFinding(
            check_id="ui::safety_banner", check_type="ui",
            severity="info" if banner_ok else "low",
            status="pass" if banner_ok else "warning",
            explanation="no-medical-claim safety banner present in Accuracy Lab pane"
            if banner_ok else "Accuracy Lab pane is missing a no-medical-claim safety banner",
            file_path=str(comp / "accuracy-lab-pane.tsx")))
    return findings


def run_workflow(
    scope: str = "ddi-flywheel", *, root: Path = _ROOT,
    test_results: dict | None = None,
) -> WorkflowResult:
    started = _utc()
    all_findings: list[AccuracyFinding] = []
    checks: list[AccuracyCheck] = []
    role: dict[str, dict] = {}

    # 1. plan
    role["planner"] = {
        "stage": "plan", "scope": scope,
        "summary": "Audit NeoGen DDI/flywheel for code integrity, provenance, "
                   "biological-rule sanity, hallucination risk, and determinism. "
                   "Stages: code_sweep, biological, evidence, hallucination, replay, "
                   "ui, validation, gate. Research-grade only; no clinical claim.",
    }

    # 2. code_sweep — authored source code. Exclude the detector package itself,
    # tests (fixtures intentionally contain risky phrases), and machine-generated
    # candidate scaffolds (rd_candidates / swarm_lab) which are not authored claims.
    cs = code_sweep.scan_paths(
        [root / "neogen"],
        exclude=("neogen/accuracy/", "/tests/", "/rd_candidates/", "/swarm_lab/", "/.bak"))
    code_findings = list(cs.findings)
    all_findings += code_findings
    checks.append(_check("check::code_sweep", "code_sweep", "Code Sweep",
                         code_findings, cs.files_scanned,
                         f"{cs.files_scanned} files, {cs.flagged} unattributed risky claim(s), "
                         f"{cs.allowed} attributed."))
    role["code_auditor"] = {"stage": "code_sweep", "files": cs.files_scanned,
                            "flagged": cs.flagged, "allowed": cs.allowed,
                            "summary": f"Scanned source for risky claims/placeholders; "
                                       f"{cs.flagged} unattributed, {cs.allowed} disclaimed/cited."}

    # 3. biological_audit
    bio_dims = bio.run_biological_benchmark()
    bio_findings = bio.findings_from(bio_dims)
    all_findings += bio_findings
    bio_checked = sum(d.checked for d in bio_dims)
    checks.append(_check("check::biological", "biological", "Biological Benchmark",
                         bio_findings, bio_checked,
                         f"{len(bio_dims)} dimensions, {bio_checked} checks, "
                         f"{sum(1 for f in bio_findings if f.is_problem)} problem(s)."))
    role["biological_auditor"] = {
        "stage": "biological_audit",
        "dimensions": {d.dimension: d.status for d in bio_dims},
        "summary": "Checked CYP/transporter directionality, in-vitro/clinical separation, "
                   "marker-reaction exclusion, strength!=severity, stereochemistry, PK sanity.",
    }

    # 4. evidence_audit
    ev_results = evidence_audit.run_evidence_audit()
    ev_findings = evidence_audit.findings_from(ev_results)
    all_findings += ev_findings
    ev_checked = sum(r.records_checked for r in ev_results)
    checks.append(_check("check::evidence", "evidence", "Evidence Provenance",
                         ev_findings, ev_checked,
                         f"{len(ev_results)} artifacts, {ev_checked} records, "
                         f"{sum(1 for f in ev_findings if f.is_problem)} problem(s)."))
    role["evidence_auditor"] = {
        "stage": "evidence_audit", "records": ev_checked,
        "summary": "Verified citation/source_url/retrieved_at, uncited-severity, "
                   "append-only review events, and no fake human review.",
    }

    # 5. hallucination_audit — reports, artifacts, agent chats, feeds.
    # Exclude raw/ (quoted FDA source bodies) and the Accuracy Lab's own design
    # docs (V47/V48), which enumerate the forbidden phrases as detection examples
    # — self-reference, not drug claims (same principle as excluding the detector
    # package from the code sweep).
    hs = code_sweep.scan_paths([
        root / "vortext" / "reports",
        root / "data" / "flywheel",
        root / ".coordination" / "agent_chats",
        root / ".coordination" / "live_feed",
    ], exclude=("/raw/", "V47_ACCURACY_LAB", "V48_ACCURACY_LAB"))
    hall_findings = list(hs.findings)
    all_findings += hall_findings
    checks.append(_check("check::hallucination", "hallucination", "Hallucination Risk",
                         hall_findings, hs.files_scanned,
                         f"{hs.files_scanned} docs, {hs.flagged} unattributed claim(s), "
                         f"{hs.allowed} attributed."))
    role["hallucination_auditor"] = {"stage": "hallucination_audit", "docs": hs.files_scanned,
                                      "flagged": hs.flagged, "allowed": hs.allowed,
                                      "summary": "Swept reports/artifacts/chats/feeds for uncited "
                                                 "clinical/FDA claims and placeholders."}

    # 6. replay_audit
    rp_findings = replay_audit.run_replay_audit()
    all_findings += rp_findings
    checks.append(_check("check::replay", "replay", "Replay Determinism",
                         rp_findings, len(rp_findings),
                         f"{sum(1 for f in rp_findings if f.status=='pass')} stable, "
                         f"{sum(1 for f in rp_findings if f.is_problem)} drift."))
    role["replay_runner"] = {"stage": "replay_audit",
                             "summary": "Recomputed provenance hashes, pair ids, content hashes; "
                                        "checked compute_status reproducibility + duplicates."}

    # 7. ui_audit
    ui_findings = _ui_audit(root)
    all_findings += ui_findings
    checks.append(_check("check::ui", "ui", "VORTEXT / API Integrity",
                         ui_findings, len(ui_findings),
                         f"{sum(1 for f in ui_findings if f.is_problem)} UI/API gap(s)."))
    role["ui_auditor"] = {"stage": "ui_audit",
                          "summary": "Checked accuracy API endpoints, terminal components, "
                                     "and the no-medical-claim safety banner."}

    # 8. validation (test results injected by the CLI; informational here)
    tr = test_results or {}
    val_findings: list[AccuracyFinding] = []
    if tr:
        failed = int(tr.get("failed", 0))
        val_findings.append(AccuracyFinding(
            check_id="check::validation::pytest", check_type="validation",
            severity="high" if failed else "info",
            status="fail" if failed else "pass",
            explanation=f"pytest: {tr.get('passed', 0)} passed, {failed} failed"
                        + (f", {tr.get('errors', 0)} collection error(s)" if tr.get("errors") else ""),
            remediation="Fix failing tests before release." if failed else None))
    else:
        val_findings.append(AccuracyFinding(
            check_id="check::validation::pytest", check_type="validation",
            severity="low", status="warning",
            explanation="no pytest results supplied to this run",
            remediation="Run with --run-tests to attach the suite result."))
    all_findings += val_findings
    checks.append(_check("check::validation", "validation", "Test Suite",
                         val_findings, len(val_findings), tr.get("summary", "no tests run")
                         if tr else "no tests run"))
    role["validation_runner"] = {"stage": "validation", "test_results": tr,
                                 "summary": tr.get("summary", "no tests attached") if tr else "no tests attached"}

    # 9. final_gate
    gate = gate_from_findings(all_findings)
    overall = round(sum(c.score for c in checks) / max(1, len(checks)), 1)
    counts = {s: sum(1 for f in all_findings if f.is_problem and f.severity == s)
              for s in ("critical", "high", "medium", "low", "info")}
    role["final_arbiter"] = {"stage": "final_gate", "gate": gate.status,
                             "overall_score": overall, "finding_counts": counts,
                             "reasons": list(gate.reasons),
                             "summary": f"Release gate {gate.status} — {'; '.join(gate.reasons)}."}

    run = AccuracyRun(
        run_id=f"accuracy-{started.replace(':', '').replace('-', '')}",
        scope=scope, started_at=started, completed_at=_utc(),
        checks=tuple(checks), release_gate=gate, overall_score=overall,
        finding_counts=counts, test_results=tr,
    )
    return WorkflowResult(
        run=run, biological_dimensions=bio_dims, evidence_results=ev_results,
        hallucination=hs, code_sweep=cs, role_summaries=role, all_findings=all_findings,
    )
