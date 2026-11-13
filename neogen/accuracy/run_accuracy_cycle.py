"""CLI entrypoint for an accuracy cycle.

    python -m neogen.accuracy.run_accuracy_cycle --scope ddi-flywheel \
        --out data/accuracy/runs/latest [--run-tests]

Runs the Opus-class workflow, optionally collects a curated pytest subset, and
writes summary.json / findings.json / release_gate.json / agent_summaries/*.md
plus a markdown report. Appends VORTEXT feed lines. No data is mutated; the FDA
science engine is untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from neogen.accuracy.workflow import WorkflowResult, run_workflow

_ROOT = Path(__file__).resolve().parents[2]
FEED_DIR = _ROOT / ".coordination" / "live_feed"
TASK_ID = "accuracy-lab-2026-05-30"

_CURATED_TESTS = [
    "tests/test_accuracy_contracts.py", "tests/test_accuracy_code_sweep.py",
    "tests/test_accuracy_evidence_audit.py", "tests/test_accuracy_biological_benchmark.py",
    "tests/test_accuracy_replay_audit.py", "tests/test_accuracy_workflow.py",
    "tests/test_fda_mechanism_joiner.py", "tests/test_ddi_pair_record_replay.py",
]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_pytest(paths: list[str]) -> dict:
    """Best-effort curated pytest; parse the summary line."""
    existing = [p for p in paths if (_ROOT / p).exists()]
    if not existing:
        return {}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *existing, "-q", "-p", "no:cacheprovider"],
            cwd=_ROOT, capture_output=True, text=True, timeout=300,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"summary": f"pytest could not run: {str(exc)[:120]}", "errors": 1}
    out = proc.stdout + proc.stderr
    passed = int((re.search(r"(\d+) passed", out) or [0, 0])[1]) if "passed" in out else 0
    failed = int((re.search(r"(\d+) failed", out) or [0, 0])[1]) if "failed" in out else 0
    errors = int((re.search(r"(\d+) error", out) or [0, 0])[1]) if "error" in out else 0
    return {"passed": passed, "failed": failed, "errors": errors,
            "files": len(existing),
            "summary": f"{passed} passed, {failed} failed, {errors} error(s) across {len(existing)} files"}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def build_summary(res: WorkflowResult) -> dict:
    run = res.run
    return {
        **{k: v for k, v in run.to_dict().items() if k != "checks"},
        "scorecards": [
            {"check_id": c.check_id, "title": c.title, "check_type": c.check_type,
             "status": c.status, "severity": c.severity, "score": c.score,
             "summary": c.summary,
             # real problems (gate-relevant) vs informational notes
             "problem_count": sum(1 for f in c.findings
                                  if f.is_problem and f.severity in ("critical", "high", "medium")),
             "note_count": sum(1 for f in c.findings
                               if f.is_problem and f.severity in ("low", "info"))}
            for c in run.checks
        ],
        "biological_matrix": [d.to_dict() | {"findings": len(d.findings)} if False else {
            "dimension": d.dimension, "checked": d.checked, "passed": d.passed,
            "failed": d.failed, "status": d.status, "details": d.details}
            for d in res.biological_dimensions],
        "code_sweep": {"files_scanned": res.code_sweep.files_scanned,
                       "flagged": res.code_sweep.flagged, "allowed": res.code_sweep.allowed},
        "hallucination": {"docs_scanned": res.hallucination.files_scanned,
                          "flagged": res.hallucination.flagged, "allowed": res.hallucination.allowed},
        "evidence_artifacts": [
            {"artifact": r.artifact, "records_checked": r.records_checked,
             "complete": r.complete, "incomplete": r.incomplete}
            for r in res.evidence_results],
    }


def write_run(res: WorkflowResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", build_summary(res))
    _write_json(out_dir / "findings.json", {
        "run_id": res.run.run_id, "generated_at": _utc(),
        "total": len(res.all_findings),
        "problems": sum(1 for f in res.all_findings if f.is_problem),
        "findings": [f.to_dict() for f in res.all_findings],
    })
    _write_json(out_dir / "release_gate.json", {
        "run_id": res.run.run_id, "generated_at": _utc(),
        "overall_score": res.run.overall_score,
        **res.run.release_gate.to_dict(),
        "scorecard_status": {c.title: c.status for c in res.run.checks},
    })
    summ_dir = out_dir / "agent_summaries"
    summ_dir.mkdir(parents=True, exist_ok=True)
    for role, payload in res.role_summaries.items():
        _write_json(summ_dir / f"{role}.json", payload)
        md = [f"# {role.replace('_', ' ').title()}", "",
              f"- **stage:** {payload.get('stage', '?')}", ""]
        md.append(payload.get("summary", ""))
        for k, v in payload.items():
            if k in ("stage", "summary"):
                continue
            md.append(f"\n- **{k}:** `{v}`")
        (summ_dir / f"{role}.md").write_text("\n".join(md) + "\n")


def write_report(res: WorkflowResult, out_dir: Path) -> None:
    r = res.run
    g = r.release_gate
    lines = [
        f"# Accuracy Cycle — {r.run_id}", "",
        "> Research/prototype accuracy only. NOT clinical validation, NOT FDA "
        "approval/acceptance/clearance, NOT medical advice. No severity asserted.", "",
        f"- **scope:** {r.scope}", f"- **release gate:** {g.status} ({'; '.join(g.reasons)})",
        f"- **overall score:** {r.overall_score}/100 (research-grade)",
        f"- **findings:** critical={g.critical} high={g.high} medium={g.medium} low={g.low} blocked={g.blocked}",
        "", "## Scorecards", "", "| Card | Status | Score | Problems | Summary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in r.checks:
        probs = sum(1 for f in c.findings if f.is_problem)
        lines.append(f"| {c.title} | {c.status} | {c.score:.0f} | {probs} | {c.summary} |")
    lines += ["", "## Biological benchmark matrix", "",
              "| Dimension | Status | Checked | Failed |", "| --- | --- | --- | --- |"]
    for d in res.biological_dimensions:
        lines.append(f"| {d.dimension} | {d.status} | {d.checked} | {d.failed} |")
    probs = [f for f in res.all_findings if f.is_problem]
    lines += ["", f"## Problems ({len(probs)})", ""]
    if not probs:
        lines.append("_No critical/high/medium problems found in this run._")
    for f in sorted(probs, key=lambda x: x.severity, reverse=True)[:50]:
        loc = f"{f.file_path}:{f.line}" if f.file_path else f.check_type
        lines.append(f"- **[{f.severity}/{f.status}]** {f.explanation} — `{loc}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def append_feeds(res: WorkflowResult) -> None:
    if not FEED_DIR.exists():
        return
    g = res.run.release_gate
    n = len(res.all_findings)
    rows = [
        ("research_loop", "neogen-accuracy-planner", "intake",
         "Accuracy Lab cycle: code+evidence+biological+hallucination+replay+ui+validation."),
        ("research_loop", "neogen-code-auditor", "code-sweep",
         f"code sweep: {res.code_sweep.flagged} unattributed, {res.code_sweep.allowed} attributed."),
        ("parchment_pipeline", "neogen-hallucination-auditor", "hallucination-audit",
         f"report/artifact sweep: {res.hallucination.flagged} unattributed claim(s)."),
        ("parchment_pipeline", "neogen-evidence-auditor", "evidence-audit",
         f"{sum(r.records_checked for r in res.evidence_results)} records checked for provenance."),
        ("research_loop", "neogen-biological-auditor", "biological-benchmark",
         f"{len(res.biological_dimensions)} biological dimensions benchmarked."),
        ("research_loop", "neogen-replay-runner", "replay-audit",
         "recomputed provenance/pair/content hashes + status reproducibility."),
        ("agent_terminal", "neogen-final-arbiter", "report",
         f"release gate {g.status}; overall {res.run.overall_score}/100; {n} findings."),
        ("claude_codex", "neogen-final-arbiter", "complete",
         f"Accuracy cycle complete: gate={g.status}, no clinical claim, engine untouched."),
    ]
    for channel, agent, step, msg in rows:
        fp = FEED_DIR / f"{channel}.log"
        line = (f'ts={_utc()} | agent={agent} | task_id={TASK_ID} | step={step} '
                f'| files=0 | msg="{msg}"\n')
        with open(fp, "a") as fh:
            fh.write(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a NeoGen accuracy cycle.")
    ap.add_argument("--scope", default="ddi-flywheel")
    ap.add_argument("--out", default="data/accuracy/runs/latest")
    ap.add_argument("--run-tests", action="store_true", help="collect a curated pytest subset")
    ap.add_argument("--no-feed", action="store_true", help="skip appending VORTEXT feed lines")
    args = ap.parse_args(argv)

    tr = _run_pytest(_CURATED_TESTS) if args.run_tests else {}
    res = run_workflow(args.scope, test_results=tr)

    out = (_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    write_run(res, out)
    write_report(res, out)
    # archive a copy under the run id for history
    archive = _ROOT / "data" / "accuracy" / "runs" / res.run.run_id
    if archive.resolve() != out.resolve():
        write_run(res, archive)
        write_report(res, archive)
    if not args.no_feed:
        append_feeds(res)

    g = res.run.release_gate
    print(json.dumps({
        "run_id": res.run.run_id, "gate": g.status, "overall_score": res.run.overall_score,
        "findings": {"critical": g.critical, "high": g.high, "medium": g.medium,
                     "low": g.low, "blocked": g.blocked},
        "out": str(out), "tests": tr.get("summary", "not run"),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
