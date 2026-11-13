"""Run the V1 PK + DDI benchmark suites and write a combined report.

Usage
-----

    python3 scripts/run_v1_benchmark.py
    python3 scripts/run_v1_benchmark.py --pk-only
    python3 scripts/run_v1_benchmark.py --ddi-only
    python3 scripts/run_v1_benchmark.py --out-dir reports/benchmarks
    python3 scripts/run_v1_benchmark.py --notes "post-P3.5 MBI recalibration"

Outputs
-------

Writes two files to ``--out-dir`` (default ``reports/benchmarks``)::

    v1_benchmark_<UTC-timestamp>.md
    v1_benchmark_<UTC-timestamp>.json

The markdown artifact is human-friendly (headline, per-case tables,
failing cases highlighted). The JSON artifact is machine-friendly for
CI dashboards and the P7 API layer.

This script is a thin CLI wrapper around :class:`BenchmarkRunner`,
:class:`Scorecard`, :class:`DDIScorecard`, and :class:`BenchmarkReport`.
No scoring math lives here — it's orchestration only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

# Allow running as `python3 scripts/run_v1_benchmark.py` from the repo
# root without needing PYTHONPATH=. — we add the repo root to sys.path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from neogen.engine.orchestrator_v2 import ENGINE_VERSION  # noqa: E402
from neogen.validation import (  # noqa: E402
    BenchmarkReport,
    BenchmarkRunner,
    DDIScorecard,
    RunMetadata,
    Scorecard,
    build_ddi_benchmark_v1,
    build_pk_benchmark_v1,
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V1 PK + DDI benchmark suites."
    )
    parser.add_argument(
        "--out-dir",
        default="reports/benchmarks",
        help="Output directory for the markdown + JSON artifacts.",
    )
    parser.add_argument(
        "--pk-only",
        action="store_true",
        help="Only run the single-drug PK benchmark.",
    )
    parser.add_argument(
        "--ddi-only",
        action="store_true",
        help="Only run the DDI benchmark.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-form notes to record in the report header.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write files — just print the summary to stdout.",
    )
    return parser.parse_args(argv)


def _try_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _run_pk(runner: BenchmarkRunner) -> Scorecard:
    cases = build_pk_benchmark_v1()
    print(f"[V1 benchmark] running PK suite — {len(cases)} cases", flush=True)
    results = runner.run_all(cases)
    return Scorecard.build(results)


def _run_ddi(runner: BenchmarkRunner) -> DDIScorecard:
    cases = build_ddi_benchmark_v1()
    print(f"[V1 benchmark] running DDI suite — {len(cases)} pairs", flush=True)
    results = runner.run_all_ddi(cases)
    return DDIScorecard.build(results)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.pk_only and args.ddi_only:
        print("error: --pk-only and --ddi-only are mutually exclusive",
              file=sys.stderr)
        return 2

    runner = BenchmarkRunner()

    pk_scorecard: Optional[Scorecard] = None
    ddi_scorecard: Optional[DDIScorecard] = None

    if not args.ddi_only:
        pk_scorecard = _run_pk(runner)
    if not args.pk_only:
        ddi_scorecard = _run_ddi(runner)

    metadata = RunMetadata.capture(
        engine_version=ENGINE_VERSION,
        git_sha=_try_git_sha(),
        notes=args.notes,
    )
    report = BenchmarkReport.build(
        metadata=metadata,
        pk_scorecard=pk_scorecard,
        ddi_scorecard=ddi_scorecard,
    )

    # Always print the text summary first.
    print()
    if pk_scorecard is not None:
        for line in pk_scorecard.to_summary_lines():
            print(line)
        print()
    if ddi_scorecard is not None:
        for line in ddi_scorecard.to_summary_lines():
            print(line)
        print()
    gate = "PASS" if report.passes_v1_gate else "FAIL"
    print(f"[V1 benchmark] combined V1 gate: {gate}")

    if args.no_write:
        return 0 if report.passes_v1_gate else 1

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, f"v1_benchmark_{stamp}.md")
    json_path = os.path.join(args.out_dir, f"v1_benchmark_{stamp}.json")
    report.write_markdown(md_path)
    report.write_json(json_path)
    print(f"[V1 benchmark] wrote {md_path}")
    print(f"[V1 benchmark] wrote {json_path}")

    return 0 if report.passes_v1_gate else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
