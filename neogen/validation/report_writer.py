"""V1 benchmark report writer — composes PK + DDI scorecards into one artifact.

This is the P6.4 deliverable. :class:`Scorecard` and :class:`DDIScorecard`
already know how to render themselves as text, markdown, or JSON; their
focus is *what the engine scored*, not *what the run was*. The report
writer adds the layer on top: run metadata (timestamp, engine version,
dataset sizes), a composite headline, and a single markdown / JSON
artifact that a human or a CI pipeline can consume.

Design rules
------------

1. **Frozen dataclasses.** ``RunMetadata`` and ``BenchmarkReport`` are
   immutable. No side effects inside the classes — writing to disk is
   an explicit, separate method.
2. **Text-only.** No HTML, no PDF, no Matplotlib. Visual output can
   layer on top later; the report itself is pure markdown + JSON so
   CI tooling can parse it.
3. **Either half is optional.** A PK-only or DDI-only run must still
   produce a valid report (the missing half is shown as "not run").
4. **No network, no git shelling by default.** ``RunMetadata.capture()``
   reads the local clock and the engine version and nothing else. The
   CLI runner can enrich the metadata with git SHA if it wants to.

This module is a pure composition layer. All scoring math lives in
:mod:`neogen.validation.scorecard`, all simulation lives in
:mod:`neogen.validation.benchmark_runner`.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from neogen.validation.benchmark_runner import (
    BenchmarkResult,
    DDIBenchmarkResult,
    MetricScore,
)
from neogen.validation.scorecard import (
    DDIScorecard,
    MetricSummary,
    Scorecard,
)


# ════════════════════════════════════════════════════════════════
# RUN METADATA
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunMetadata:
    """Non-scoring context for a benchmark run.

    Attributes
    ----------
    timestamp_utc
        ISO-8601 timestamp of when the run was captured.
    engine_version
        Version string taken from the scorecard (or passed in
        explicitly if no results ran).
    host
        ``"{user}@{hostname}"`` — who ran the benchmark. Defaults to
        the current process's environment.
    git_sha
        Optional short SHA for the repo state. Never shelled out to
        by this module — callers must provide it.
    notes
        Free-form commentary for the run (e.g. "post-P3.5 MBI
        recalibration"). Rendered in the markdown header.
    """

    timestamp_utc: str
    engine_version: str
    host: str
    git_sha: Optional[str] = None
    notes: str = ""

    @classmethod
    def capture(
        cls,
        engine_version: str,
        git_sha: Optional[str] = None,
        notes: str = "",
    ) -> "RunMetadata":
        """Snapshot local run context. Pure — no git shell-out."""
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
        hostname = os.environ.get("HOSTNAME") or os.uname().nodename
        return cls(
            timestamp_utc=now,
            engine_version=engine_version,
            host=f"{user}@{hostname}",
            git_sha=git_sha,
            notes=notes,
        )


# ════════════════════════════════════════════════════════════════
# BENCHMARK REPORT
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BenchmarkReport:
    """Composite V1 benchmark report.

    Bundles an optional :class:`Scorecard` (single-drug PK), an optional
    :class:`DDIScorecard`, and a :class:`RunMetadata` block into one
    immutable object that can render itself as markdown or JSON.
    """

    metadata: RunMetadata
    pk_scorecard: Optional[Scorecard]
    ddi_scorecard: Optional[DDIScorecard]

    @classmethod
    def build(
        cls,
        metadata: RunMetadata,
        pk_scorecard: Optional[Scorecard] = None,
        ddi_scorecard: Optional[DDIScorecard] = None,
    ) -> "BenchmarkReport":
        if pk_scorecard is None and ddi_scorecard is None:
            raise ValueError(
                "BenchmarkReport.build requires at least one scorecard "
                "(pk_scorecard, ddi_scorecard, or both)"
            )
        return cls(
            metadata=metadata,
            pk_scorecard=pk_scorecard,
            ddi_scorecard=ddi_scorecard,
        )

    # ── headline aggregation ───────────────────────────────────

    @property
    def total_scored(self) -> int:
        n = 0
        if self.pk_scorecard is not None:
            n += self.pk_scorecard.total_scored
        if self.ddi_scorecard is not None:
            n += self.ddi_scorecard.aucr_summary.n_scored
        return n

    @property
    def total_within_tolerance(self) -> int:
        n = 0
        if self.pk_scorecard is not None:
            n += self.pk_scorecard.total_within_tolerance
        if self.ddi_scorecard is not None:
            n += self.ddi_scorecard.aucr_summary.n_within_tolerance
        return n

    @property
    def combined_pct_within_tolerance(self) -> float:
        if self.total_scored == 0:
            return float("nan")
        return 100.0 * self.total_within_tolerance / self.total_scored

    @property
    def passes_v1_gate(self) -> bool:
        """Both halves must pass their own 60% gate if present.

        Either half being missing is not a pass — it's an unknown, and
        callers that want a strict pass/fail should run both halves.
        """
        pk_pass = self.pk_scorecard is None or self.pk_scorecard.passes_v1_gate
        ddi_pass = (
            self.ddi_scorecard is None or self.ddi_scorecard.passes_v1_gate
        )
        has_any = self.pk_scorecard is not None or self.ddi_scorecard is not None
        return has_any and pk_pass and ddi_pass

    # ── rendering ──────────────────────────────────────────────

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# NeoGen V1 Benchmark Report")
        lines.append("")
        lines.extend(self._metadata_block())
        lines.append("")
        lines.extend(self._headline_block())
        lines.append("")
        if self.pk_scorecard is not None:
            lines.extend(self._pk_section())
            lines.append("")
        else:
            lines.append("## PK benchmark — not run")
            lines.append("")
        if self.ddi_scorecard is not None:
            lines.extend(self._ddi_section())
            lines.append("")
        else:
            lines.append("## DDI benchmark — not run")
            lines.append("")
        failures = self._failure_section()
        if failures:
            lines.extend(failures)
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "metadata": {
                "timestamp_utc": self.metadata.timestamp_utc,
                "engine_version": self.metadata.engine_version,
                "host": self.metadata.host,
                "git_sha": self.metadata.git_sha,
                "notes": self.metadata.notes,
            },
            "headline": {
                "total_scored": self.total_scored,
                "total_within_tolerance": self.total_within_tolerance,
                "combined_pct_within_tolerance": (
                    self.combined_pct_within_tolerance
                    if not math.isnan(self.combined_pct_within_tolerance)
                    else None
                ),
                "passes_v1_gate": self.passes_v1_gate,
            },
            "pk": _pk_json(self.pk_scorecard),
            "ddi": _ddi_json(self.ddi_scorecard),
        }
        return json.dumps(payload, indent=2, default=_json_default)

    def write_markdown(self, path: str) -> str:
        """Write the markdown report to disk. Returns the path."""
        text = self.to_markdown()
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def write_json(self, path: str) -> str:
        """Write the JSON report to disk. Returns the path."""
        text = self.to_json()
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    # ── rendering helpers ──────────────────────────────────────

    def _metadata_block(self) -> list[str]:
        meta = self.metadata
        lines = [
            f"- **Timestamp (UTC):** {meta.timestamp_utc}",
            f"- **Engine version:** `{meta.engine_version}`",
            f"- **Host:** {meta.host}",
        ]
        if meta.git_sha:
            lines.append(f"- **Git SHA:** `{meta.git_sha}`")
        if meta.notes:
            lines.append(f"- **Notes:** {meta.notes}")
        return lines

    def _headline_block(self) -> list[str]:
        gate = "PASS" if self.passes_v1_gate else "FAIL"
        pct = self.combined_pct_within_tolerance
        pct_str = f"{pct:.1f}%" if not math.isnan(pct) else "n/a"
        return [
            "## Headline",
            "",
            f"- **Combined metrics scored:** {self.total_scored}",
            f"- **Combined within 2-fold:** "
            f"{self.total_within_tolerance}/{self.total_scored} = {pct_str}",
            f"- **V1 freeze gate (≥60% within 2× on every half run):** {gate}",
        ]

    def _pk_section(self) -> list[str]:
        sc = self.pk_scorecard
        assert sc is not None
        lines = ["## PK benchmark", ""]
        lines.extend("    " + ln for ln in sc.to_summary_lines())
        lines.append("")
        lines.append("### Per-case PK results")
        lines.append("")
        lines.append(sc.to_markdown_table())
        return lines

    def _ddi_section(self) -> list[str]:
        sc = self.ddi_scorecard
        assert sc is not None
        lines = ["## DDI benchmark", ""]
        lines.extend("    " + ln for ln in sc.to_summary_lines())
        lines.append("")
        lines.append("### Per-case DDI results")
        lines.append("")
        lines.append(sc.to_markdown_table())
        return lines

    def _failure_section(self) -> list[str]:
        """Surface out-of-tolerance cases so they're easy to spot in CI."""
        pk_fails: list[str] = []
        if self.pk_scorecard is not None:
            for r in self.pk_scorecard.results:
                if not r.success:
                    pk_fails.append(
                        f"- `{r.case_name}` — engine raised: "
                        f"{r.error_message}"
                    )
                    continue
                bad_metrics = _failing_metrics(
                    [("Cmax", r.cmax), ("AUC", r.auc), ("t½", r.half_life)]
                )
                if bad_metrics:
                    pk_fails.append(
                        f"- `{r.case_name}` ({r.drug_name}) — {bad_metrics}"
                    )

        ddi_fails: list[str] = []
        if self.ddi_scorecard is not None:
            for r in self.ddi_scorecard.results:
                if not r.success:
                    ddi_fails.append(
                        f"- `{r.case_name}` — engine raised: "
                        f"{r.error_message}"
                    )
                    continue
                if r.aucr.within_tolerance is False:
                    ddi_fails.append(
                        f"- `{r.case_name}` ({r.victim} + {r.perpetrator}) "
                        f"— AUCR pred={_fmt_num(r.aucr.predicted)}, "
                        f"obs={_fmt_num(r.aucr.observed)}, "
                        f"fold={_fmt_num(r.aucr.fold_error, 2)}×"
                    )

        if not pk_fails and not ddi_fails:
            return []

        lines = ["## Failing / out-of-tolerance cases", ""]
        if pk_fails:
            lines.append("### PK")
            lines.append("")
            lines.extend(pk_fails)
            lines.append("")
        if ddi_fails:
            lines.append("### DDI")
            lines.append("")
            lines.extend(ddi_fails)
        return lines


# ════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ════════════════════════════════════════════════════════════════


def _failing_metrics(
    pairs: list[tuple[str, MetricScore]],
) -> str:
    """Render a compact `Cmax fold=3.1×, AUC fold=2.4×` string for a case.

    Returns an empty string if all scored metrics are in tolerance.
    """
    chunks: list[str] = []
    for name, m in pairs:
        if m.within_tolerance is False and m.fold_error is not None:
            chunks.append(f"{name} fold={m.fold_error:.2f}×")
    return ", ".join(chunks)


def _pk_json(sc: Optional[Scorecard]) -> Optional[dict]:
    if sc is None:
        return None
    return {
        "engine_version": sc.engine_version,
        "n_cases": sc.n_cases,
        "n_successful": sc.n_successful,
        "total_scored": sc.total_scored,
        "total_within_tolerance": sc.total_within_tolerance,
        "overall_pct_within_tolerance": _finite_or_none(
            sc.overall_pct_within_tolerance
        ),
        "overall_gmfe": _finite_or_none(sc.overall_gmfe),
        "passes_v1_gate": sc.passes_v1_gate,
        "per_metric": {
            "cmax": _metric_summary_json(sc.cmax_summary),
            "auc": _metric_summary_json(sc.auc_summary),
            "half_life": _metric_summary_json(sc.half_life_summary),
        },
        "cases": [_pk_case_json(r) for r in sc.results],
    }


def _ddi_json(sc: Optional[DDIScorecard]) -> Optional[dict]:
    if sc is None:
        return None
    return {
        "engine_version": sc.engine_version,
        "n_cases": sc.n_cases,
        "n_successful": sc.n_successful,
        "passes_v1_gate": sc.passes_v1_gate,
        "aucr_summary": _metric_summary_json(sc.aucr_summary),
        "cases": [_ddi_case_json(r) for r in sc.results],
    }


def _metric_summary_json(m: MetricSummary) -> dict:
    return {
        "name": m.name,
        "n_scored": m.n_scored,
        "n_within_tolerance": m.n_within_tolerance,
        "pct_within_tolerance": _finite_or_none(m.pct_within_tolerance),
        "gmfe": _finite_or_none(m.gmfe),
    }


def _pk_case_json(r: BenchmarkResult) -> dict:
    return {
        "name": r.case_name,
        "drug": r.drug_name,
        "success": bool(r.success),
        "error": r.error_message,
        "cmax": _metric_score_json(r.cmax),
        "auc": _metric_score_json(r.auc),
        "half_life": _metric_score_json(r.half_life),
        "all_within_tolerance": _bool_or_none(r.all_within_tolerance),
    }


def _ddi_case_json(r: DDIBenchmarkResult) -> dict:
    return {
        "name": r.case_name,
        "victim": r.victim,
        "perpetrator": r.perpetrator,
        "success": bool(r.success),
        "error": r.error_message,
        "aucr": _metric_score_json(r.aucr),
        "within_tolerance": _bool_or_none(r.within_tolerance),
    }


def _metric_score_json(m: MetricScore) -> dict:
    return {
        "name": m.name,
        "observed": _finite_or_none(m.observed),
        "predicted": _finite_or_none(m.predicted),
        "fold_error": _finite_or_none(m.fold_error),
        "within_tolerance": _bool_or_none(m.within_tolerance),
    }


def _bool_or_none(v: object) -> Optional[bool]:
    """Coerce numpy booleans / None into a true Python ``bool`` or ``None``.

    JSON has no concept of ``np.bool_``. Without this, ``np.True_``
    falls through to the default encoder and gets turned into ``1.0``,
    which makes machine consumers unable to distinguish a *passed*
    metric from an *unscored* one.
    """
    if v is None:
        return None
    return bool(v)


def _finite_or_none(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv):
        return None
    return fv


def _json_default(obj: object) -> object:
    """Fallback JSON encoder — handles numpy scalars."""
    try:
        return float(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(obj)


def _fmt_num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(fv):
        return "∞"
    return f"{fv:.{digits}f}"


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
