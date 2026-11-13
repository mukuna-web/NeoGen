"""Scorecard — aggregates benchmark results into a V1 quality report.

The scorecard is the P6 deliverable. Its job is to turn a batch of
:class:`BenchmarkResult` / :class:`DDIBenchmarkResult` rows into the
headline numbers that decide whether V1 is "good":

* **% within 2-fold** across all scored metrics.
* **Geometric mean fold error (GMFE)** — the standard PBPK accuracy
  statistic. GMFE = 1.0 is a perfect match; GMFE = 2.0 means the
  typical prediction is off by a factor of two in either direction.
* **Per-drug breakdown** for auditability.

Headline metric
---------------

V1 freeze target is **≥60% within 2-fold**, tracked across Cmax, AUC,
and t½ for the single-drug set, and AUCR for the DDI set. This matches
the plan in ``plans/rosy-splashing-pudding.md §6``.

Output formats
--------------

The scorecard supports:

* :meth:`to_summary_lines` — compact terminal-friendly text block
* :meth:`to_markdown_table` — one row per case, suitable for commit
  messages and ``reports/*.md``
* :meth:`to_json` — machine-readable dump for the P7 API layer

No Matplotlib/HTML/PDF here — visual output is P6.4 material and lives
in a separate module so the scorecard itself stays text-pure.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from neogen.validation.benchmark_runner import (
    BenchmarkResult,
    DDIBenchmarkResult,
    MetricScore,
)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _geometric_mean(values: Sequence[float]) -> float:
    """Log-mean of positive values. Empty → ``nan``, negatives → ``nan``."""
    clean = [v for v in values if v > 0 and math.isfinite(v)]
    if not clean:
        return float("nan")
    log_sum = sum(math.log(v) for v in clean)
    return math.exp(log_sum / len(clean))


# ════════════════════════════════════════════════════════════════
# SINGLE-DRUG SCORECARD
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated stats for one metric (Cmax / AUC / t½ / AUCR)."""

    name: str
    n_scored: int
    n_within_tolerance: int
    gmfe: float

    @property
    def pct_within_tolerance(self) -> float:
        if self.n_scored == 0:
            return float("nan")
        return 100.0 * self.n_within_tolerance / self.n_scored


@dataclass(frozen=True)
class Scorecard:
    """V1 benchmark scorecard.

    Constructed from a tuple of :class:`BenchmarkResult` rows. Holds
    both the raw results (for auditing / per-case drill-down) and the
    aggregated metric summaries.
    """

    results: Tuple[BenchmarkResult, ...]
    cmax_summary: MetricSummary
    auc_summary: MetricSummary
    half_life_summary: MetricSummary
    total_scored: int
    total_within_tolerance: int
    overall_gmfe: float
    engine_version: str

    @classmethod
    def build(cls, results: Sequence[BenchmarkResult]) -> "Scorecard":
        results_tuple = tuple(results)

        # Exclude perpetrator-role cases from scoring — they're in the
        # dataset for DDI baseline computation but their single-drug PK
        # is not meaningful for T1 accuracy (CLint is tuned for DDI, not
        # single-drug AUC). They still appear in the results for display.
        scored_results = [
            r for r in results_tuple
            if getattr(r, "role", "victim") != "perpetrator"
        ]

        cmax_scored = [r.cmax for r in scored_results if r.cmax.is_scored]
        auc_scored = [r.auc for r in scored_results if r.auc.is_scored]
        hl_scored = [r.half_life for r in scored_results if r.half_life.is_scored]

        cmax_summary = _build_metric_summary("Cmax", cmax_scored)
        auc_summary = _build_metric_summary("AUC", auc_scored)
        hl_summary = _build_metric_summary("t½", hl_scored)

        all_scored = cmax_scored + auc_scored + hl_scored
        total_scored = len(all_scored)
        total_within = sum(1 for m in all_scored if m.within_tolerance)

        gmfe = _geometric_mean([m.fold_error for m in all_scored])

        engine_version = (
            results_tuple[0].engine_version if results_tuple else "unknown"
        )

        return cls(
            results=results_tuple,
            cmax_summary=cmax_summary,
            auc_summary=auc_summary,
            half_life_summary=hl_summary,
            total_scored=total_scored,
            total_within_tolerance=total_within,
            overall_gmfe=gmfe,
            engine_version=engine_version,
        )

    # ── derived numbers ────────────────────────────────────────

    @property
    def n_cases(self) -> int:
        return len(self.results)

    @property
    def n_successful(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def overall_pct_within_tolerance(self) -> float:
        if self.total_scored == 0:
            return float("nan")
        return 100.0 * self.total_within_tolerance / self.total_scored

    @property
    def passes_v1_gate(self) -> bool:
        """V1 freeze target: ≥60% within 2-fold across all metrics."""
        return self.overall_pct_within_tolerance >= 60.0

    # ── rendering ──────────────────────────────────────────────

    def to_summary_lines(self) -> Tuple[str, ...]:
        lines = [
            f"== NeoGen V1 PK Benchmark Scorecard ==",
            f"  Engine version:     {self.engine_version}",
            f"  Cases run:          {self.n_cases} "
            f"({self.n_successful} successful)",
            f"  Metrics scored:     {self.total_scored}",
            f"  Within 2-fold:      "
            f"{self.total_within_tolerance}/{self.total_scored} "
            f"= {self.overall_pct_within_tolerance:.1f}%",
            f"  GMFE (all metrics): {self.overall_gmfe:.2f}×",
            "  -- per metric --",
        ]
        for m in (
            self.cmax_summary,
            self.auc_summary,
            self.half_life_summary,
        ):
            if m.n_scored == 0:
                lines.append(f"  {m.name:<8} (not scored)")
                continue
            lines.append(
                f"  {m.name:<8} "
                f"{m.n_within_tolerance}/{m.n_scored} within 2× "
                f"({m.pct_within_tolerance:.1f}%)  "
                f"GMFE {m.gmfe:.2f}×"
            )
        gate = "PASS" if self.passes_v1_gate else "FAIL"
        lines.append(f"  V1 freeze gate (≥60% within 2×): {gate}")
        return tuple(lines)

    def to_markdown_table(self) -> str:
        """One row per case with predicted/observed/fold-error columns."""
        header = (
            "| Case | Drug | Cmax pred | Cmax obs | Cmax fold | "
            "AUC pred | AUC obs | AUC fold | t½ pred | t½ obs | t½ fold | Pass |"
        )
        sep = "|" + "|".join(["---"] * 12) + "|"
        rows = [header, sep]
        for r in self.results:
            rows.append(_render_row(r))
        return "\n".join(rows)

    def to_json(self) -> str:
        payload = {
            "engine_version": self.engine_version,
            "n_cases": self.n_cases,
            "n_successful": self.n_successful,
            "overall_pct_within_2fold": self.overall_pct_within_tolerance,
            "overall_gmfe": self.overall_gmfe,
            "passes_v1_gate": self.passes_v1_gate,
            "per_metric": {
                "cmax": _metric_json(self.cmax_summary),
                "auc": _metric_json(self.auc_summary),
                "half_life": _metric_json(self.half_life_summary),
            },
            "cases": [_case_json(r) for r in self.results],
        }
        return json.dumps(payload, indent=2)


# ════════════════════════════════════════════════════════════════
# DDI SCORECARD
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DDIScorecard:
    """V1 DDI benchmark scorecard — one AUCR per victim/perpetrator pair."""

    results: Tuple[DDIBenchmarkResult, ...]
    aucr_summary: MetricSummary
    engine_version: str

    @classmethod
    def build(cls, results: Sequence[DDIBenchmarkResult]) -> "DDIScorecard":
        results_tuple = tuple(results)
        scored = [r.aucr for r in results_tuple if r.aucr.is_scored]
        summary = _build_metric_summary("AUCR", scored)
        engine_version = (
            results_tuple[0].engine_version if results_tuple else "unknown"
        )
        return cls(
            results=results_tuple,
            aucr_summary=summary,
            engine_version=engine_version,
        )

    @property
    def n_cases(self) -> int:
        return len(self.results)

    @property
    def n_successful(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def passes_v1_gate(self) -> bool:
        return self.aucr_summary.pct_within_tolerance >= 60.0

    def to_summary_lines(self) -> Tuple[str, ...]:
        s = self.aucr_summary
        lines = [
            "== NeoGen V1 DDI Benchmark Scorecard ==",
            f"  Engine version:   {self.engine_version}",
            f"  Pairs run:        {self.n_cases} "
            f"({self.n_successful} successful)",
            f"  AUCR scored:      {s.n_scored}",
            f"  Within 2-fold:    "
            f"{s.n_within_tolerance}/{s.n_scored} "
            f"= {s.pct_within_tolerance:.1f}%",
            f"  GMFE:             {s.gmfe:.2f}×",
        ]
        gate = "PASS" if self.passes_v1_gate else "FAIL"
        lines.append(f"  V1 DDI gate (≥60% within 2×): {gate}")
        return tuple(lines)

    def to_markdown_table(self) -> str:
        header = (
            "| Case | Victim | Perpetrator | Predicted AUCR | "
            "Observed AUCR | Fold error | Pass |"
        )
        sep = "|" + "|".join(["---"] * 7) + "|"
        rows = [header, sep]
        for r in self.results:
            rows.append(_render_ddi_row(r))
        return "\n".join(rows)


# ════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════


def _build_metric_summary(
    name: str, scored: Sequence[MetricScore]
) -> MetricSummary:
    n_scored = len(scored)
    n_within = sum(1 for m in scored if m.within_tolerance)
    gmfe = _geometric_mean([m.fold_error for m in scored])
    return MetricSummary(
        name=name,
        n_scored=n_scored,
        n_within_tolerance=n_within,
        gmfe=gmfe,
    )


def _metric_json(m: MetricSummary) -> dict:
    return {
        "n_scored": m.n_scored,
        "n_within_tolerance": m.n_within_tolerance,
        "pct_within_tolerance": m.pct_within_tolerance,
        "gmfe": m.gmfe,
    }


def _case_json(r: BenchmarkResult) -> dict:
    def _m(ms: MetricScore) -> dict:
        return {
            "observed": ms.observed,
            "predicted": ms.predicted,
            "fold_error": ms.fold_error,
            "within_tolerance": ms.within_tolerance,
        }

    return {
        "name": r.case_name,
        "drug": r.drug_name,
        "success": r.success,
        "error": r.error_message,
        "cmax": _m(r.cmax),
        "auc": _m(r.auc),
        "half_life": _m(r.half_life),
        "all_within_tolerance": r.all_within_tolerance,
    }


def _fmt_num(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    if not math.isfinite(v):
        return "∞"
    return f"{v:.{digits}f}"


def _fmt_pass(within: Optional[bool]) -> str:
    if within is None:
        return "—"
    return "✓" if within else "✗"


def _render_row(r: BenchmarkResult) -> str:
    if not r.success:
        return (
            f"| {r.case_name} | {r.drug_name} | — | — | — | — | — | "
            f"— | — | — | — | ✗ (err: {r.error_message}) |"
        )
    all_pass = "✓" if r.all_within_tolerance else "✗"
    return (
        f"| {r.case_name} | {r.drug_name} | "
        f"{_fmt_num(r.cmax.predicted)} | {_fmt_num(r.cmax.observed)} | "
        f"{_fmt_num(r.cmax.fold_error, 2)} | "
        f"{_fmt_num(r.auc.predicted)} | {_fmt_num(r.auc.observed)} | "
        f"{_fmt_num(r.auc.fold_error, 2)} | "
        f"{_fmt_num(r.half_life.predicted)} | {_fmt_num(r.half_life.observed)} | "
        f"{_fmt_num(r.half_life.fold_error, 2)} | {all_pass} |"
    )


def _render_ddi_row(r: DDIBenchmarkResult) -> str:
    if not r.success:
        return (
            f"| {r.case_name} | {r.victim} | {r.perpetrator} | — | "
            f"{_fmt_num(r.aucr.observed)} | — | ✗ (err: {r.error_message}) |"
        )
    return (
        f"| {r.case_name} | {r.victim} | {r.perpetrator} | "
        f"{_fmt_num(r.aucr.predicted)} | {_fmt_num(r.aucr.observed)} | "
        f"{_fmt_num(r.aucr.fold_error, 2)} | "
        f"{_fmt_pass(r.aucr.within_tolerance)} |"
    )
