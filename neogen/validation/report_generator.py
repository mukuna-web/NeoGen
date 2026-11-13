"""FDA-Format PBPK Model Verification Report Generator — Phase 5.2.

Generates a structured Markdown report following the FDA PBPK Model
Verification format as described in:

* FDA Guidance: Physiologically Based Pharmacokinetic Analyses —
  Format and Content (2018)
* EMA Guideline on the Reporting of Physiologically Based
  Pharmacokinetic Modelling and Simulation (2018)

The report is self-contained and printable, covering:

1. Model structure and system parameters
2. Single-drug PK validation (predicted vs observed)
3. DDI validation (predicted vs observed AUCR)
4. Mechanistic invariants (Tier 4 tests)
5. Sensitivity analysis (placeholder)
6. Conclusions and regulatory assessment

Usage
-----
>>> from neogen.validation.report_generator import generate_fda_report
>>> report_md = generate_fda_report(pk_results, ddi_results, tier4, "v1.2.0")
>>> Path("reports/fda_verification_report.md").write_text(report_md)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from neogen.validation.benchmark_runner import BenchmarkResult, DDIBenchmarkResult


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _fmt(val: Optional[float], decimals: int = 2) -> str:
    """Format a float for table display, or '-' if None/NaN."""
    if val is None:
        return "-"
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "-"
    return f"{val:.{decimals}f}"


def _geometric_mean(values: Sequence[float]) -> float:
    """Geometric mean of positive finite values."""
    clean = [v for v in values if v > 0 and math.isfinite(v)]
    if not clean:
        return float("nan")
    log_sum = sum(math.log(v) for v in clean)
    return math.exp(log_sum / len(clean))


def _pct_within(results: Sequence[Any], tolerance: float = 2.0) -> str:
    """Compute percentage of scored results within tolerance fold."""
    scored = [r for r in results if r is not None]
    if not scored:
        return "-"
    within = sum(1 for r in scored if r <= tolerance)
    pct = 100.0 * within / len(scored)
    return f"{within}/{len(scored)} ({pct:.1f}%)"


# ════════════════════════════════════════════════════════════════
# SECTION GENERATORS
# ════════════════════════════════════════════════════════════════


def _section_header(engine_version: str) -> str:
    """Report header with metadata."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "# NeoGen PBPK Model Verification Report\n\n"
        "## FDA-Format Physiologically Based Pharmacokinetic Analysis\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Engine Version | {engine_version} |\n"
        f"| Report Generated | {ts} |\n"
        f"| Platform | NeoGen Mechanistic PBPK Engine |\n"
        f"| Model Type | 14-compartment whole-body PBPK |"
    )


def _section_model_structure() -> str:
    """Describe the PBPK model structure per FDA guidance."""
    return (
        "## 1. Model Structure\n\n"
        "### 1.1 Structural Model\n\n"
        "NeoGen implements a 14-compartment whole-body PBPK model with "
        "the following organ compartments: lung, heart, brain, muscle, "
        "skin, adipose, bone, gut, spleen, liver, kidney, pancreas, "
        "thymus, and rest-of-body.\n\n"
        "### 1.2 Absorption Model\n\n"
        "Oral absorption uses a multi-segment GI tract model with:\n"
        "- Noyes-Whitney dissolution kinetics\n"
        "- Segment-specific permeability (Peff)\n"
        "- Gut-wall CYP3A4/CYP2D6 first-pass metabolism\n"
        "- P-gp efflux modulation\n\n"
        "### 1.3 Distribution Model\n\n"
        "Tissue-to-plasma partition coefficients (Kp) predicted using "
        "the Rodgers-Rowland method with optional Vss calibration from "
        "clinical data.\n\n"
        "### 1.4 Elimination Model\n\n"
        "- Hepatic clearance: well-stirred model with MPPGL scaling\n"
        "- Renal clearance: GFR-based filtration + active tubular secretion\n"
        "- Saturable metabolism: Michaelis-Menten kinetics when Km/Vmax available\n\n"
        "### 1.5 DDI Model\n\n"
        "- Reversible CYP inhibition (competitive, static)\n"
        "- Mechanism-based CYP inactivation (time-dependent, ODE-based)\n"
        "- CYP induction (Emax model)\n"
        "- OATP1B1 transporter inhibition (IC50 model)\n"
        "- Gut-wall MBI (steady-state enterocyte CYP3A4)"
    )


def _section_pk_validation(pk_results: Sequence[BenchmarkResult]) -> str:
    """Generate the PK validation section with predicted-vs-observed tables."""
    lines = [
        "## 2. Single-Drug PK Validation\n",
        "### 2.1 Predicted vs Observed Summary\n",
    ]

    successful = [r for r in pk_results if r.success]
    failed = [r for r in pk_results if not r.success]

    lines.append(
        f"Total cases: {len(pk_results)} | "
        f"Successful: {len(successful)} | "
        f"Failed: {len(failed)}\n"
    )

    # Detailed table
    lines.append(
        "| Drug | Cmax Pred | Cmax Obs | Cmax FE | "
        "AUC Pred | AUC Obs | AUC FE | "
        "t1/2 Pred | t1/2 Obs | t1/2 FE | Pass |"
    )
    lines.append("|" + "|".join(["---"] * 11) + "|")

    for r in successful:
        pass_mark = "Yes" if r.all_within_tolerance else "No"
        lines.append(
            f"| {r.drug_name} "
            f"| {_fmt(r.cmax.predicted)} | {_fmt(r.cmax.observed)} "
            f"| {_fmt(r.cmax.fold_error)} "
            f"| {_fmt(r.auc.predicted)} | {_fmt(r.auc.observed)} "
            f"| {_fmt(r.auc.fold_error)} "
            f"| {_fmt(r.half_life.predicted)} | {_fmt(r.half_life.observed)} "
            f"| {_fmt(r.half_life.fold_error)} "
            f"| {pass_mark} |"
        )

    # Summary statistics
    cmax_fes = [r.cmax.fold_error for r in successful if r.cmax.is_scored]
    auc_fes = [r.auc.fold_error for r in successful if r.auc.is_scored]
    hl_fes = [r.half_life.fold_error for r in successful if r.half_life.is_scored]
    all_fes = cmax_fes + auc_fes + hl_fes

    lines.append("\n### 2.2 Aggregate Statistics\n")
    lines.append("| Metric | N Scored | Within 2-fold | GMFE |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Cmax | {len(cmax_fes)} "
        f"| {_pct_within(cmax_fes)} "
        f"| {_fmt(_geometric_mean(cmax_fes))} |"
    )
    lines.append(
        f"| AUC | {len(auc_fes)} "
        f"| {_pct_within(auc_fes)} "
        f"| {_fmt(_geometric_mean(auc_fes))} |"
    )
    lines.append(
        f"| t1/2 | {len(hl_fes)} "
        f"| {_pct_within(hl_fes)} "
        f"| {_fmt(_geometric_mean(hl_fes))} |"
    )
    lines.append(
        f"| **Overall** | {len(all_fes)} "
        f"| {_pct_within(all_fes)} "
        f"| {_fmt(_geometric_mean(all_fes))} |"
    )

    return "\n".join(lines)


def _section_ddi_validation(ddi_results: Sequence[DDIBenchmarkResult]) -> str:
    """Generate the DDI validation section."""
    lines = [
        "## 3. Drug-Drug Interaction Validation\n",
        "### 3.1 Predicted vs Observed AUCR\n",
    ]

    successful = [r for r in ddi_results if r.success]
    failed = [r for r in ddi_results if not r.success]

    lines.append(
        f"Total DDI cases: {len(ddi_results)} | "
        f"Successful: {len(successful)} | "
        f"Failed: {len(failed)}\n"
    )

    lines.append(
        "| Victim | Perpetrator | AUCR Pred | AUCR Obs | Fold Error | Pass |"
    )
    lines.append("|" + "|".join(["---"] * 6) + "|")

    for r in successful:
        pass_mark = "Yes" if r.within_tolerance else "No"
        lines.append(
            f"| {r.victim} | {r.perpetrator} "
            f"| {_fmt(r.aucr.predicted)} | {_fmt(r.aucr.observed)} "
            f"| {_fmt(r.aucr.fold_error)} | {pass_mark} |"
        )

    # AUCR summary
    aucr_fes = [r.aucr.fold_error for r in successful if r.aucr.is_scored]
    lines.append("\n### 3.2 DDI Aggregate Statistics\n")
    lines.append("| Metric | N Scored | Within 2-fold | GMFE |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| AUCR | {len(aucr_fes)} "
        f"| {_pct_within(aucr_fes)} "
        f"| {_fmt(_geometric_mean(aucr_fes))} |"
    )

    return "\n".join(lines)


def _section_mechanistic_invariants(tier4_results: Dict[str, Any]) -> str:
    """Report on Tier 4 mechanistic invariant tests."""
    lines = [
        "## 4. Mechanistic Invariants (Tier 4)\n",
        "These tests verify that the PBPK model satisfies fundamental "
        "pharmacokinetic principles independent of specific drug data.\n",
    ]

    if not tier4_results:
        lines.append("*No Tier 4 results provided.*\n")
        return "\n".join(lines)

    lines.append("| Invariant | Status | Details |")
    lines.append("|---|---|---|")

    for name, result in tier4_results.items():
        if isinstance(result, dict):
            status = "PASS" if result.get("passed", False) else "FAIL"
            detail = result.get("detail", "-")
        elif isinstance(result, bool):
            status = "PASS" if result else "FAIL"
            detail = "-"
        else:
            status = str(result)
            detail = "-"
        lines.append(f"| {name} | {status} | {detail} |")

    return "\n".join(lines)


def _section_sensitivity_analysis() -> str:
    """Placeholder for sensitivity analysis section."""
    return (
        "## 5. Sensitivity Analysis\n\n"
        "*Sensitivity analysis module is planned for a future release. "
        "Key parameters to evaluate:*\n\n"
        "- fu (fraction unbound in plasma): +/- 50%\n"
        "- CLint (intrinsic clearance): +/- 2-fold\n"
        "- Peff (effective permeability): +/- 50%\n"
        "- Kp (tissue partition coefficients): +/- 30%\n"
        "- Ki/IC50 (inhibition constants): +/- 3-fold\n\n"
        "*When implemented, tornado plots and spider plots will be "
        "generated for each benchmark drug.*"
    )


def _section_conclusions() -> str:
    """Generate conclusions section."""
    return (
        "## 6. Conclusions\n\n"
        "This report summarizes the verification and validation of the "
        "NeoGen PBPK model against published clinical pharmacokinetic "
        "data. The model's predictive performance should be assessed "
        "against the following regulatory acceptance criteria:\n\n"
        "- **Single-drug PK**: >= 60% of predictions within 2-fold of "
        "observed values (GMFE < 2.0)\n"
        "- **DDI predictions**: AUCR predictions within 2-fold of "
        "observed clinical DDI magnitude\n"
        "- **Mechanistic invariants**: all Tier 4 tests passing\n\n"
        "### Limitations\n\n"
        "- Population variability not yet characterized (single reference individual)\n"
        "- Enterohepatic recirculation not modeled\n"
        "- Limited transporter DDI coverage (OATP1B1 only in V1)\n"
        "- Pediatric scaling is scaffold-only\n\n"
        "---\n\n"
        "*Report generated by NeoGen PBPK Engine. "
        "For regulatory submissions, this report should be supplemented "
        "with full model development documentation, parameter justification, "
        "and application-specific verification.*"
    )


# ════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════


def generate_fda_report(
    pk_results: Sequence[BenchmarkResult],
    ddi_results: Sequence[DDIBenchmarkResult],
    tier4_results: Dict[str, Any],
    engine_version: str,
) -> str:
    """Generate FDA-format PBPK model verification report.

    Parameters
    ----------
    pk_results
        Sequence of single-drug BenchmarkResult from the benchmark runner.
    ddi_results
        Sequence of DDIBenchmarkResult from the DDI benchmark runner.
    tier4_results
        Dict mapping invariant names to pass/fail results. Each value
        can be a bool or a dict with 'passed' and 'detail' keys.
    engine_version
        Engine version string (e.g. "v1.2.0").

    Returns
    -------
    str
        Complete Markdown report suitable for FDA submission appendix.
    """
    sections = [
        _section_header(engine_version),
        _section_model_structure(),
        _section_pk_validation(pk_results),
        _section_ddi_validation(ddi_results),
        _section_mechanistic_invariants(tier4_results),
        _section_sensitivity_analysis(),
        _section_conclusions(),
    ]
    return "\n\n".join(sections)
