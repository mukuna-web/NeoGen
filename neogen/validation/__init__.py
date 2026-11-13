"""NeoGen V1 Validation Engine — tiered benchmarking against clinical data.

Public API
----------

    from neogen.validation import ValidationRunner

    report = ValidationRunner().run_all()
    if not report.overall_pass:
        print(report.to_markdown())

Tiers
-----

  * **Tier 1 — Single-drug PK.** Runs the V1 engine on a blind
    50-drug holdout cohort with known clinical CL, Vd, t½ and
    reports percent within 2-fold. Gate: ≥70%.
  * **Tier 2 — DDI AUCR.** Runs 5 clinical DDI pairs through the
    DDIOrchestrator + PBPKSolver and compares predicted AUCR vs
    observed. Gate: ≥60% within 2-fold.
  * **Tier 3 — Bioavailability.** Runs ~15 oral drugs in both oral
    and IV mode and compares F_empirical = (AUC_oral/Dose_oral) /
    (AUC_iv/Dose_iv) against clinical F. Gate: advisory 50% within
    ±20% of observed.
  * **Tier 4 — Mechanistic invariants.** Runs deterministic checks
    (IV F=1, mass balance, well-stirred algebra, non-negativity,
    post-Cmax monotonicity, etc.). Gate: 100% must pass.

All harnesses operate on the frozen V1 entities and return
immutable report objects serializable via
:class:`ValidationReport.to_json` / ``to_markdown``.
"""

from neogen.validation.cohort import (
    Tier1Case,
    Tier2Case,
    Tier3Case,
    data_validation_dir,
    load_tier1_blind_holdout,
    load_tier2_pairs,
    load_tier3_cohort,
)
from neogen.validation.metrics import (
    absolute_fold_error,
    average_fold_error,
    fold_error,
    geometric_mean_fold_error,
    percent_error,
    percent_within_fold,
    rmse_log10,
    summarise,
    within_n_fold,
    within_pct,
)
from neogen.validation.report import ValidationReport
from neogen.validation.runner import RunTimings, ValidationRunner
from neogen.validation.tier1 import Tier1CaseResult, Tier1Harness, Tier1Report
from neogen.validation.tier2 import Tier2CaseResult, Tier2Harness, Tier2Report
from neogen.validation.tier3 import Tier3CaseResult, Tier3Harness, Tier3Report
from neogen.validation.tier4 import InvariantResult, Tier4Harness, Tier4Report

# P6 benchmark harness (our V2 additions)
from neogen.validation.benchmark_case import BenchmarkCase, DDIBenchmarkCase
from neogen.validation.benchmark_runner import (
    BenchmarkResult,
    BenchmarkRunner,
    DDIBenchmarkResult,
    MetricScore,
)
from neogen.validation.datasets.ddi_benchmark_v1 import build_ddi_benchmark_v1
from neogen.validation.datasets.pk_benchmark_v1 import build_pk_benchmark_v1
from neogen.validation.report_writer import BenchmarkReport, RunMetadata
from neogen.validation.scorecard import DDIScorecard, Scorecard
from neogen.validation.sensitivity import (
    run_sensitivity_analysis,
    sensitivity_summary,
)
from neogen.validation.tier6_ddi_screening import (
    ScreeningMetrics,
    StaticDDIResult,
    compute_static_aucr,
    screen_drug_pairs,
)
from neogen.validation.profile_validator import (
    ProfileValidationResult,
    validate_profile,
    load_clinical_profiles,
)

__all__ = [
    # Cohort types
    "Tier1Case",
    "Tier2Case",
    "Tier3Case",
    "load_tier1_blind_holdout",
    "load_tier2_pairs",
    "load_tier3_cohort",
    "data_validation_dir",
    # Metrics
    "fold_error",
    "absolute_fold_error",
    "within_n_fold",
    "percent_error",
    "within_pct",
    "geometric_mean_fold_error",
    "average_fold_error",
    "percent_within_fold",
    "rmse_log10",
    "summarise",
    # Harnesses
    "Tier1Harness",
    "Tier2Harness",
    "Tier3Harness",
    "Tier4Harness",
    # Per-case results
    "Tier1CaseResult",
    "Tier2CaseResult",
    "Tier3CaseResult",
    "InvariantResult",
    # Per-tier reports
    "Tier1Report",
    "Tier2Report",
    "Tier3Report",
    "Tier4Report",
    # Top-level
    "ValidationReport",
    "ValidationRunner",
    "RunTimings",
    # P6 benchmark harness
    "BenchmarkCase",
    "DDIBenchmarkCase",
    "BenchmarkResult",
    "BenchmarkRunner",
    "DDIBenchmarkResult",
    "MetricScore",
    "build_ddi_benchmark_v1",
    "build_pk_benchmark_v1",
    "BenchmarkReport",
    "RunMetadata",
    "DDIScorecard",
    "Scorecard",
    # Phase 4 validation modules
    "run_sensitivity_analysis",
    "sensitivity_summary",
    "ScreeningMetrics",
    "StaticDDIResult",
    "compute_static_aucr",
    "screen_drug_pairs",
    "ProfileValidationResult",
    "validate_profile",
    "load_clinical_profiles",
]
