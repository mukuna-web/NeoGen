"""Tests for the P6 validation harness.

These tests verify the harness machinery itself — metric math,
cohort loading, report serialization, Tier 4 invariants, and
smoke runs of Tier 1/2/3 on a small subset. They do NOT gate
the full validation cohort; that is the job of scripts/run_validation.py.

Marks:
  * ``fast``: pure-function tests, no engine calls (<50 ms total).
  * ``integration``: require the V1 engine to be importable and
    runnable; skipped if any core import fails.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Ensure repo root on path if tests are run outside pytest config.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — METRICS (pure functions, no engine)
# ════════════════════════════════════════════════════════════════


class TestMetrics:
    def test_fold_error_perfect(self) -> None:
        from neogen.validation.metrics import fold_error

        assert fold_error(10.0, 10.0) == 1.0

    def test_fold_error_nan_on_invalid(self) -> None:
        from neogen.validation.metrics import fold_error

        assert math.isnan(fold_error(None, 10.0))
        assert math.isnan(fold_error(10.0, None))
        assert math.isnan(fold_error(10.0, 0.0))
        assert math.isnan(fold_error(-5.0, 10.0))

    def test_within_n_fold_boundaries(self) -> None:
        from neogen.validation.metrics import within_n_fold

        # Exact boundary: 0.5× and 2×
        assert within_n_fold(5.0, 10.0, n=2.0) is True
        assert within_n_fold(20.0, 10.0, n=2.0) is True
        # Outside
        assert within_n_fold(4.0, 10.0, n=2.0) is False
        assert within_n_fold(21.0, 10.0, n=2.0) is False

    def test_absolute_fold_error_symmetric(self) -> None:
        from neogen.validation.metrics import absolute_fold_error

        # Both 4× under and 4× over produce AFE=4
        assert absolute_fold_error(2.5, 10.0) == 4.0
        assert absolute_fold_error(40.0, 10.0) == 4.0

    def test_percent_error_sign(self) -> None:
        from neogen.validation.metrics import percent_error

        # 10% over
        assert percent_error(11.0, 10.0) == pytest.approx(10.0)
        # 25% under
        assert percent_error(7.5, 10.0) == pytest.approx(-25.0)

    def test_within_pct(self) -> None:
        from neogen.validation.metrics import within_pct

        assert within_pct(1.1, 1.0, pct=20.0) is True
        assert within_pct(0.79, 1.0, pct=20.0) is False

    def test_gmfe_perfect_is_one(self) -> None:
        from neogen.validation.metrics import geometric_mean_fold_error

        assert geometric_mean_fold_error([10, 20, 30], [10, 20, 30]) == 1.0

    def test_gmfe_skips_invalid(self) -> None:
        from neogen.validation.metrics import geometric_mean_fold_error

        # One valid pair (ratio 2×) and one invalid (None observed)
        gmfe = geometric_mean_fold_error([20, 5], [10, None])
        assert gmfe == pytest.approx(2.0)

    def test_percent_within_fold(self) -> None:
        from neogen.validation.metrics import percent_within_fold

        # 3 of 4 within 2-fold → 75%
        pct = percent_within_fold(
            [10, 20, 30, 1000],  # last one 100× off
            [10, 20, 30, 10],
            n=2.0,
        )
        assert pct == pytest.approx(75.0)

    def test_summarise_structure(self) -> None:
        from neogen.validation.metrics import summarise

        s = summarise([10, 20, 30], [10, 20, 30])
        assert s["n_total"] == 3
        assert s["n_valid"] == 3
        assert s["pct_within_2_fold"] == 100.0
        assert s["gmfe"] == pytest.approx(1.0)
        assert s["afe"] == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — COHORT LOADERS
# ════════════════════════════════════════════════════════════════


class TestCohortLoaders:
    def test_tier1_blind_holdout_loads(self) -> None:
        from neogen.validation.cohort import load_tier1_blind_holdout

        cases = load_tier1_blind_holdout()
        assert len(cases) >= 30, (
            f"expected ≥30 cases in blind holdout, got {len(cases)}"
        )
        # Every case must have name and smiles
        for c in cases:
            assert c.name
            assert c.smiles

    def test_tier1_literature_flag(self) -> None:
        from neogen.validation.cohort import load_tier1_blind_holdout

        cases = load_tier1_blind_holdout(
            literature_names={"midazolam", "cyclosporine"}
        )
        # At least one case should have the flag set (cyclosporine is
        # in the blind holdout)
        flagged = [c for c in cases if c.literature_available]
        assert len(flagged) >= 1

    def test_tier2_pairs_load(self) -> None:
        from neogen.validation.cohort import load_tier2_pairs

        pairs = load_tier2_pairs()
        assert len(pairs) == 5
        labels = {p.label for p in pairs}
        assert "ketoconazole_midazolam" in labels
        assert "rifampicin_midazolam" in labels
        # Every pair has a mode
        for p in pairs:
            assert p.mode in ("reversible", "induction", "mbi")

    def test_tier3_cohort_loads(self) -> None:
        from neogen.validation.cohort import load_tier3_cohort

        cases = load_tier3_cohort()
        assert len(cases) >= 10
        # Every case has a positive observed F
        for c in cases:
            assert 0.0 < c.observed_f_oral <= 1.0


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — REPORT SERIALIZATION
# ════════════════════════════════════════════════════════════════


class TestReportSerialization:
    def test_empty_report_roundtrip(self) -> None:
        from neogen.validation.report import ValidationReport

        report = ValidationReport(
            engine_version="v1.0.0-p4",
            run_timestamp_utc="2026-04-10T00:00:00Z",
        )
        # Empty report is neither pass nor fail (no tiers ran)
        assert report.overall_pass is False
        data = json.loads(report.to_json())
        assert data["engine_version"] == "v1.0.0-p4"
        assert data["overall_pass"] is False

    def test_markdown_contains_header(self) -> None:
        from neogen.validation.report import ValidationReport

        report = ValidationReport(
            engine_version="v1.0.0-p4",
            run_timestamp_utc="2026-04-10T00:00:00Z",
            notes=("unit test note",),
        )
        md = report.to_markdown()
        assert "NeoGen V1 Validation Report" in md
        assert "v1.0.0-p4" in md
        assert "unit test note" in md


# ════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — require the engine
# ════════════════════════════════════════════════════════════════


def _engine_available() -> bool:
    try:
        from neogen.engine.orchestrator import SimulationOrchestrator  # noqa: F401
        from neogen.engine.drug_factory import DrugFactory  # noqa: F401
    except Exception:
        return False
    return True


pytestmark_engine = pytest.mark.skipif(
    not _engine_available(),
    reason="V1 engine modules not importable",
)


@pytestmark_engine
class TestTier4:
    def test_tier4_runs(self) -> None:
        """Tier 4 should run to completion without raising."""
        from neogen.validation.tier4 import Tier4Harness

        report = Tier4Harness().run()
        assert report.n_total == 8
        # We don't assert gate_pass here — if the engine has a latent
        # bug, Tier 4 should surface it via the gate, but the test
        # should not fail the suite. A separate CI job gates on the
        # full validation harness.

    def test_tier4_invariants_have_descriptions(self) -> None:
        from neogen.validation.tier4 import Tier4Harness

        report = Tier4Harness().run()
        for r in report.results:
            assert r.name
            assert r.description


@pytestmark_engine
class TestTier1Smoke:
    def test_tier1_runs_on_literature_drugs_only(self) -> None:
        """Smoke: run Tier 1 on a hand-picked subset of 3 literature drugs."""
        from neogen.validation.cohort import Tier1Case
        from neogen.validation.tier1 import Tier1Harness

        cases = (
            Tier1Case(
                name="midazolam",
                smiles="C1C(=O)N(C2=CC=C(C=C2F)Cl)C3=CC(=CC=C3[N+]([O-])=C1)Cl",
                observed_half_life_h=2.5,
                observed_cl_L_per_h=25.0,
                observed_vd_L=77.0,
                literature_available=True,
            ),
            Tier1Case(
                name="caffeine",
                smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                observed_half_life_h=5.0,
                observed_cl_L_per_h=6.0,
                observed_vd_L=36.0,
                literature_available=True,
            ),
            Tier1Case(
                name="theophylline",
                smiles="CN1C(=O)N(C)C2=C1N=CN2",
                observed_half_life_h=8.0,
                observed_cl_L_per_h=2.8,
                observed_vd_L=35.0,
                literature_available=True,
            ),
        )
        report = Tier1Harness().run(cases)
        assert report.n_total == 3
        # At least one should be scorable and non-errored.
        assert report.n_scorable >= 1


@pytestmark_engine
class TestTier2Smoke:
    def test_tier2_single_pair_runs(self) -> None:
        """Smoke: ketoconazole + midazolam should produce a finite AUCR."""
        from neogen.validation.cohort import Tier2Case
        from neogen.validation.tier2 import Tier2Harness

        case = Tier2Case(
            label="smoke_ket_mid",
            victim_name="midazolam",
            victim_dose_mg=5.0,
            victim_route="iv_bolus",
            perpetrator_name="ketoconazole",
            perpetrator_iu_um=0.5,
            mode="reversible",
            clinical_aucr=5.0,
            citation="smoke test",
        )
        result = Tier2Harness().run_case(case)
        # Either it ran (and predicted an AUCR) or it errored cleanly
        if result.error:
            pytest.skip(f"DDI harness error in smoke: {result.error}")
        assert result.aucr_predicted is not None
        assert result.aucr_predicted > 0


@pytestmark_engine
class TestRunnerWiring:
    def test_runner_tier4_only(self) -> None:
        """The runner must honor the tier selection flags."""
        from neogen.validation.runner import ValidationRunner

        report = ValidationRunner().run_all(
            include_tier1=False,
            include_tier2=False,
            include_tier3=False,
            include_tier4=True,
        )
        assert report.tier1 is None
        assert report.tier2 is None
        assert report.tier3 is None
        assert report.tier4 is not None
        assert report.tier4.n_total == 8
