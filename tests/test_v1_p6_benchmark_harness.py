"""P6.1 — benchmark harness unit tests.

Covers the three new modules in ``neogen/validation/``:

* :mod:`benchmark_case` — frozen case constructors and validation.
* :mod:`benchmark_runner` — exception-safe execution of cases.
* :mod:`scorecard` — aggregation, gate logic, rendering.

These are *harness-level* tests — they do NOT validate the engine's
clinical accuracy. Benchmark datasets (P6.2/P6.3) and the acceptance
gate (P6.5) cover that. Here we only check that the harness itself
behaves deterministically and handles edge cases.
"""
from __future__ import annotations

import json
import math

import pytest

from neogen.validation import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkRunner,
    DDIBenchmarkCase,
    DDIScorecard,
    MetricScore,
    Scorecard,
)
from neogen.validation.benchmark_runner import _fold_error, _score_metric
from neogen.validation.scorecard import _geometric_mean


# ════════════════════════════════════════════════════════════════
# BENCHMARK CASE — construction & validation
# ════════════════════════════════════════════════════════════════


class TestBenchmarkCaseConstruction:
    def test_minimal_valid_case(self) -> None:
        c = BenchmarkCase(
            name="midazolam-iv-5mg",
            drug_name="midazolam",
            dose_mg=5.0,
            route="IV",
            source="Olkkola 1994",
            observed_cmax_mg_per_L=0.08,
        )
        assert c.route == "IV"
        assert c.schedule_kind == "single"
        assert c.tolerance_fold == 2.0
        assert c.has_any_reference is True

    def test_route_canonicalization(self) -> None:
        c = BenchmarkCase(
            name="x",
            drug_name="midazolam",
            dose_mg=5.0,
            route="iv",
            source="s",
            observed_cmax_mg_per_L=0.1,
        )
        assert c.route == "IV"

    def test_bad_route_rejected(self) -> None:
        with pytest.raises(ValueError, match="route"):
            BenchmarkCase(
                name="x",
                drug_name="midazolam",
                dose_mg=5.0,
                route="subcutaneous",
                source="s",
            )

    def test_tolerance_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="tolerance_fold"):
            BenchmarkCase(
                name="x",
                drug_name="midazolam",
                dose_mg=5.0,
                route="IV",
                source="s",
                tolerance_fold=0.5,
            )

    def test_blank_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source"):
            BenchmarkCase(
                name="x",
                drug_name="midazolam",
                dose_mg=5.0,
                route="IV",
                source="   ",
            )

    def test_multi_dose_requires_interval_and_duration(self) -> None:
        with pytest.raises(ValueError, match="multi-dose"):
            BenchmarkCase(
                name="x",
                drug_name="midazolam",
                dose_mg=5.0,
                route="ORAL",
                source="s",
                schedule_kind="multi",
            )

    def test_has_any_reference_false_when_all_none(self) -> None:
        c = BenchmarkCase(
            name="x",
            drug_name="midazolam",
            dose_mg=5.0,
            route="IV",
            source="s",
        )
        assert c.has_any_reference is False

    def test_ddi_case_validation(self) -> None:
        d = DDIBenchmarkCase(
            name="keto+midaz",
            victim_drug_name="midazolam",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=5.0,
            victim_route="iv",
            perpetrator_steady_state_conc_um=1.0,
            observed_aucr=10.0,
            source="Olkkola 1994",
        )
        assert d.victim_route == "IV"
        assert d.perpetrator_duration_on_h == 72.0

    def test_ddi_case_rejects_nonpositive_aucr(self) -> None:
        with pytest.raises(ValueError, match="observed_aucr"):
            DDIBenchmarkCase(
                name="x",
                victim_drug_name="midazolam",
                perpetrator_drug_name="ketoconazole",
                victim_dose_mg=5.0,
                victim_route="IV",
                perpetrator_steady_state_conc_um=1.0,
                observed_aucr=0.0,
                source="s",
            )


# ════════════════════════════════════════════════════════════════
# HELPERS — fold-error + geometric mean
# ════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_fold_error_symmetric(self) -> None:
        assert _fold_error(10.0, 5.0) == pytest.approx(2.0)
        assert _fold_error(5.0, 10.0) == pytest.approx(2.0)

    def test_fold_error_identity(self) -> None:
        assert _fold_error(1.0, 1.0) == pytest.approx(1.0)

    def test_fold_error_zero_is_infinite(self) -> None:
        assert _fold_error(0.0, 1.0) == math.inf
        assert _fold_error(1.0, 0.0) == math.inf

    def test_geometric_mean_identity(self) -> None:
        assert _geometric_mean([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_geometric_mean_of_two_and_half(self) -> None:
        # GM(2, 0.5) = 1.0
        assert _geometric_mean([2.0, 0.5]) == pytest.approx(1.0)

    def test_geometric_mean_empty_is_nan(self) -> None:
        assert math.isnan(_geometric_mean([]))

    def test_geometric_mean_skips_nonpositive(self) -> None:
        assert _geometric_mean([2.0, 0.0, -1.0]) == pytest.approx(2.0)


# ════════════════════════════════════════════════════════════════
# METRIC SCORE
# ════════════════════════════════════════════════════════════════


class TestMetricScore:
    def test_scored_metric_within_tolerance(self) -> None:
        m = _score_metric("Cmax", predicted=1.5, observed=1.0, tolerance_fold=2.0)
        assert m.is_scored
        assert m.within_tolerance is True
        assert m.fold_error == pytest.approx(1.5)

    def test_scored_metric_outside_tolerance(self) -> None:
        m = _score_metric("AUC", predicted=10.0, observed=1.0, tolerance_fold=2.0)
        assert m.within_tolerance is False
        assert m.fold_error == pytest.approx(10.0)

    def test_missing_predicted_not_scored(self) -> None:
        m = _score_metric("t½", predicted=None, observed=2.0, tolerance_fold=2.0)
        assert not m.is_scored
        assert m.within_tolerance is None

    def test_missing_observed_not_scored(self) -> None:
        m = _score_metric("Cmax", predicted=1.0, observed=None, tolerance_fold=2.0)
        assert not m.is_scored


# ════════════════════════════════════════════════════════════════
# RUNNER — exception safety + end-to-end smoke
# ════════════════════════════════════════════════════════════════


class TestBenchmarkRunner:
    def test_unknown_drug_returns_failed_result(self) -> None:
        case = BenchmarkCase(
            name="nonsense",
            drug_name="unobtainium",
            dose_mg=10.0,
            route="IV",
            source="imaginary",
            observed_cmax_mg_per_L=1.0,
        )
        runner = BenchmarkRunner()
        result = runner.run_case(case)
        assert isinstance(result, BenchmarkResult)
        assert result.success is False
        assert result.error_message is not None
        # All metrics are unscored on failure
        assert not result.cmax.is_scored
        assert not result.auc.is_scored
        assert not result.half_life.is_scored

    def test_case_without_reference_is_skipped(self) -> None:
        case = BenchmarkCase(
            name="bare",
            drug_name="midazolam",
            dose_mg=5.0,
            route="IV",
            source="s",
        )
        runner = BenchmarkRunner()
        result = runner.run_case(case)
        assert result.success is False
        assert result.error_message == "no reference data"

    def test_midazolam_iv_runs_successfully(self) -> None:
        case = BenchmarkCase(
            name="midazolam-iv-5mg",
            drug_name="midazolam",
            dose_mg=5.0,
            route="IV",
            source="Olkkola 1994",
            observed_cmax_mg_per_L=0.08,
            observed_auc_mg_h_per_L=0.22,
            observed_half_life_h=2.5,
        )
        runner = BenchmarkRunner()
        result = runner.run_case(case)
        assert result.success is True
        assert result.engine_version.startswith("v1")
        assert result.cmax.is_scored
        assert result.auc.is_scored
        assert result.half_life.is_scored
        # Every fold error is ≥ 1.0 by construction
        for m in result.scored_metrics:
            assert m.fold_error >= 1.0

    def test_run_all_is_tuple(self) -> None:
        cases = [
            BenchmarkCase(
                name="midazolam-iv-5mg",
                drug_name="midazolam",
                dose_mg=5.0,
                route="IV",
                source="Olkkola 1994",
                observed_cmax_mg_per_L=0.08,
            ),
        ]
        runner = BenchmarkRunner()
        results = runner.run_all(cases)
        assert isinstance(results, tuple)
        assert len(results) == 1


# ════════════════════════════════════════════════════════════════
# SCORECARD — aggregation, gate, rendering
# ════════════════════════════════════════════════════════════════


def _make_result(
    *,
    name: str = "case",
    drug: str = "midazolam",
    cmax_pred: float = 1.0,
    cmax_obs: float = 1.0,
    auc_pred: float = 1.0,
    auc_obs: float = 1.0,
    hl_pred: float = 1.0,
    hl_obs: float = 1.0,
    success: bool = True,
) -> BenchmarkResult:
    return BenchmarkResult(
        case_name=name,
        drug_name=drug,
        success=success,
        engine_version="v1.0.0-test",
        cmax=_score_metric("Cmax", cmax_pred, cmax_obs, 2.0),
        auc=_score_metric("AUC", auc_pred, auc_obs, 2.0),
        half_life=_score_metric("t½", hl_pred, hl_obs, 2.0),
        error_message=None if success else "boom",
    )


class TestScorecard:
    def test_perfect_match_passes_gate(self) -> None:
        card = Scorecard.build([_make_result(), _make_result(name="b")])
        assert card.n_cases == 2
        assert card.total_scored == 6
        assert card.total_within_tolerance == 6
        assert card.overall_pct_within_tolerance == pytest.approx(100.0)
        assert card.overall_gmfe == pytest.approx(1.0)
        assert card.passes_v1_gate is True

    def test_all_outside_tolerance_fails_gate(self) -> None:
        card = Scorecard.build(
            [
                _make_result(cmax_pred=5.0, auc_pred=5.0, hl_pred=5.0),
            ]
        )
        assert card.total_within_tolerance == 0
        assert card.passes_v1_gate is False

    def test_v1_gate_partial_pass(self) -> None:
        results = [
            _make_result(name="a"),  # 3 within
            _make_result(name="b"),  # 3 within
            _make_result(name="c", cmax_pred=5.0),  # 2 within, 1 out
            _make_result(
                name="d",
                cmax_pred=5.0,
                auc_pred=5.0,
                hl_pred=5.0,
            ),  # 0 within, 3 out
        ]
        card = Scorecard.build(results)
        # 8 within / 12 scored = 66.7% → passes gate
        assert card.overall_pct_within_tolerance == pytest.approx(
            100.0 * 8 / 12
        )
        assert card.passes_v1_gate is True

    def test_v1_gate_just_below_fails(self) -> None:
        # 1 case, 1 within, 2 out → 33% → below 60% gate
        card = Scorecard.build(
            [_make_result(cmax_pred=5.0, auc_pred=5.0)]
        )
        assert card.overall_pct_within_tolerance == pytest.approx(
            100.0 / 3
        )
        assert card.passes_v1_gate is False

    def test_failed_cases_contribute_zero_scored_metrics(self) -> None:
        ok = _make_result(name="ok")
        # A "real" failure has unscored metric slots (None predicted/observed).
        unscored = MetricScore(
            name="Cmax",
            observed=None,
            predicted=None,
            fold_error=None,
            within_tolerance=None,
        )
        bad = BenchmarkResult(
            case_name="bad",
            drug_name="midazolam",
            success=False,
            engine_version="v1.0.0-test",
            cmax=unscored,
            auc=MetricScore(
                name="AUC",
                observed=None,
                predicted=None,
                fold_error=None,
                within_tolerance=None,
            ),
            half_life=MetricScore(
                name="t½",
                observed=None,
                predicted=None,
                fold_error=None,
                within_tolerance=None,
            ),
            error_message="boom",
        )
        card = Scorecard.build([ok, bad])
        assert card.n_cases == 2
        assert card.n_successful == 1
        # Only "ok" contributes scored metrics
        assert card.total_scored == 3

    def test_summary_lines_render(self) -> None:
        card = Scorecard.build([_make_result()])
        lines = card.to_summary_lines()
        text = "\n".join(lines)
        assert "NeoGen V1 PK Benchmark Scorecard" in text
        assert "PASS" in text
        assert "Cmax" in text

    def test_markdown_table_shape(self) -> None:
        card = Scorecard.build([_make_result(name="a"), _make_result(name="b")])
        md = card.to_markdown_table()
        lines = md.splitlines()
        # header + separator + 2 rows
        assert len(lines) == 4
        assert lines[0].startswith("| Case")
        assert lines[1].startswith("|---")

    def test_json_round_trip(self) -> None:
        card = Scorecard.build([_make_result()])
        payload = json.loads(card.to_json())
        assert payload["n_cases"] == 1
        assert payload["passes_v1_gate"] is True
        assert payload["per_metric"]["cmax"]["n_scored"] == 1
        assert payload["cases"][0]["drug"] == "midazolam"

    def test_empty_scorecard_gate_fails(self) -> None:
        card = Scorecard.build([])
        assert card.n_cases == 0
        assert math.isnan(card.overall_pct_within_tolerance)
        assert card.passes_v1_gate is False


# ════════════════════════════════════════════════════════════════
# DDI SCORECARD
# ════════════════════════════════════════════════════════════════


def _make_ddi_result(
    *,
    name: str = "keto+midaz",
    aucr_pred: float = 10.0,
    aucr_obs: float = 10.0,
    success: bool = True,
):
    from neogen.validation.benchmark_runner import DDIBenchmarkResult

    return DDIBenchmarkResult(
        case_name=name,
        victim="midazolam",
        perpetrator="ketoconazole",
        success=success,
        engine_version="v1.0.0-test",
        aucr=_score_metric("AUCR", aucr_pred, aucr_obs, 2.0),
        error_message=None if success else "boom",
    )


class TestDDIScorecard:
    def test_perfect_ddi_pass(self) -> None:
        card = DDIScorecard.build([_make_ddi_result()])
        assert card.passes_v1_gate is True
        assert card.aucr_summary.pct_within_tolerance == pytest.approx(100.0)

    def test_poor_ddi_fail(self) -> None:
        card = DDIScorecard.build(
            [_make_ddi_result(aucr_pred=50.0, aucr_obs=1.0)]
        )
        assert card.passes_v1_gate is False

    def test_ddi_markdown_shape(self) -> None:
        card = DDIScorecard.build([_make_ddi_result(), _make_ddi_result(name="b")])
        md = card.to_markdown_table()
        lines = md.splitlines()
        assert len(lines) == 4
        assert "AUCR" in lines[0] or "Predicted" in lines[0]

    def test_ddi_summary_pass_line(self) -> None:
        card = DDIScorecard.build([_make_ddi_result()])
        text = "\n".join(card.to_summary_lines())
        assert "DDI Benchmark Scorecard" in text
        assert "PASS" in text
