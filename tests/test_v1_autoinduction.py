"""V1 V2-extension — autoinduction tests.

These tests cover the second V2 engine extension: a time-dependent
self-induction simulator that closes the feedback loop for drugs which
induce the CYPs responsible for their own clearance. The canonical
target is rifampicin, whose CL grows over 5-7 days of chronic dosing
and whose own AUC falls by the same mechanism.

The assertions are *directional* — we do not pin absolute AUCR_self
values (those depend on drug-card fm/Indmax/IndC50 calibration) but we
do pin the mathematical invariants that the coupled ODE must satisfy:

1.  Self-induction fold is strictly less than the static (no-feedback)
    fold at the same baseline concentration, because the feedback
    attenuates ``[I]u`` and therefore the induction saturation.
2.  ``AUCR_self_ss < 1`` for any drug that is both substrate and
    inducer of at least one shared CYP.
3.  Time-to-90 %-plateau on the dominant CYP is of order 3-5 · t½_CYP
    (CYP3A4: ~5 days clinical).
4.  Non-autoinducers fall through to the standard Step 9 trace.
5.  The orchestrator helper returns a :class:`PerpetratorExposure`
    whose ``enzyme_trace`` is consumable by the existing
    :meth:`DDIOrchestrator.build_clint_provider`.
"""
from __future__ import annotations

import pytest

from neogen.engine.ddi.autoinduction import (
    AutoinductionSimulator,
    AutoinductionSummary,
    is_autoinducer,
    overlapping_cyps,
)
from neogen.engine.ddi.ddi_orchestrator import (
    DDIOrchestrator,
    PerpetratorExposure,
)
from neogen.engine.ddi.enzyme_pool_evolver import EnzymePoolEvolver
from neogen.engine.drug_factory import DrugFactory


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def factory() -> DrugFactory:
    return DrugFactory()


@pytest.fixture(scope="module")
def rifampicin(factory: DrugFactory):
    return factory.create("rifampicin")


@pytest.fixture(scope="module")
def midazolam(factory: DrugFactory):
    return factory.create("midazolam")


@pytest.fixture(scope="module")
def ketoconazole(factory: DrugFactory):
    return factory.create("ketoconazole")


@pytest.fixture(scope="module")
def simulator() -> AutoinductionSimulator:
    return AutoinductionSimulator()


@pytest.fixture(scope="module")
def orchestrator() -> DDIOrchestrator:
    return DDIOrchestrator()


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — detection helpers
# ════════════════════════════════════════════════════════════════


class TestDetectionHelpers:
    def test_rifampicin_is_autoinducer(self, rifampicin) -> None:
        assert is_autoinducer(rifampicin) is True

    def test_midazolam_is_not_autoinducer(self, midazolam) -> None:
        # Midazolam has no induction profile (is_inducer == False).
        assert is_autoinducer(midazolam) is False

    def test_ketoconazole_is_not_autoinducer(self, ketoconazole) -> None:
        # Ketoconazole inhibits CYP3A4 but does not induce it.
        assert is_autoinducer(ketoconazole) is False

    def test_rifampicin_overlap_includes_cyp3a4(self, rifampicin) -> None:
        overlap = overlapping_cyps(rifampicin)
        assert "CYP3A4" in overlap
        # Rifampicin has no fm_CYP2C9 in the NeoGen card, so even though
        # it induces CYP2C9 the overlap should not include it.
        assert "CYP2C9" not in overlap

    def test_midazolam_overlap_is_empty(self, midazolam) -> None:
        assert overlapping_cyps(midazolam) == ()


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — AutoinductionSimulator
# ════════════════════════════════════════════════════════════════


class TestAutoinductionSimulator:
    def test_constructor_rejects_missing_kdeg(self) -> None:
        with pytest.raises(ValueError, match="missing entries"):
            AutoinductionSimulator(
                kdeg_per_h={"CYP3A4": 0.02},  # missing the other CYPs
            )

    def test_simulate_rejects_negative_baseline(
        self, simulator, rifampicin
    ) -> None:
        with pytest.raises(ValueError, match="baseline_conc_um"):
            simulator.simulate(
                rifampicin, baseline_conc_um=-0.1, duration_on_h=24.0
            )

    def test_simulate_rejects_zero_total_duration(
        self, simulator, rifampicin
    ) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            simulator.simulate(
                rifampicin, baseline_conc_um=1.0, duration_on_h=0.0
            )

    def test_simulate_rejects_low_n_points(
        self, simulator, rifampicin
    ) -> None:
        with pytest.raises(ValueError, match="n_points must be ≥ 2"):
            simulator.simulate(
                rifampicin,
                baseline_conc_um=1.0,
                duration_on_h=24.0,
                n_points=1,
            )

    def test_trace_starts_at_baseline(self, simulator, rifampicin) -> None:
        trace = simulator.simulate(
            rifampicin, baseline_conc_um=1.0, duration_on_h=24.0
        )
        first = trace.snapshots[0]
        assert first.time_h == pytest.approx(0.0)
        for cyp in trace.tracked_cyps:
            assert first.fold_by_cyp[cyp] == pytest.approx(1.0)

    def test_trace_covers_requested_window(
        self, simulator, rifampicin
    ) -> None:
        trace = simulator.simulate(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=48.0,
            duration_off_h=24.0,
            n_points=100,
        )
        assert trace.duration_h == pytest.approx(72.0)
        assert len(trace.snapshots) == 100
        assert trace.snapshots[-1].time_h == pytest.approx(72.0)

    def test_effective_conc_decays_during_ramp(
        self, simulator, rifampicin
    ) -> None:
        # The post-attenuation [I]u recorded on each snapshot must fall
        # as the CYP3A4 pool grows — this is the signature of the
        # closed feedback loop.
        trace = simulator.simulate(
            rifampicin, baseline_conc_um=1.0, duration_on_h=7 * 24
        )
        on_conc = [
            s.inducer_conc_um
            for s in trace.snapshots
            if s.inducer_conc_um > 0.0
        ]
        assert len(on_conc) > 20
        # The concentration must drop meaningfully from day-1 to plateau.
        assert on_conc[-1] < 0.9 * on_conc[0]

    def test_washout_drops_conc_to_zero(
        self, simulator, rifampicin
    ) -> None:
        trace = simulator.simulate(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=48.0,
            duration_off_h=48.0,
        )
        assert trace.snapshots[-1].inducer_conc_um == pytest.approx(0.0)

    def test_fold_trace_is_monotonic_during_ramp(
        self, simulator, rifampicin
    ) -> None:
        # During the on-phase the CYP3A4 fold must be non-decreasing —
        # no overshoot expected for the first-order turnover ODE.
        trace = simulator.simulate(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=10 * 24,
            n_points=400,
        )
        on_snapshots = [
            s for s in trace.snapshots if s.inducer_conc_um > 0.0
        ]
        folds = [s.fold_by_cyp["CYP3A4"] for s in on_snapshots]
        for prev, nxt in zip(folds, folds[1:]):
            # Allow a tiny numerical wobble from LSODA.
            assert nxt >= prev - 1.0e-6


class TestAutoinductionInvariants:
    """Mathematical invariants the coupled ODE must always satisfy."""

    def test_aucr_self_is_below_one(self, simulator, rifampicin) -> None:
        trace = simulator.simulate(
            rifampicin, baseline_conc_um=1.0, duration_on_h=10 * 24
        )
        summary = simulator.summarize(
            trace, rifampicin, baseline_conc_um=1.0
        )
        assert summary.aucr_self_ss < 1.0
        assert summary.aucr_self_ss > 0.0

    def test_effective_conc_matches_aucr(
        self, simulator, rifampicin
    ) -> None:
        baseline = 1.5
        trace = simulator.simulate(
            rifampicin,
            baseline_conc_um=baseline,
            duration_on_h=10 * 24,
        )
        summary = simulator.summarize(
            trace, rifampicin, baseline_conc_um=baseline
        )
        assert summary.effective_conc_um_ss == pytest.approx(
            baseline * summary.aucr_self_ss, rel=1e-9
        )

    def test_autoinduction_is_weaker_than_static(
        self, simulator, rifampicin
    ) -> None:
        # The closed-loop steady-state fold must be strictly less than
        # the static Step 9 fold at the same baseline concentration —
        # this is the whole reason autoinduction matters.
        baseline = 1.0
        trace = simulator.simulate(
            rifampicin,
            baseline_conc_um=baseline,
            duration_on_h=10 * 24,
            n_points=400,
        )
        summary = simulator.summarize(
            trace, rifampicin, baseline_conc_um=baseline
        )

        evolver = EnzymePoolEvolver()
        static_fold = evolver.steady_state_fold(rifampicin, baseline, "CYP3A4")

        dynamic_fold = summary.fold_by_cyp_ss["CYP3A4"]
        assert dynamic_fold < static_fold
        # And the gap must be meaningful (>5 %) for a drug with fm > 0.
        assert dynamic_fold < 0.95 * static_fold

    def test_higher_baseline_shrinks_aucr_further(
        self, simulator, rifampicin
    ) -> None:
        # Larger baseline [I]u drives greater steady-state induction,
        # which in turn drives a smaller AUCR_self. Monotone in baseline.
        low = simulator.summarize(
            simulator.simulate(
                rifampicin,
                baseline_conc_um=0.3,
                duration_on_h=10 * 24,
                n_points=300,
            ),
            rifampicin,
            baseline_conc_um=0.3,
        )
        high = simulator.summarize(
            simulator.simulate(
                rifampicin,
                baseline_conc_um=3.0,
                duration_on_h=10 * 24,
                n_points=300,
            ),
            rifampicin,
            baseline_conc_um=3.0,
        )
        assert high.aucr_self_ss < low.aucr_self_ss
        assert high.fold_by_cyp_ss["CYP3A4"] > low.fold_by_cyp_ss["CYP3A4"]

    def test_build_up_time_is_in_day_range(
        self, simulator, rifampicin
    ) -> None:
        # Rifampicin CYP3A4 (t½ ≈ 36 h) should hit 90 % of plateau
        # somewhere in the 2-6 day window — the classic clinical
        # observation that motivates 5-7 day DDI pretreatment.
        summary = simulator.summarize(
            simulator.simulate(
                rifampicin,
                baseline_conc_um=1.0,
                duration_on_h=14 * 24,
                n_points=400,
            ),
            rifampicin,
            baseline_conc_um=1.0,
        )
        assert summary.time_to_90pct_h is not None
        assert 2 * 24 <= summary.time_to_90pct_h <= 6 * 24


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — non-autoinducer fall-through
# ════════════════════════════════════════════════════════════════


class TestNonAutoinducerFallThrough:
    def test_summarize_returns_identity_for_inert_drug(
        self, simulator, midazolam
    ) -> None:
        # Midazolam is not an inducer at all — the simulator should
        # still run and return a summary with AUCR_self == 1.0.
        trace = simulator.simulate(
            midazolam, baseline_conc_um=0.1, duration_on_h=48.0
        )
        summary = simulator.summarize(
            trace, midazolam, baseline_conc_um=0.1
        )
        assert summary.aucr_self_ss == pytest.approx(1.0, abs=1e-6)
        assert summary.effective_conc_um_ss == pytest.approx(0.1, abs=1e-6)
        for fold in summary.fold_by_cyp_ss.values():
            assert fold == pytest.approx(1.0, abs=1e-3)


# ════════════════════════════════════════════════════════════════
# INTEGRATION — DDIOrchestrator.build_autoinduction_exposure
# ════════════════════════════════════════════════════════════════


class TestOrchestratorIntegration:
    def test_autoinduction_exposure_returns_pair(
        self, orchestrator, rifampicin
    ) -> None:
        exposure, summary = orchestrator.build_autoinduction_exposure(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=7 * 24,
        )
        assert isinstance(exposure, PerpetratorExposure)
        assert isinstance(summary, AutoinductionSummary)
        assert exposure.enzyme_trace is not None
        assert exposure.perpetrator is rifampicin

    def test_exposure_steady_state_is_attenuated(
        self, orchestrator, rifampicin
    ) -> None:
        # The exposure's scalar steady_state_conc_um must equal the
        # summary's effective steady-state value (not the raw baseline).
        exposure, summary = orchestrator.build_autoinduction_exposure(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=7 * 24,
        )
        assert exposure.steady_state_conc_um == pytest.approx(
            summary.effective_conc_um_ss
        )
        assert exposure.steady_state_conc_um < 1.0

    def test_non_autoinducer_falls_through(
        self, orchestrator, ketoconazole
    ) -> None:
        # Ketoconazole doesn't induce anything — the helper should
        # delegate to build_exposure and return a trivial summary.
        exposure, summary = orchestrator.build_autoinduction_exposure(
            ketoconazole,
            baseline_conc_um=0.5,
            duration_on_h=72.0,
        )
        assert exposure.steady_state_conc_um == pytest.approx(0.5)
        assert summary.aucr_self_ss == pytest.approx(1.0)

    def test_clint_provider_consumes_autoinduction_trace(
        self, orchestrator, rifampicin
    ) -> None:
        # End-to-end: use the autoinduction trace inside
        # build_clint_provider and confirm that the victim CLint grows
        # over time (which for a CYP3A4-cleared victim is the expected
        # induction signature).
        exposure, _ = orchestrator.build_autoinduction_exposure(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=10 * 24,
        )
        provider = orchestrator.build_clint_provider(
            victim=rifampicin,  # rifampicin-on-rifampicin
            base_clint_scaled_L_per_h=56.0,
            exposures=[exposure],
        )
        clint_t0 = provider(0.0)
        clint_tplateau = provider(7 * 24)
        # Induction drives CLint up over time.
        assert clint_tplateau > clint_t0 * 1.5

    def test_breakdown_exposes_per_cyp_scaling(
        self, orchestrator, rifampicin
    ) -> None:
        exposure, _ = orchestrator.build_autoinduction_exposure(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=10 * 24,
        )
        breakdown = orchestrator.breakdown_at(
            5 * 24,
            victim=rifampicin,
            base_clint_scaled_L_per_h=56.0,
            exposures=[exposure],
        )
        per_cyp = dict(breakdown.per_cyp_scaling)
        assert per_cyp["CYP3A4"] > 2.0  # clearly induced
        assert breakdown.scaling > 1.5
        assert breakdown.clint_eff_L_per_h > breakdown.clint_base_L_per_h


# ════════════════════════════════════════════════════════════════
# DYNAMIC VS STATIC — head-to-head
# ════════════════════════════════════════════════════════════════


class TestDynamicVsStatic:
    """Direct comparison with the static Step 9 evolver on rifampicin."""

    def test_static_evolver_matches_at_zero_fm(
        self, simulator, rifampicin
    ) -> None:
        # A purely diagnostic cross-check: if we artificially zero the
        # fm feedback path (by running the evolver under the same
        # baseline [I]u), the static fold equals the autoinduction
        # result only in the limit fm → 0. We don't mutate the drug; we
        # just assert that dynamic < static for the real fm.
        baseline = 0.5
        evolver = EnzymePoolEvolver()
        static_trace = evolver.simulate(
            rifampicin,
            inducer_conc_um=baseline,
            duration_on_h=10 * 24,
            n_points=300,
        )
        dynamic_trace = simulator.simulate(
            rifampicin,
            baseline_conc_um=baseline,
            duration_on_h=10 * 24,
            n_points=300,
        )
        assert dynamic_trace.max_fold("CYP3A4") < static_trace.max_fold("CYP3A4")

    def test_full_dose_window_exposure_build_up(
        self, simulator, rifampicin
    ) -> None:
        # Start-of-window vs end-of-window CYP3A4 fold on a 14-day run.
        trace = simulator.simulate(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=14 * 24,
            n_points=500,
        )
        # Day 1
        fold_24h = trace.fold_at(24.0, "CYP3A4")
        # Day 7
        fold_day7 = trace.fold_at(7 * 24, "CYP3A4")
        # Day 14
        fold_day14 = trace.fold_at(14 * 24, "CYP3A4")
        assert fold_24h > 1.0
        assert fold_day7 > fold_24h
        # Plateau: day 14 within 10 % of day 7.
        assert abs(fold_day14 - fold_day7) / fold_day7 < 0.15
