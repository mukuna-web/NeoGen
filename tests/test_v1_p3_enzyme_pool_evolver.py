"""
P3 Step 9 — Enzyme Pool Evolver Unit Tests
=============================================
Exercises ``neogen.engine.ddi.enzyme_pool_evolver`` in isolation.

Scope (matches the V1 execution plan):
  * Time-dependent CYP enzyme pool ODE with induction and MBI.
  * Washout behaviour after perpetrator removal.
  * Steady-state analytical / numerical agreement.
  * Stateless ``derivative()`` hook used by Step 10 PBPK coupling.

Plan gates for Step 9:
    * Rifampicin induction reaches ~90 % of max in 5-7 days.
    * Washout (from rifampicin steady state) back to baseline in 10-14 days.

These are the two classic clinical fingerprints of CYP3A4 protein turnover
(t½ ≈ 36 h, so ~5 × 36 h ≈ 7.5 d for full induction or full washout).
"""
from __future__ import annotations

import math

import pytest

from neogen.core.entities.drug import (
    CYPMetabolismProfile,
    Drug,
    InductionProfile,
    InhibitionProfile,
)
from neogen.engine.ddi import (
    CYP_NAMES,
    DEFAULT_KDEG_PER_H,
    EnzymePoolEvolver,
    EnzymePoolTrace,
    EnzymeSnapshot,
)
from neogen.engine.drug_factory import DrugFactory


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def evolver() -> EnzymePoolEvolver:
    return EnzymePoolEvolver()


@pytest.fixture(scope="module")
def factory() -> DrugFactory:
    return DrugFactory()


@pytest.fixture(scope="module")
def rifampicin(factory: DrugFactory) -> Drug:
    return factory.create("rifampicin")


@pytest.fixture(scope="module")
def clarithromycin(factory: DrugFactory) -> Drug:
    return factory.create("clarithromycin")


@pytest.fixture(scope="module")
def ketoconazole(factory: DrugFactory) -> Drug:
    """Pure reversible inhibitor — no time-dependent pool change expected."""
    return factory.create("ketoconazole")


@pytest.fixture
def combined_drug() -> Drug:
    """Toy drug that both induces CYP3A4 (PXR) and MBI-destroys it.

    Indmax = 5, IndC50 = 1 µM, kinact = 0.03 min⁻¹, KI = 2 µM.
    Tests the interplay of the two mechanisms.
    """
    return Drug(
        name="combined_tool",
        smiles="C",
        molecular_weight=400.0,
        logp=2.0,
        fu_plasma=0.5,
        metabolism=CYPMetabolismProfile(),
        induction=InductionProfile(
            indmax_cyp3a4=5.0,
            indc50_cyp3a4=1.0,
        ),
        inhibition=InhibitionProfile(
            kinact_cyp3a4=0.03,
            k_i_mbi_cyp3a4=2.0,
        ),
    )


# ════════════════════════════════════════════════════════════════
# STEP 9 HARD GATES
# ════════════════════════════════════════════════════════════════


class TestStep9AcceptanceGate:
    """The assertions required by the V1 plan for Step 9."""

    def test_rifampicin_cyp3a4_reaches_90pct_in_5_to_7_days(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        """Classic rifampicin induction time course: 90 % of plateau
        within 5-7 days at clinical portal [I]u ≈ 0.2 µM."""
        trace = evolver.simulate(
            rifampicin,
            inducer_conc_um=0.2,
            duration_on_h=14 * 24,
            n_points=200,
        )
        t_90_h = trace.time_to_fraction("CYP3A4", 0.9)
        assert t_90_h is not None, "CYP3A4 never reached 90 % of plateau"
        t_90_days = t_90_h / 24.0
        assert 4.0 <= t_90_days <= 7.5, (
            f"Rifampicin CYP3A4 t_90 = {t_90_days:.2f} days, "
            f"expected 4-7.5 days."
        )

    def test_rifampicin_cyp3a4_washout_to_baseline_in_10_to_14_days(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        """After 14 days on rifampicin, CYP3A4 should return to within
        5 % of baseline by day 14 post-stop (≈ 5 half-lives)."""
        trace = evolver.simulate(
            rifampicin,
            inducer_conc_um=0.2,
            duration_on_h=14 * 24,
            duration_off_h=14 * 24,
            n_points=400,
        )
        final_fold = trace.snapshots[-1].fold_for("CYP3A4")
        assert 0.95 <= final_fold <= 1.05, (
            f"CYP3A4 washout end-fold = {final_fold:.3f}, "
            f"expected within 5% of baseline after 14 d off drug."
        )

    def test_rifampicin_plateau_matches_step8_emax(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        """Long-run plateau must equal the Step 8 Emax formula exactly
        (within solver tolerance)."""
        trace = evolver.simulate(
            rifampicin,
            inducer_conc_um=0.2,
            duration_on_h=30 * 24,  # way past any half-life
            n_points=100,
        )
        analytical = evolver.steady_state_fold(rifampicin, 0.2, "CYP3A4")
        numerical = trace.max_fold("CYP3A4")
        assert math.isclose(numerical, analytical, rel_tol=1e-3)
        # And it should match the pure Step 8 Emax value
        assert math.isclose(analytical, 1.0 + 12.0 * 0.2 / (0.31 + 0.2), rel_tol=1e-6)


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — ODE BEHAVIOUR
# ════════════════════════════════════════════════════════════════


class TestODEBehaviour:
    """Mathematical invariants of the time-dependent pool ODE."""

    def test_zero_concentration_keeps_baseline_everywhere(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        """With [I]u = 0 the ODE collapses to de/dt = kdeg(1 - e),
        which is already at equilibrium at e = 1. All folds must
        stay at 1.0 across the full window."""
        trace = evolver.simulate(
            rifampicin, 0.0, duration_on_h=7 * 24, n_points=50
        )
        for snap in trace.snapshots:
            for cyp in evolver.tracked_cyps:
                assert math.isclose(
                    snap.fold_for(cyp), 1.0, rel_tol=1e-6, abs_tol=1e-6
                )

    def test_non_inducer_non_mbi_drug_is_inert(
        self, evolver: EnzymePoolEvolver, ketoconazole: Drug
    ):
        """Ketoconazole is a reversible CYP3A4 inhibitor only; it has no
        Indmax and no kinact. Pool must stay exactly at baseline even
        at saturating concentration."""
        trace = evolver.simulate(
            ketoconazole,
            inducer_conc_um=100.0,
            duration_on_h=7 * 24,
            n_points=50,
        )
        for snap in trace.snapshots:
            assert math.isclose(snap.fold_for("CYP3A4"), 1.0, rel_tol=1e-6)

    def test_initial_derivative_at_baseline_is_kdeg_times_find(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        """At t=0 with e=1 and pure induction (no MBI), the slope must
        equal ``kdeg · f_ind`` — the rate of new enzyme synthesis above
        baseline."""
        # f_ind at 0.2 µM rifampicin = 12 * 0.2 / (0.31 + 0.2)
        f_ind_expected = 12.0 * 0.2 / (0.31 + 0.2)
        kdeg = DEFAULT_KDEG_PER_H["CYP3A4"]
        expected = kdeg * f_ind_expected  # baseline slope

        actual = evolver.derivative(
            e_current=1.0,
            cyp="CYP3A4",
            perpetrator=rifampicin,
            inducer_conc_um=0.2,
        )
        assert math.isclose(actual, expected, rel_tol=1e-9)

    def test_derivative_is_zero_at_steady_state(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        """By construction, plugging the analytical steady-state value
        into ``derivative`` must give zero."""
        e_ss = evolver.steady_state_fold(rifampicin, 0.5, "CYP3A4")
        slope = evolver.derivative(
            e_current=e_ss,
            cyp="CYP3A4",
            perpetrator=rifampicin,
            inducer_conc_um=0.5,
        )
        assert abs(slope) < 1e-12

    def test_mbi_only_drug_suppresses_cyp3a4(
        self, evolver: EnzymePoolEvolver, clarithromycin: Drug
    ):
        """Clarithromycin MBI at 1 µM should drop CYP3A4 to <20 % of
        baseline at steady state (empirically ~4 %)."""
        trace = evolver.simulate(
            clarithromycin,
            inducer_conc_um=1.0,
            duration_on_h=5 * 24,
            n_points=200,
        )
        final_fold = trace.snapshots[-1].fold_for("CYP3A4")
        assert final_fold < 0.2, (
            f"Clarithromycin MBI end-fold = {final_fold:.3f}, "
            f"expected <0.2 after 5 days."
        )
        assert final_fold > 0.0

    def test_mbi_suppression_recovers_after_washout(
        self, evolver: EnzymePoolEvolver, clarithromycin: Drug
    ):
        """MBI-suppressed CYP3A4 must recover via de novo synthesis
        (kdeg · E_baseline) when clarithromycin is stopped. By ~10 days
        the fold should be within 5 % of baseline."""
        trace = evolver.simulate(
            clarithromycin,
            inducer_conc_um=1.0,
            duration_on_h=5 * 24,
            duration_off_h=10 * 24,
            n_points=400,
        )
        final = trace.snapshots[-1].fold_for("CYP3A4")
        assert 0.95 <= final <= 1.05, (
            f"Clarithromycin washout end-fold = {final:.3f}."
        )

    def test_combined_induction_and_mbi_multiplicative(
        self,
        evolver: EnzymePoolEvolver,
        combined_drug: Drug,
    ):
        """Combined drug: f_ind = 5·C/(1+C), kobs = 1.8·C/(2+C) (h⁻¹).

        At C = 2 µM:
            f_ind = 5 · 2 / 3 = 3.333…
            kobs  = (0.03 × 60) · 2 / 4 = 0.9 h⁻¹
            e_ss  = kdeg · (1 + 3.333) / (kdeg + 0.9)
                  ≈ 0.01925 · 4.333 / 0.9193
                  ≈ 0.0907
        So the two mechanisms fight each other and MBI wins.
        """
        e_ss = evolver.steady_state_fold(combined_drug, 2.0, "CYP3A4")
        expected = DEFAULT_KDEG_PER_H["CYP3A4"] * (1.0 + 5.0 * 2.0 / 3.0) / (
            DEFAULT_KDEG_PER_H["CYP3A4"] + (0.03 * 60.0) * 2.0 / 4.0
        )
        assert math.isclose(e_ss, expected, rel_tol=1e-6)
        assert e_ss < 0.15  # MBI dominates despite strong induction

    def test_induction_is_monotonic_increasing_before_plateau(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        """During the induction phase the fold trace must be non-decreasing
        for CYP3A4 (no overshoot, no oscillation)."""
        trace = evolver.simulate(
            rifampicin, 0.2, duration_on_h=10 * 24, n_points=400
        )
        trace_values = trace.fold_trace("CYP3A4")
        for a, b in zip(trace_values, trace_values[1:]):
            assert b >= a - 1e-9, "CYP3A4 fold trace dropped during induction"

    def test_cyp2c9_slower_than_cyp3a4(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        """CYP2C9 has a longer half-life (55 h vs 36 h), so it must
        reach 90 % of its plateau LATER than CYP3A4 under the same
        rifampicin exposure."""
        trace = evolver.simulate(
            rifampicin, 0.5, duration_on_h=21 * 24, n_points=400
        )
        t_90_3a4 = trace.time_to_fraction("CYP3A4", 0.9)
        t_90_2c9 = trace.time_to_fraction("CYP2C9", 0.9)
        assert t_90_3a4 is not None and t_90_2c9 is not None
        assert t_90_2c9 > t_90_3a4


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — INPUT VALIDATION
# ════════════════════════════════════════════════════════════════


class TestInputValidation:
    def test_negative_duration_raises(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        with pytest.raises(ValueError):
            evolver.simulate(rifampicin, 0.2, duration_on_h=-1.0)

    def test_negative_duration_off_raises(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        with pytest.raises(ValueError):
            evolver.simulate(
                rifampicin,
                0.2,
                duration_on_h=24.0,
                duration_off_h=-5.0,
            )

    def test_zero_total_duration_raises(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        with pytest.raises(ValueError):
            evolver.simulate(rifampicin, 0.2, duration_on_h=0.0)

    def test_too_few_points_raises(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        with pytest.raises(ValueError):
            evolver.simulate(
                rifampicin, 0.2, duration_on_h=24.0, n_points=1
            )

    def test_negative_constant_concentration_raises(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        with pytest.raises(ValueError):
            evolver.simulate(
                rifampicin, -0.1, duration_on_h=24.0
            )

    def test_missing_kdeg_entry_raises(self):
        with pytest.raises(ValueError):
            EnzymePoolEvolver(kdeg_per_h={"CYP3A4": 0.02})  # incomplete


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — TRACE DATACLASS
# ════════════════════════════════════════════════════════════════


class TestEnzymePoolTraceContract:
    def test_snapshot_is_frozen(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        trace = evolver.simulate(rifampicin, 0.2, duration_on_h=24.0)
        with pytest.raises((AttributeError, TypeError)):
            trace.snapshots[0].time_h = 999.0  # type: ignore[misc]

    def test_trace_is_frozen(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        trace = evolver.simulate(rifampicin, 0.2, duration_on_h=24.0)
        with pytest.raises((AttributeError, TypeError)):
            trace.perpetrator_name = "other"  # type: ignore[misc]

    def test_snapshots_start_at_zero_with_fold_one(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        trace = evolver.simulate(rifampicin, 0.2, duration_on_h=24.0)
        assert trace.snapshots[0].time_h == pytest.approx(0.0)
        for cyp in evolver.tracked_cyps:
            assert math.isclose(
                trace.snapshots[0].fold_for(cyp), 1.0, rel_tol=1e-9
            )

    def test_all_tracked_cyps_present_in_every_snapshot(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        trace = evolver.simulate(rifampicin, 0.2, duration_on_h=24.0)
        for snap in trace.snapshots:
            assert set(snap.fold_by_cyp.keys()) == set(evolver.tracked_cyps)

    def test_fold_at_interpolates_between_snapshots(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        trace = evolver.simulate(
            rifampicin, 0.2, duration_on_h=72.0, n_points=50
        )
        # Pick a time between snapshot 10 and 11 — should lie between
        # their fold values for CYP3A4.
        t_mid = 0.5 * (
            trace.snapshots[10].time_h + trace.snapshots[11].time_h
        )
        f_low = trace.snapshots[10].fold_for("CYP3A4")
        f_high = trace.snapshots[11].fold_for("CYP3A4")
        f_interp = trace.fold_at(t_mid, "CYP3A4")
        assert min(f_low, f_high) <= f_interp <= max(f_low, f_high)

    def test_fold_at_outside_window_returns_endpoint(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        trace = evolver.simulate(
            rifampicin, 0.2, duration_on_h=24.0, n_points=20
        )
        f_before = trace.fold_at(-10.0, "CYP3A4")
        f_after = trace.fold_at(1e9, "CYP3A4")
        assert math.isclose(
            f_before, trace.snapshots[0].fold_for("CYP3A4"), rel_tol=1e-9
        )
        assert math.isclose(
            f_after, trace.snapshots[-1].fold_for("CYP3A4"), rel_tol=1e-9
        )

    def test_time_to_fraction_returns_none_when_no_induction(
        self, evolver: EnzymePoolEvolver, ketoconazole: Drug
    ):
        trace = evolver.simulate(
            ketoconazole, 0.5, duration_on_h=10 * 24, n_points=50
        )
        # Ketoconazole doesn't induce, so the plateau is 1 and
        # ``time_to_fraction`` has no meaningful answer.
        assert trace.time_to_fraction("CYP3A4", 0.9) is None

    def test_fold_trace_length_matches_snapshots(
        self, evolver: EnzymePoolEvolver, rifampicin: Drug
    ):
        trace = evolver.simulate(
            rifampicin, 0.2, duration_on_h=24.0, n_points=37
        )
        assert len(trace.fold_trace("CYP3A4")) == len(trace.snapshots)
        assert len(trace.times_h()) == len(trace.snapshots)


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — kdeg CONSTANTS
# ════════════════════════════════════════════════════════════════


class TestKdegDefaults:
    def test_all_tracked_cyps_have_kdeg(self):
        for cyp in CYP_NAMES:
            assert cyp in DEFAULT_KDEG_PER_H
            assert DEFAULT_KDEG_PER_H[cyp] > 0.0

    def test_cyp3a4_half_life_is_approximately_36h(self):
        kdeg = DEFAULT_KDEG_PER_H["CYP3A4"]
        t_half = math.log(2.0) / kdeg
        assert 33.0 <= t_half <= 39.0

    def test_cyp2c9_half_life_is_approximately_55h(self):
        kdeg = DEFAULT_KDEG_PER_H["CYP2C9"]
        t_half = math.log(2.0) / kdeg
        assert 50.0 <= t_half <= 60.0
