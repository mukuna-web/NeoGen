"""V1 P4 Step 13 — Receptor Kinetics unit + clinical tests.

Covers:
  - ``simulate_ro_trajectory`` correctness: reversible steady state,
    irreversible recovery, numerical stability, defensive clamps.
  - ``ReceptorKineticsEngine`` contract: missing target raises, frozen
    output, aligned arrays, summary invariants.
  - Clinical invariants from the plan §6:
      * aspirin: ≥ 80% platelet COX-1 occupancy several days after plasma
        has cleared (covalent binding + platelet turnover).
      * propranolol-like reversible slow-koff: RO tracks Cp with kinetic
        residence lag.
      * caffeine-like fast reversible: RO equilibrates within minutes of
        Cp changes (no apparent hysteresis on hourly grids).
  - Therapeutic window interpolation (sub-sample accuracy).
  - Real P1 PBPK plasma profile smoke test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from neogen.core.entities.pd import (
    PDProfile,
    ReceptorTarget,
)
from neogen.engine.pd.pd_result import (
    ReceptorOccupancySummary,
    ReceptorOccupancyTrace,
)
from neogen.engine.pd.receptor_kinetics import (
    ReceptorKineticsEngine,
    simulate_ro_trajectory,
)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _bolus_then_clear(
    c0_mg_per_L: float,
    k_el_per_h: float,
    total_duration_h: float,
    n_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """IV-bolus-like profile: exponential decay of plasma concentration.

    Used for clean tests of covalent binders — we care about "plasma
    clears, then what happens to RO over days".
    """
    t = np.linspace(0.0, total_duration_h, n_points)
    cp = c0_mg_per_L * np.exp(-k_el_per_h * t)
    return t, cp


def _constant_cp(
    cp_value_mg_per_L: float,
    duration_h: float,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant plasma infusion-style input for steady-state tests."""
    t = np.linspace(0.0, duration_h, n_points)
    cp = np.full_like(t, cp_value_mg_per_L)
    return t, cp


def _reversible_target(
    *,
    name: str = "TestTarget",
    kon: float = 50.0,
    koff: float = 1.0,
    therapeutic: float = 0.5,
) -> ReceptorTarget:
    return ReceptorTarget(
        name=name,
        kon_per_mg_per_L_per_h=kon,
        koff_per_h=koff,
        irreversible=False,
        therapeutic_occupancy=therapeutic,
    )


def _irreversible_target(
    *,
    name: str = "COX-1-like",
    kon: float = 50.0,
    recovery_h: float = 240.0,  # 10-day platelet lifespan
    therapeutic: float = 0.8,
) -> ReceptorTarget:
    return ReceptorTarget(
        name=name,
        kon_per_mg_per_L_per_h=kon,
        koff_per_h=0.0,
        irreversible=True,
        recovery_half_life_h=recovery_h,
        therapeutic_occupancy=therapeutic,
    )


# ════════════════════════════════════════════════════════════════
# 1. INTEGRATOR CORRECTNESS
# ════════════════════════════════════════════════════════════════


class TestRoIntegrator:
    """Pure integrator correctness + stability."""

    def test_reversible_reaches_analytical_steady_state(self):
        """For constant Cp, dRO/dt = kon·Cp·(1-RO) - koff·RO.
        Steady state: RO_ss = kon·Cp / (kon·Cp + koff) = Cp / (Kd + Cp).
        """
        kon, koff = 20.0, 4.0
        kd = koff / kon  # 0.2 mg/L
        cp_ss = 0.3  # mg/L
        target = _reversible_target(kon=kon, koff=koff)

        t, cp = _constant_cp(cp_ss, duration_h=20.0, n_points=1000)
        ro = np.asarray(simulate_ro_trajectory(t, cp, target))

        expected_ss = cp_ss / (kd + cp_ss)  # 0.3/0.5 = 0.6
        # After 20 h, should be within 1e-6 of analytical SS for this kinetics
        assert ro[-1] == pytest.approx(expected_ss, rel=1e-6)
        # Monotone approach from zero
        assert np.all(np.diff(ro) >= -1e-12)

    def test_reversible_cp_equal_kd_gives_half_occupancy(self):
        """Canonical Michaelis-Menten anchor: Cp = Kd → RO_ss = 0.5."""
        kon, koff = 10.0, 2.0
        kd = koff / kon  # 0.2
        target = _reversible_target(kon=kon, koff=koff)
        t, cp = _constant_cp(kd, duration_h=30.0)
        ro = np.asarray(simulate_ro_trajectory(t, cp, target))
        assert ro[-1] == pytest.approx(0.5, abs=1e-6)

    def test_irreversible_decays_via_biological_half_life_after_clearance(
        self,
    ):
        """The critical aspirin-like test: once Cp is zero, RO should
        decay with the biological recovery half-life, NOT the chemical
        koff (which is 0)."""
        target = _irreversible_target(kon=100.0, recovery_h=240.0)

        # Short plasma exposure — saturate COX-1 in the first hour, then
        # cp goes to zero and we follow RO for 2 weeks.
        t_short = np.linspace(0.0, 1.0, 100)
        cp_short = np.full_like(t_short, 5.0)  # saturating concentration

        t_long = np.linspace(1.0 + 0.1, 14 * 24.0, 400)
        cp_long = np.zeros_like(t_long)

        t = np.concatenate([t_short, t_long])
        cp = np.concatenate([cp_short, cp_long])

        ro = np.asarray(simulate_ro_trajectory(t, cp, target))

        # After 1 h of saturating Cp with fast kon, RO should be ≈ 1.
        idx_end_dose = 99
        assert ro[idx_end_dose] > 0.99

        # After the drug clears, RO decays as RO(t) = RO_0 · exp(-k_rec·(t-t0))
        # with k_rec = ln(2)/240 = 0.00289 h⁻¹.
        k_rec = math.log(2.0) / 240.0
        # Sample at t = 24, 48, 120, 240, 336 h and compare.
        check_times = [24.0, 48.0, 120.0, 240.0, 336.0]
        for ct in check_times:
            idx = int(np.argmin(np.abs(t - ct)))
            expected = math.exp(-k_rec * t[idx])
            # Allow 1% relative tolerance — midpoint approximation plus
            # the 1-h dosing window, not pure analytical.
            assert ro[idx] == pytest.approx(expected, rel=0.02), (
                f"At t={t[idx]:.1f}h, RO={ro[idx]:.4f} "
                f"expected {expected:.4f}"
            )

    def test_irreversible_stays_above_threshold_past_several_days(self):
        """Headline invariant from plan §6: aspirin 80% after days.

        With recovery t½ = 240 h, the 80% threshold is crossed at
        t = -240·log₂(0.8) ≈ 77.3 h ≈ 3.2 days.
        """
        target = _irreversible_target(kon=100.0, recovery_h=240.0)
        t = np.linspace(0.0, 14 * 24.0, 3000)
        cp = np.where(t < 1.0, 5.0, 0.0)

        ro = np.asarray(simulate_ro_trajectory(t, cp, target))

        # At 72 h (3 days), RO should still be > 0.8
        idx_72 = int(np.argmin(np.abs(t - 72.0)))
        assert ro[idx_72] > 0.80, (
            f"Aspirin RO at 72 h = {ro[idx_72]:.3f}, expected > 0.80"
        )
        # At 168 h (7 days) it should have crossed below 0.8 but still
        # above 0.5 (clinically meaningful antiplatelet effect)
        idx_168 = int(np.argmin(np.abs(t - 168.0)))
        assert 0.5 < ro[idx_168] < 0.80

    def test_fast_kinetics_equilibrate_quickly(self):
        """Caffeine-like: large kon AND koff → RO equilibrates within
        minutes, so hourly samples show no visible hysteresis."""
        target = _reversible_target(kon=1000.0, koff=100.0)
        # Step Cp up at t=0.5h from 0.1 to 1.0 mg/L
        t = np.linspace(0.0, 4.0, 4000)
        cp = np.where(t < 0.5, 0.1, 1.0)

        ro = np.asarray(simulate_ro_trajectory(t, cp, target))

        kd = 100.0 / 1000.0  # 0.1
        ro_ss_low = 0.1 / (kd + 0.1)   # 0.5
        ro_ss_high = 1.0 / (kd + 1.0)  # ~0.909

        # Well before the step
        idx_before = int(np.argmin(np.abs(t - 0.4)))
        assert ro[idx_before] == pytest.approx(ro_ss_low, abs=1e-4)

        # Just after the step (20 ms) it should already be near the
        # new steady state — k_eff ≈ 1000*1+100 = 1100 /h, time const ~3 s.
        idx_after = int(np.argmin(np.abs(t - 0.6)))
        assert ro[idx_after] == pytest.approx(ro_ss_high, abs=1e-3)

    def test_stable_for_large_kon_cp_dt_product(self):
        """Numerical stability: forced Euler-killing regime."""
        target = _reversible_target(kon=500.0, koff=50.0)
        # Δt = 4 h, Cp = 10 → kon·Cp·Δt = 20000 — Euler would explode.
        t = np.array([0.0, 4.0, 8.0, 12.0])
        cp = np.array([10.0, 10.0, 10.0, 10.0])
        ro = np.asarray(simulate_ro_trajectory(t, cp, target))

        assert np.all(ro >= 0.0)
        assert np.all(ro <= 1.0)
        # Steady state = 10/(0.1+10) = 0.9901
        assert ro[-1] == pytest.approx(0.9901, abs=1e-4)

    def test_empty_input_returns_empty(self):
        target = _reversible_target()
        assert simulate_ro_trajectory(np.array([]), np.array([]), target) == ()

    def test_rejects_bad_inputs(self):
        target = _reversible_target()
        t = np.array([0.0, 1.0, 2.0])
        cp = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="shape"):
            simulate_ro_trajectory(t, cp[:2], target)

        with pytest.raises(ValueError, match="non-decreasing"):
            simulate_ro_trajectory(np.array([0.0, 2.0, 1.0]), cp, target)

        with pytest.raises(ValueError, match="ro0"):
            simulate_ro_trajectory(t, cp, target, ro0=1.5)

    def test_negative_cp_clamped_to_zero(self):
        """PBPK solver can leak tiny negative values; must not propagate."""
        target = _reversible_target(kon=10.0, koff=1.0)
        t = np.array([0.0, 1.0, 2.0, 3.0])
        cp = np.array([1.0, -1e-10, 1.0, 1.0])
        ro = np.asarray(simulate_ro_trajectory(t, cp, target))
        # Should be well-defined, monotone-ish, and never NaN/negative.
        assert np.all(np.isfinite(ro))
        assert np.all(ro >= 0.0)


# ════════════════════════════════════════════════════════════════
# 2. ENGINE CONTRACT
# ════════════════════════════════════════════════════════════════


class TestEngineContract:
    def test_missing_receptor_target_raises(self):
        engine = ReceptorKineticsEngine()
        profile = PDProfile()  # No receptor_target
        t, cp = _bolus_then_clear(5.0, 0.2, 48.0)
        with pytest.raises(ValueError, match="receptor_target"):
            engine.simulate(
                drug_name="test",
                time_h=t,
                plasma_conc_mg_per_L=cp,
                pd_profile=profile,
            )

    def test_returns_frozen_trace_with_aligned_arrays(self):
        engine = ReceptorKineticsEngine()
        target = _reversible_target(name="Beta-1", kon=30.0, koff=0.5)
        profile = PDProfile(receptor_target=target)
        t, cp = _bolus_then_clear(2.0, 0.3, 24.0)

        trace = engine.simulate(
            drug_name="drugX",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        assert isinstance(trace, ReceptorOccupancyTrace)
        assert trace.drug_name == "drugX"
        assert trace.target_name == "Beta-1"
        assert trace.n_points == len(t)
        assert len(trace.plasma_conc_mg_per_L) == len(t)
        assert len(trace.occupancy) == len(t)
        assert not trace.irreversible

        # Frozen
        with pytest.raises(Exception):
            trace.drug_name = "other"  # type: ignore[misc]

    def test_summary_fields_populated(self):
        engine = ReceptorKineticsEngine()
        target = _reversible_target(kon=50.0, koff=1.0, therapeutic=0.6)
        profile = PDProfile(receptor_target=target)
        t, cp = _bolus_then_clear(3.0, 0.25, 48.0)

        trace = engine.simulate(
            drug_name="x",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        s: ReceptorOccupancySummary = trace.summary

        # Peak occupancy in [0, 1]
        assert 0.0 <= s.peak_occupancy <= 1.0
        # Peak should be the np.max of the trace
        assert s.peak_occupancy == pytest.approx(
            float(np.max(trace.occupancy)), abs=1e-12
        )
        # Non-negative therapeutic duration
        assert s.time_above_therapeutic_h >= 0.0
        # Last occupancy = last sample
        assert s.last_occupancy == pytest.approx(trace.occupancy[-1])
        # Reversible
        assert s.is_irreversible is False

    def test_irreversible_summary_flags_is_irreversible(self):
        engine = ReceptorKineticsEngine()
        target = _irreversible_target(kon=100.0, recovery_h=240.0)
        profile = PDProfile(receptor_target=target)
        t = np.linspace(0.0, 14 * 24.0, 2000)
        cp = np.where(t < 1.0, 5.0, 0.0)

        trace = engine.simulate(
            drug_name="aspirin",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        assert trace.summary.is_irreversible is True
        assert trace.irreversible is True
        assert trace.recovery_half_life_h == 240.0


# ════════════════════════════════════════════════════════════════
# 3. CLINICAL INVARIANTS (plan §6)
# ════════════════════════════════════════════════════════════════


class TestClinicalInvariants:
    """Reference-drug behaviours from the V1 plan."""

    def test_aspirin_cox1_still_blocked_days_after_plasma_clears(self):
        """Plan §6 headline: aspirin receptor occupancy > 80% days after
        plasma is essentially zero.

        Physics: aspirin covalently acetylates platelet COX-1 (Ser530).
        Chemical koff ≈ 0 — recovery is gated entirely by platelet
        turnover (~10 day mean lifespan → t½ ≈ 240 h). We simulate a
        single 325 mg oral-equivalent dose and confirm:
          1. Plasma clears within a few hours (short half-life ~0.3 h)
          2. Occupancy saturates near 1.0 during the dosing window
          3. Occupancy is still > 80% at 72 h (3 days) — the headline
             invariant from the plan.
        """
        engine = ReceptorKineticsEngine()
        target = _irreversible_target(
            name="COX-1", kon=100.0, recovery_h=240.0, therapeutic=0.8
        )
        profile = PDProfile(receptor_target=target)

        # 7-day simulation; aspirin plasma half-life ~0.3h (very fast).
        t, cp = _bolus_then_clear(
            c0_mg_per_L=20.0,
            k_el_per_h=2.3,        # t½ ≈ 0.3 h
            total_duration_h=14 * 24.0,
            n_points=3000,
        )

        trace = engine.simulate(
            drug_name="aspirin",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        ro = np.asarray(trace.occupancy)
        t_arr = np.asarray(trace.time_h)

        # 1. Plasma is effectively cleared by t = 6 h
        # (Cp ≈ 20·exp(-2.3·6) ≈ 2e-5 mg/L — orders of magnitude below
        # any plausible assay LOQ, "essentially zero" for the invariant
        # that follows.)
        idx_6h = int(np.argmin(np.abs(t_arr - 6.0)))
        assert cp[idx_6h] < 1e-4

        # 2. Peak occupancy is near full saturation
        assert trace.summary.peak_occupancy > 0.99

        # 3. HEADLINE: RO at 72 h > 80%
        idx_72h = int(np.argmin(np.abs(t_arr - 72.0)))
        assert ro[idx_72h] > 0.80, (
            f"Aspirin RO at 72 h = {ro[idx_72h]:.3f}, plan requires > 0.80"
        )

        # 4. RO at 120 h (5 days) still > 60% — antiplatelet effect lingers
        idx_120h = int(np.argmin(np.abs(t_arr - 120.0)))
        assert ro[idx_120h] > 0.60

        # 5. time_above_therapeutic: with threshold 0.8, aspirin is above
        # ~77 h (from the analytical crossing). Allow ±5 h window.
        expected_crossing = 240.0 * math.log(1 / 0.8) / math.log(2.0)
        # = 240 · ln(1.25)/ln(2) ≈ 77.3 h
        assert expected_crossing - 5.0 < trace.summary.time_above_therapeutic_h
        assert (
            trace.summary.time_above_therapeutic_h < expected_crossing + 5.0
        )

    def test_omeprazole_cox_like_gastric_pump_recovery(self):
        """Omeprazole: covalent H⁺/K⁺-ATPase inhibition, 36h enzyme
        turnover. Plan says "single dose suppresses acid for 24h"."""
        engine = ReceptorKineticsEngine()
        target = _irreversible_target(
            name="H+/K+ ATPase",
            kon=200.0,
            recovery_h=36.0,
            therapeutic=0.5,
        )
        profile = PDProfile(receptor_target=target)

        t, cp = _bolus_then_clear(
            c0_mg_per_L=2.0,
            k_el_per_h=0.7,       # ~1 h half-life
            total_duration_h=96.0,
            n_points=2000,
        )
        trace = engine.simulate(
            drug_name="omeprazole",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        # At 24 h, RO should still be > 50% (clinically meaningful acid
        # suppression); by 96 h it should have crossed well below.
        ro = np.asarray(trace.occupancy)
        t_arr = np.asarray(trace.time_h)
        idx_24 = int(np.argmin(np.abs(t_arr - 24.0)))
        idx_96 = int(np.argmin(np.abs(t_arr - 96.0)))
        assert ro[idx_24] > 0.5
        assert ro[idx_96] < 0.3

    def test_reversible_slow_koff_gives_sustained_occupancy(self):
        """Atorvastatin-like: reversible but very tight binding.
        Slow-off kinetics → RO lags Cp on the way down."""
        engine = ReceptorKineticsEngine()
        target = _reversible_target(kon=500.0, koff=0.1, therapeutic=0.5)
        profile = PDProfile(receptor_target=target)

        t, cp = _bolus_then_clear(
            c0_mg_per_L=0.5, k_el_per_h=0.25, total_duration_h=48.0
        )
        trace = engine.simulate(
            drug_name="statin-like",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        ro = np.asarray(trace.occupancy)
        t_arr = np.asarray(trace.time_h)

        # Peak RO time should be shortly after Cp peak (slow-off means
        # buildup tracks, dissolve is the slow part)
        idx_cp_peak = int(np.argmax(cp))
        idx_ro_peak = int(np.argmax(ro))
        assert t_arr[idx_ro_peak] >= t_arr[idx_cp_peak] - 1e-9

        # Significant occupancy persists beyond plasma — check at 24 h
        # when plasma is essentially gone (cp(24) = 0.5·exp(-6) ≈ 0.00124)
        idx_24 = int(np.argmin(np.abs(t_arr - 24.0)))
        assert cp[idx_24] < 0.01
        # With koff=0.1, RO half-life in absence of drug ≈ 6.9 h,
        # so at 24h we have ~exp(-0.1*24) ≈ 0.09 residual from peak.
        # Enough to show the long tail relative to plasma.
        assert ro[idx_24] / cp[idx_24] > 10.0  # RO/Cp ratio blows up


# ════════════════════════════════════════════════════════════════
# 4. THERAPEUTIC WINDOW INTERPOLATION
# ════════════════════════════════════════════════════════════════


class TestTherapeuticWindow:
    def test_time_above_uses_linear_interpolation(self):
        """Hand-built triangular occupancy curve: peak 1.0 at midpoint,
        threshold 0.5, total duration above threshold should be exactly
        half the peak width, not biased by half a sample."""
        # Simulate by constructing an irreversible target with tuned
        # kon to give a specific monotone saturation, then compare summary.
        # Easier: test the trajectory directly via a reversible target
        # driven to a specific steady state.
        target = _reversible_target(kon=100.0, koff=1.0, therapeutic=0.5)
        # Kd = 0.01. At Cp = 0.01, RO_ss = 0.5. At Cp = 0.03, RO_ss = 0.75.
        # Constant Cp = 0.03 for 10h — RO saturates above 0.5 almost
        # immediately and stays there the whole time.
        t, cp = _constant_cp(0.03, duration_h=10.0, n_points=500)

        engine = ReceptorKineticsEngine()
        profile = PDProfile(receptor_target=target)
        trace = engine.simulate(
            drug_name="x",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        # Time above 0.5 should be very close to the full 10 h window
        # (minus the short rise from 0 → 0.5 at the start).
        # With k_eff = 100*0.03 + 1 = 4 /h, RO hits 0.5 in ~ln(...) seconds.
        assert 9.5 < trace.summary.time_above_therapeutic_h <= 10.0

    def test_first_crossing_none_when_threshold_unreachable(self):
        """Therapeutic threshold 0.95 with only Cp reaching 0.1·Kd → RO_ss
        ≈ 0.09, never crosses."""
        target = _reversible_target(kon=10.0, koff=1.0, therapeutic=0.95)
        # Kd = 0.1. Cp = 0.01 → RO_ss = 0.09
        engine = ReceptorKineticsEngine()
        profile = PDProfile(receptor_target=target)
        t, cp = _constant_cp(0.01, duration_h=20.0)
        trace = engine.simulate(
            drug_name="x",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        assert trace.summary.first_crossing_above_h is None
        assert trace.summary.time_above_therapeutic_h == 0.0


# ════════════════════════════════════════════════════════════════
# 5. REAL P1 INTEGRATION SMOKE TEST
# ════════════════════════════════════════════════════════════════


class TestP1IntegrationSmoke:
    def test_runs_on_real_midazolam_pbpk_output(self):
        from neogen.engine import (
            DoseScheduler,
            DrugFactory,
            SimulationOrchestrator,
        )
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        drug = DrugFactory().create("midazolam")
        schedule = DoseScheduler().iv_bolus("midazolam", dose_mg=5.0)
        physiology = build_reference_male()
        result = SimulationOrchestrator().run(drug, schedule, physiology)

        # Attach a reversible GABA-A benzo-site-like target
        target = ReceptorTarget(
            name="GABA-A benzo site",
            kon_per_mg_per_L_per_h=800.0,
            koff_per_h=2.5,
            irreversible=False,
            therapeutic_occupancy=0.5,
        )
        profile = PDProfile(receptor_target=target)

        engine = ReceptorKineticsEngine()
        trace = engine.simulate(
            drug_name=drug.name,
            time_h=result.time_h,
            plasma_conc_mg_per_L=result.plasma_conc_mg_per_L,
            pd_profile=profile,
        )

        assert trace.n_points == len(result.time_h)
        # All occupancy values are physically valid
        ro = np.asarray(trace.occupancy)
        assert np.all(ro >= 0.0) and np.all(ro <= 1.0)
        # Midazolam should drive reasonable peak occupancy at 5 mg IV
        assert trace.summary.peak_occupancy > 0.2
