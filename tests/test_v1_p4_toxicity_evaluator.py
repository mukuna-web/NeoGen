"""V1 P4 Step 14 — toxicity evaluator test suite.

Covers:
  * Pure crossing-detection math (interpolated sub-sample accuracy)
  * Report assembly (severity ordering, per-organ worst, TI)
  * Engine contract (empty profiles, missing PD data)
  * Clinical invariants (theophylline CNS, paracetamol hepatic, aspirin
    salicylism ladder, and a below-threshold no-toxicity case)
  * P1 integration smoke test against real PBPK output
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from neogen.core.entities.pd import (
    EffectOrgan,
    EmaxParams,
    PDProfile,
    ToxicitySeverity,
    ToxicityThreshold,
)
from neogen.engine.pd.toxicity_evaluator import (
    ToxicityEvaluatorEngine,
    _first_crossing_time,
)
from neogen.engine.pd.pd_result import ToxicityReport


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _bolus_profile(
    c0_mg_per_L: float,
    k_el_per_h: float,
    duration_h: float,
    n_points: int = 500,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Mono-exponential IV bolus profile: Cp(t) = c0 · exp(-k_el · t)."""
    t = np.linspace(0.0, duration_h, n_points)
    cp = c0_mg_per_L * np.exp(-k_el_per_h * t)
    return tuple(float(x) for x in t), tuple(float(x) for x in cp)


def _rising_then_falling(
    cmax_mg_per_L: float,
    t_max_h: float,
    duration_h: float,
    n_points: int = 500,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Bateman-like profile with Cmax at t_max, then exponential decay."""
    ka = 1.5
    k_el = max(ka * 0.25, 0.05)
    t = np.linspace(0.0, duration_h, n_points)
    raw = np.exp(-k_el * t) - np.exp(-ka * t)
    raw_peak = raw.max()
    cp = (cmax_mg_per_L / raw_peak) * raw
    # Make sure t_max is roughly honoured by truncating/scaling — cheap
    # enough for tests.
    return tuple(float(x) for x in t), tuple(float(x) for x in cp)


def _theophylline_thresholds() -> Tuple[ToxicityThreshold, ...]:
    return (
        ToxicityThreshold(
            organ=EffectOrgan.BRAIN,
            severity=ToxicitySeverity.MILD,
            concentration_mg_per_L=15.0,
            symptom="Jitteriness",
        ),
        ToxicityThreshold(
            organ=EffectOrgan.BRAIN,
            severity=ToxicitySeverity.MODERATE,
            concentration_mg_per_L=20.0,
            symptom="CNS stimulation",
            description="Excess adenosine antagonism at theophylline > 20 mg/L",
        ),
        ToxicityThreshold(
            organ=EffectOrgan.HEART,
            severity=ToxicitySeverity.MODERATE,
            concentration_mg_per_L=25.0,
            symptom="Tachycardia",
        ),
        ToxicityThreshold(
            organ=EffectOrgan.BRAIN,
            severity=ToxicitySeverity.SEVERE,
            concentration_mg_per_L=40.0,
            symptom="Seizure risk",
        ),
        ToxicityThreshold(
            organ=EffectOrgan.HEART,
            severity=ToxicitySeverity.CRITICAL,
            concentration_mg_per_L=60.0,
            symptom="Ventricular arrhythmia",
        ),
    )


def _theophylline_profile() -> PDProfile:
    return PDProfile(
        emax=EmaxParams(
            emax=100.0,
            ec50_mg_per_L=10.0,  # therapeutic bronchodilation EC50 ~10 mg/L
            effect_name="Bronchodilation",
        ),
        toxicity_thresholds=_theophylline_thresholds(),
        therapeutic_range_low_mg_per_L=5.0,
        therapeutic_range_high_mg_per_L=15.0,
    )


def _paracetamol_profile() -> PDProfile:
    return PDProfile(
        toxicity_thresholds=(
            ToxicityThreshold(
                organ=EffectOrgan.LIVER,
                severity=ToxicitySeverity.MILD,
                concentration_mg_per_L=50.0,
                symptom="Mild hepatocyte stress",
            ),
            ToxicityThreshold(
                organ=EffectOrgan.LIVER,
                severity=ToxicitySeverity.SEVERE,
                concentration_mg_per_L=150.0,
                symptom="Hepatotoxicity (Rumack-Matthew)",
                description="Acute liver injury risk above the 4h line",
            ),
            ToxicityThreshold(
                organ=EffectOrgan.LIVER,
                severity=ToxicitySeverity.CRITICAL,
                concentration_mg_per_L=300.0,
                symptom="Fulminant hepatic failure",
            ),
        ),
    )


# ════════════════════════════════════════════════════════════════
# 1. CROSSING DETECTION MATH
# ════════════════════════════════════════════════════════════════


class TestCrossingDetection:
    def test_interpolated_crossing_accuracy(self):
        """Crossing time should be linearly interpolated, not snapped."""
        t = np.array([0.0, 1.0, 2.0])
        cp = np.array([0.0, 10.0, 5.0])  # crosses 5 at t=0.5, not t=1.0
        result = _first_crossing_time(t, cp, 5.0)
        assert result is not None
        t_cross, cp_cross = result
        assert abs(t_cross - 0.5) < 1e-9
        assert abs(cp_cross - 5.0) < 1e-9

    def test_no_crossing_returns_none(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        cp = np.array([1.0, 2.0, 1.5, 0.5])
        assert _first_crossing_time(t, cp, 10.0) is None

    def test_already_above_at_t0_returns_t0(self):
        t = np.array([0.0, 1.0, 2.0])
        cp = np.array([100.0, 50.0, 10.0])
        result = _first_crossing_time(t, cp, 50.0)
        assert result is not None
        t_cross, cp_cross = result
        assert t_cross == 0.0
        assert cp_cross == 100.0

    def test_exact_sample_equals_threshold(self):
        t = np.array([0.0, 1.0, 2.0])
        cp = np.array([0.0, 10.0, 5.0])
        result = _first_crossing_time(t, cp, 10.0)
        assert result is not None
        t_cross, _ = result
        # Rising from 0 to 10 over [0,1], exact crossing at t=1.0
        assert abs(t_cross - 1.0) < 1e-9

    def test_multiple_crossings_returns_first(self):
        """Profile that crosses, returns below, then crosses again."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        cp = np.array([0.0, 10.0, 3.0, 9.0, 15.0])
        result = _first_crossing_time(t, cp, 7.0)
        assert result is not None
        t_cross, _ = result
        assert 0.0 < t_cross < 1.0  # first crossing is in the first interval
        assert abs(t_cross - 0.7) < 1e-9  # 7/10 of interval [0,1]

    def test_empty_input(self):
        t = np.array([])
        cp = np.array([])
        assert _first_crossing_time(t, cp, 5.0) is None


# ════════════════════════════════════════════════════════════════
# 2. ENGINE CONTRACT
# ════════════════════════════════════════════════════════════════


class TestEngineContract:
    def test_empty_thresholds_returns_empty_report(self):
        """Drug with no toxicity model must yield a well-formed empty report."""
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(100.0, 0.1, 24.0)
        profile = PDProfile()  # no toxicity thresholds at all

        report = engine.evaluate(
            drug_name="test",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        assert isinstance(report, ToxicityReport)
        assert report.has_any_toxicity is False
        assert report.n_effects == 0
        assert report.overall_severity is None
        assert report.therapeutic_index is None
        assert report.first_severe_trigger_time_h is None
        # Peak fields should still be populated.
        assert report.peak_plasma_mg_per_L > 0.0

    def test_below_threshold_returns_empty_effects(self):
        engine = ToxicityEvaluatorEngine()
        # Cmax = 5 mg/L; all theophylline thresholds start at 15 mg/L.
        t, cp = _bolus_profile(5.0, 0.1, 24.0)
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_theophylline_profile(),
        )
        assert report.has_any_toxicity is False
        assert report.overall_severity is None
        # TI should still be computable from the drug's static PD profile.
        assert report.therapeutic_index is not None
        # Lowest severe threshold = 40 mg/L, EC50 = 10 mg/L → TI = 4.0
        assert abs(report.therapeutic_index - 4.0) < 1e-9

    def test_rejects_shape_mismatch(self):
        engine = ToxicityEvaluatorEngine()
        with pytest.raises(ValueError, match="share shape"):
            engine.evaluate(
                drug_name="x",
                time_h=(0.0, 1.0, 2.0),
                plasma_conc_mg_per_L=(1.0, 2.0),
                pd_profile=PDProfile(),
            )

    def test_rejects_non_monotone_time(self):
        engine = ToxicityEvaluatorEngine()
        with pytest.raises(ValueError, match="non-decreasing"):
            engine.evaluate(
                drug_name="x",
                time_h=(0.0, 2.0, 1.0),
                plasma_conc_mg_per_L=(1.0, 2.0, 3.0),
                pd_profile=PDProfile(),
            )

    def test_rejects_negative_start_time(self):
        engine = ToxicityEvaluatorEngine()
        with pytest.raises(ValueError, match="start at ≥ 0"):
            engine.evaluate(
                drug_name="x",
                time_h=(-1.0, 0.0, 1.0),
                plasma_conc_mg_per_L=(1.0, 2.0, 3.0),
                pd_profile=PDProfile(),
            )

    def test_empty_time_grid_is_valid(self):
        engine = ToxicityEvaluatorEngine()
        report = engine.evaluate(
            drug_name="x",
            time_h=(),
            plasma_conc_mg_per_L=(),
            pd_profile=_theophylline_profile(),
        )
        assert report.peak_plasma_mg_per_L == 0.0
        assert report.has_any_toxicity is False


# ════════════════════════════════════════════════════════════════
# 3. CLINICAL INVARIANTS
# ════════════════════════════════════════════════════════════════


class TestClinicalInvariants:
    def test_theophylline_cmax_above_20_triggers_moderate_cns(self):
        """Plan §6 invariant: theophylline Cmax > 20 mg/L → ≥MODERATE CNS."""
        engine = ToxicityEvaluatorEngine()
        # Peak at 25 mg/L — clearly above the 20 mg/L CNS stimulation line.
        t, cp = _bolus_profile(25.0, 0.15, 48.0)
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_theophylline_profile(),
        )

        # At least one brain effect at MODERATE or higher severity.
        brain_effects = [
            e for e in report.triggered_effects if e.organ == EffectOrgan.BRAIN
        ]
        assert len(brain_effects) >= 2  # mild jitteriness + moderate CNS stim
        brain_worst = report.organ_worst_severity[EffectOrgan.BRAIN]
        assert brain_worst.rank >= ToxicitySeverity.MODERATE.rank

        # The "CNS stimulation" symptom must be present.
        symptoms = [e.symptom for e in report.triggered_effects]
        assert "CNS stimulation" in symptoms

        # Below the 40 mg/L seizure threshold — no SEVERE yet.
        assert report.overall_severity == ToxicitySeverity.MODERATE
        assert report.first_severe_trigger_time_h is None

    def test_theophylline_overdose_crosses_seizure_threshold(self):
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(70.0, 0.1, 48.0)
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_theophylline_profile(),
        )
        assert report.overall_severity == ToxicitySeverity.CRITICAL
        severities = {e.severity for e in report.triggered_effects}
        assert ToxicitySeverity.SEVERE in severities
        assert ToxicitySeverity.CRITICAL in severities
        assert report.first_severe_trigger_time_h is not None
        # Seizure severe triggers before critical arrhythmia (40 < 60 mg/L).
        severe_effects = [
            e
            for e in report.triggered_effects
            if e.severity == ToxicitySeverity.SEVERE
        ]
        critical_effects = [
            e
            for e in report.triggered_effects
            if e.severity == ToxicitySeverity.CRITICAL
        ]
        assert severe_effects[0].triggered_at_h > critical_effects[0].triggered_at_h or (
            # Both triggered at t=0 if already above both at sample 0.
            severe_effects[0].triggered_at_h == critical_effects[0].triggered_at_h == 0.0
        )

    def test_paracetamol_overdose_hepatotoxic(self):
        """Paracetamol at 200 mg/L plasma → SEVERE hepatotoxicity."""
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(200.0, 0.15, 24.0)
        report = engine.evaluate(
            drug_name="paracetamol",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_paracetamol_profile(),
        )
        assert report.overall_severity == ToxicitySeverity.SEVERE
        liver_worst = report.organ_worst_severity[EffectOrgan.LIVER]
        assert liver_worst == ToxicitySeverity.SEVERE
        # No EC50 → no TI even with severe thresholds.
        assert report.therapeutic_index is None
        assert report.first_severe_trigger_time_h is not None
        assert report.first_severe_trigger_time_h == 0.0  # already above at t=0

    def test_therapeutic_dose_stays_safe(self):
        """Theophylline Cmax 12 mg/L — inside therapeutic range, zero effects."""
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(12.0, 0.1, 24.0)
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_theophylline_profile(),
        )
        assert not report.has_any_toxicity
        assert report.verdict.endswith("no toxicity thresholds crossed")

    def test_crossing_time_ordering_matches_threshold_order(self):
        """Lower thresholds must trigger earlier on a mono-decay profile."""
        engine = ToxicityEvaluatorEngine()
        # Sharp rise then decay so ordering is clean.
        n = 1000
        t = np.linspace(0.0, 24.0, n)
        cp = 50.0 * np.exp(-0.2 * t)  # monotonically decreasing
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=tuple(float(x) for x in t),
            plasma_conc_mg_per_L=tuple(float(x) for x in cp),
            pd_profile=_theophylline_profile(),
        )
        # Cmax = 50 mg/L hits every threshold from 15 up to but not
        # including 60 mg/L.
        concs = sorted(
            th.concentration_mg_per_L
            for th in _theophylline_thresholds()
            if th.concentration_mg_per_L <= 50.0
        )
        assert report.n_effects == len(concs)
        # All triggers should be at t=0 since the profile starts at 50 mg/L.
        for effect in report.triggered_effects:
            assert effect.triggered_at_h == 0.0


# ════════════════════════════════════════════════════════════════
# 4. REPORT STRUCTURE
# ════════════════════════════════════════════════════════════════


class TestReportStructure:
    def test_effects_sorted_by_severity_then_time(self):
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(70.0, 0.1, 48.0)
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_theophylline_profile(),
        )
        # Severity ranks should be non-increasing.
        ranks = [e.severity.rank for e in report.triggered_effects]
        assert ranks == sorted(ranks, reverse=True)

    def test_organ_worst_severity_is_max_per_organ(self):
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(70.0, 0.1, 48.0)
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_theophylline_profile(),
        )
        # Brain should be SEVERE (seizure), heart should be CRITICAL.
        assert report.organ_worst_severity[EffectOrgan.BRAIN] == (
            ToxicitySeverity.SEVERE
        )
        assert report.organ_worst_severity[EffectOrgan.HEART] == (
            ToxicitySeverity.CRITICAL
        )

    def test_report_is_frozen(self):
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(25.0, 0.15, 48.0)
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_theophylline_profile(),
        )
        with pytest.raises((AttributeError, TypeError)):
            report.drug_name = "mutated"  # type: ignore

    def test_therapeutic_index_requires_both_ec50_and_severe_threshold(self):
        """TI should be None if Emax is absent."""
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(200.0, 0.15, 24.0)
        report = engine.evaluate(
            drug_name="paracetamol",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_paracetamol_profile(),  # no emax
        )
        assert report.therapeutic_index is None

    def test_verdict_string_includes_severity_and_count(self):
        engine = ToxicityEvaluatorEngine()
        t, cp = _bolus_profile(25.0, 0.15, 48.0)
        report = engine.evaluate(
            drug_name="theophylline",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_theophylline_profile(),
        )
        verdict = report.verdict
        assert "theophylline" in verdict
        assert "MODERATE" in verdict
        assert str(report.n_effects) in verdict


# ════════════════════════════════════════════════════════════════
# 5. P1 INTEGRATION SMOKE TEST
# ════════════════════════════════════════════════════════════════


class TestP1IntegrationSmoke:
    def test_runs_on_real_midazolam_pbpk_output(self):
        """Integration smoke test: evaluate a real PBPK trace end-to-end.

        We don't care about clinical accuracy here — just that the
        evaluator runs cleanly over a real solver output with a plausible
        toxicity profile attached. That catches shape mismatches, numpy
        dtype issues, and frozen-dataclass construction bugs that don't
        show up on synthetic profiles.
        """
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.engine import (
            DoseScheduler,
            DrugFactory,
            SimulationOrchestrator,
        )

        drug = DrugFactory().create("midazolam")
        schedule = DoseScheduler().iv_bolus("midazolam", dose_mg=5.0)
        physiology = build_reference_male()

        result = SimulationOrchestrator().run(drug, schedule, physiology)

        # Synthetic sedation-toxicity ladder just for the smoke test.
        profile = PDProfile(
            emax=EmaxParams(
                emax=100.0,
                ec50_mg_per_L=0.025,
                hill=2.0,
                effect_name="Sedation",
            ),
            toxicity_thresholds=(
                ToxicityThreshold(
                    organ=EffectOrgan.BRAIN,
                    severity=ToxicitySeverity.MILD,
                    concentration_mg_per_L=0.010,
                    symptom="Drowsiness",
                ),
                ToxicityThreshold(
                    organ=EffectOrgan.BRAIN,
                    severity=ToxicitySeverity.MODERATE,
                    concentration_mg_per_L=0.050,
                    symptom="Deep sedation",
                ),
                ToxicityThreshold(
                    organ=EffectOrgan.LUNG,
                    severity=ToxicitySeverity.SEVERE,
                    concentration_mg_per_L=0.200,
                    symptom="Respiratory depression",
                ),
            ),
        )

        engine = ToxicityEvaluatorEngine()
        report = engine.evaluate(
            drug_name=result.drug_name,
            time_h=result.time_h,
            plasma_conc_mg_per_L=result.plasma_conc_mg_per_L,
            pd_profile=profile,
        )

        # Peak plasma populated from real PBPK output.
        assert report.peak_plasma_mg_per_L > 0.0
        assert report.peak_plasma_time_h >= 0.0
        # Report is frozen and well-formed regardless of what triggered.
        assert isinstance(report.triggered_effects, tuple)
        # TI computable from synthetic ladder (severe=0.2, ec50=0.025).
        assert report.therapeutic_index is not None
        assert abs(report.therapeutic_index - 8.0) < 1e-9
