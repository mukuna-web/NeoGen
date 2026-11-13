"""V1.2 Track A — gut-wall CYP3A4 MBI tests.

Covers :mod:`neogen.engine.ddi.gut_wall_mbi` in isolation
(closed-form steady-state math) and as integrated into
:class:`PermeabilityEngine` (end-to-end absorption with MBI).

Test layers:

  1. Pure-math tests — exercise ``_kobs_from_perpetrator``,
     ``_steady_state_pool_fraction``, and
     ``_combined_kobs_segment`` against closed-form expectations.
  2. Public API tests — ``compute_steady_state_fg_per_segment``
     and ``compute_cumulative_fg`` on known drugs (simvastatin,
     midazolam, caffeine).
  3. Integration tests — full ``PermeabilityEngine.simulate``
     with and without ``gut_wall_mbi_exposures``.
  4. Regression invariants — baseline behavior (no perpetrators)
     must be byte-identical to the V1 static formula.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from neogen.engine.absorption.gut_transit import build_reference_gi_tract
from neogen.engine.absorption.permeability_engine import (
    PermeabilityEngine,
    segment_fg,
    is_gut_cyp3a4_substrate,
)
from neogen.engine.ddi.gut_wall_mbi import (
    DEFAULT_GUT_EXTRACTION_ALPHA,
    ENTEROCYTE_CYP3A4_HALF_LIFE_H,
    ENTEROCYTE_CYP3A4_KDEG_PER_H,
    GUT_LIVER_CYP3A4_RATIO,
    GutWallMBIExposure,
    _combined_kobs_segment,
    _kobs_from_perpetrator,
    _steady_state_pool_fraction,
    baseline_segment_fg,
    compute_cumulative_fg,
    compute_steady_state_fg_per_segment,
    diagnose_gut_wall_mbi,
    mbi_segment_fg,
    qgut_segment_fg,
)
from neogen.engine.hepatic_clearance import HepaticClearanceEngine
from neogen.engine.drug_factory import DrugFactory
from neogen.core.entities.formulation import Formulation, FormulationType
from neogen.core.constants.reference_physiology import build_reference_male


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def factory() -> DrugFactory:
    return DrugFactory()


@pytest.fixture(scope="module")
def tract():
    return build_reference_gi_tract()


@pytest.fixture(scope="module")
def physiology():
    return build_reference_male()


@pytest.fixture(scope="module")
def hce():
    return HepaticClearanceEngine()


# ════════════════════════════════════════════════════════════════
# LAYER 1 — PURE MATH TESTS
# ════════════════════════════════════════════════════════════════


class TestPhysiologicalConstants:
    def test_enterocyte_half_life(self) -> None:
        assert ENTEROCYTE_CYP3A4_HALF_LIFE_H == 23.0

    def test_kdeg_matches_half_life(self) -> None:
        assert ENTEROCYTE_CYP3A4_KDEG_PER_H == pytest.approx(
            math.log(2.0) / 23.0, rel=1e-12
        )

    def test_default_alpha_matches_v1(self) -> None:
        """Must match the V1 permeability_engine constant exactly so
        baseline (no perpetrator) reproduces V1 byte-for-byte."""
        from neogen.engine.absorption.permeability_engine import (
            _DEFAULT_GUT_EXTRACTION_ALPHA,
        )
        assert DEFAULT_GUT_EXTRACTION_ALPHA == _DEFAULT_GUT_EXTRACTION_ALPHA


class TestKobsFromPerpetrator:
    def test_zero_conc_returns_zero(self, factory) -> None:
        drug = factory.create("clarithromycin")
        assert _kobs_from_perpetrator(drug, 0.0) == 0.0

    def test_no_mbi_params_returns_zero(self, factory) -> None:
        """Ketoconazole is a reversible inhibitor — no kinact."""
        drug = factory.create("ketoconazole")
        assert _kobs_from_perpetrator(drug, 10.0) == 0.0

    def test_clarithromycin_saturating(self, factory) -> None:
        """At [I] >> K_I, kobs approaches kinact_per_h (saturation)."""
        drug = factory.create("clarithromycin")
        kinact_per_min = drug.inhibition.kinact_cyp3a4
        kinact_per_h = kinact_per_min * 60.0
        # Very large luminal concentration → kobs ≈ kinact
        kobs = _kobs_from_perpetrator(drug, 100_000.0)
        assert kobs == pytest.approx(kinact_per_h, rel=1e-3)

    def test_clarithromycin_linear_regime(self, factory) -> None:
        """At [I] << K_I, kobs ~ kinact × [I] / K_I (linear)."""
        drug = factory.create("clarithromycin")
        kinact_per_h = drug.inhibition.kinact_cyp3a4 * 60.0
        k_i = drug.inhibition.k_i_mbi_cyp3a4
        low_conc = 0.01 * k_i  # well below K_I
        expected = kinact_per_h * low_conc / k_i
        kobs = _kobs_from_perpetrator(drug, low_conc)
        assert kobs == pytest.approx(expected, rel=0.02)

    def test_negative_conc_returns_zero(self, factory) -> None:
        drug = factory.create("clarithromycin")
        assert _kobs_from_perpetrator(drug, -5.0) == 0.0


class TestSteadyStatePoolFraction:
    def test_no_inhibition_returns_one(self) -> None:
        assert _steady_state_pool_fraction(kobs_per_h=0.0, kdeg_per_h=0.03) == 1.0

    def test_negative_kobs_returns_one(self) -> None:
        assert _steady_state_pool_fraction(kobs_per_h=-1.0, kdeg_per_h=0.03) == 1.0

    def test_kobs_equals_kdeg_gives_half(self) -> None:
        """When kobs == kdeg, pool should settle at 50 percent."""
        e_ss = _steady_state_pool_fraction(kobs_per_h=0.03, kdeg_per_h=0.03)
        assert e_ss == pytest.approx(0.5, rel=1e-6)

    def test_kobs_much_greater_gives_near_zero(self) -> None:
        e_ss = _steady_state_pool_fraction(kobs_per_h=100.0, kdeg_per_h=0.03)
        assert 0.0 < e_ss < 0.001

    def test_bounded_in_unit_interval(self) -> None:
        for kobs in [0.001, 0.03, 1.0, 100.0]:
            e_ss = _steady_state_pool_fraction(kobs_per_h=kobs, kdeg_per_h=0.03)
            assert 0.0 <= e_ss <= 1.0


class TestCombinedKobsSegment:
    def test_empty_exposures_returns_zero(self) -> None:
        assert _combined_kobs_segment([], segment_idx=3) == 0.0

    def test_single_exposure(self, factory) -> None:
        drug = factory.create("clarithromycin")
        exp = GutWallMBIExposure(perpetrator=drug, luminal_conc_um=100.0)
        combined = _combined_kobs_segment([exp], segment_idx=3)
        direct = _kobs_from_perpetrator(drug, 100.0)
        assert combined == pytest.approx(direct, rel=1e-12)

    def test_two_perpetrators_additive(self, factory) -> None:
        """Cocktail of two MBIs should sum kobs contributions."""
        clari = factory.create("clarithromycin")
        # Reuse clarithromycin with a different concentration as a
        # stand-in for a second MBI perpetrator. kinact/K_I are the
        # same but the luminal concentration differs.
        exp_a = GutWallMBIExposure(perpetrator=clari, luminal_conc_um=50.0)
        exp_b = GutWallMBIExposure(perpetrator=clari, luminal_conc_um=200.0)
        combined = _combined_kobs_segment([exp_a, exp_b], segment_idx=3)
        direct_a = _kobs_from_perpetrator(clari, 50.0)
        direct_b = _kobs_from_perpetrator(clari, 200.0)
        assert combined == pytest.approx(direct_a + direct_b, rel=1e-12)

    def test_reversible_only_drug_contributes_nothing(self, factory) -> None:
        keto = factory.create("ketoconazole")  # reversible, no kinact
        exp = GutWallMBIExposure(perpetrator=keto, luminal_conc_um=100.0)
        assert _combined_kobs_segment([exp], segment_idx=3) == 0.0


# ════════════════════════════════════════════════════════════════
# LAYER 2 — PUBLIC API TESTS
# ════════════════════════════════════════════════════════════════


class TestQgutBaseline:
    """Qgut flow-limited fg baseline tests. The Qgut model derives
    fg from CLint and fm_3A4 rather than a calibrated alpha."""

    def test_simvastatin_baseline_fg_below_one(
        self, factory, tract, physiology, hce
    ) -> None:
        drug = factory.create("simvastatin")
        clint = hce._scale_clint(drug, physiology)
        fg = compute_steady_state_fg_per_segment(
            drug, tract, exposures=(),
            clint_liver_scaled_L_per_h=clint,
        )
        # Jejunum_1 (highest CYP3A4) should have fg < 1
        jej1_idx = [i for i, s in enumerate(tract.segments) if s.name == "Jejunum_1"][0]
        assert fg[jej1_idx] < 1.0
        # Stomach (no CYP3A4) should have fg = 1
        assert fg[0] == 1.0

    def test_midazolam_baseline_fg_physiological(
        self, factory, tract, physiology, hce
    ) -> None:
        """Midazolam per-segment fg should be in physiological range.
        Note: the cumulative product overstates extraction because
        it assumes drug passes through ALL segments. The ODE-based
        PermeabilityEngine gives the true Fg (~0.64 for midazolam)."""
        drug = factory.create("midazolam")
        clint = hce._scale_clint(drug, physiology)
        fg = compute_steady_state_fg_per_segment(
            drug, tract, exposures=(),
            clint_liver_scaled_L_per_h=clint,
        )
        # Jejunum_1 fg should be below 1 (extraction) but not zero
        jej1_idx = [i for i, s in enumerate(tract.segments) if s.name == "Jejunum_1"][0]
        assert 0.1 < fg[jej1_idx] < 0.9

    def test_non_substrate_returns_all_ones(
        self, factory, tract, physiology, hce
    ) -> None:
        caffeine = factory.create("caffeine")
        clint = hce._scale_clint(caffeine, physiology)
        fg = compute_steady_state_fg_per_segment(
            caffeine, tract, exposures=(),
            clint_liver_scaled_L_per_h=clint,
        )
        assert np.all(fg == 1.0)

    def test_non_substrate_with_mbi_still_ones(
        self, factory, tract, physiology, hce
    ) -> None:
        """Non-CYP3A4-substrates should be unaffected by gut-wall MBI."""
        caffeine = factory.create("caffeine")
        clari = factory.create("clarithromycin")
        clint = hce._scale_clint(caffeine, physiology)
        exp = GutWallMBIExposure(perpetrator=clari, luminal_conc_um=1000.0)
        fg = compute_steady_state_fg_per_segment(
            caffeine, tract, exposures=[exp],
            clint_liver_scaled_L_per_h=clint,
        )
        assert np.all(fg == 1.0)

    def test_qgut_segment_fg_no_cyp3a4_returns_one(self, factory) -> None:
        """Segments without CYP3A4 should always return fg = 1."""
        from neogen.engine.absorption.gut_transit import GISegment
        seg = GISegment(
            name="Stomach", pH=1.5, volume_mL=250.0,
            surface_area_cm2=3.5, transit_time_h=0.25, radius_cm=5.0,
            has_cyp3a4=False, cyp3a4_activity=0.0,
        )
        drug = factory.create("midazolam")
        fg = qgut_segment_fg(seg, 0, drug=drug, clint_liver_scaled_L_per_h=100.0)
        assert fg == 1.0

    def test_qgut_higher_clint_gives_lower_fg(self, factory, tract) -> None:
        """Higher CLint should produce lower fg (more extraction)."""
        drug = factory.create("midazolam")
        jej1 = tract.segment_by_name("Jejunum_1")
        fg_low = qgut_segment_fg(
            jej1, 2, drug=drug, clint_liver_scaled_L_per_h=50.0
        )
        fg_high = qgut_segment_fg(
            jej1, 2, drug=drug, clint_liver_scaled_L_per_h=500.0
        )
        assert fg_high < fg_low

    def test_qgut_mbi_raises_fg(self, factory, tract) -> None:
        """MBI (e_ss < 1) should raise fg toward 1."""
        drug = factory.create("midazolam")
        jej1 = tract.segment_by_name("Jejunum_1")
        fg_base = qgut_segment_fg(
            jej1, 2, drug=drug, clint_liver_scaled_L_per_h=200.0, e_ss=1.0
        )
        fg_mbi = qgut_segment_fg(
            jej1, 2, drug=drug, clint_liver_scaled_L_per_h=200.0, e_ss=0.1
        )
        assert fg_mbi > fg_base

    def test_qgut_induction_lowers_fg(self, factory, tract) -> None:
        """Induction (e_ss > 1) should lower fg (more extraction)."""
        drug = factory.create("midazolam")
        jej1 = tract.segment_by_name("Jejunum_1")
        fg_base = qgut_segment_fg(
            jej1, 2, drug=drug, clint_liver_scaled_L_per_h=200.0, e_ss=1.0
        )
        fg_ind = qgut_segment_fg(
            jej1, 2, drug=drug, clint_liver_scaled_L_per_h=200.0, e_ss=5.0
        )
        assert fg_ind < fg_base


class TestFgLiftUnderMBI:
    def test_simvastatin_clarithromycin_saturates(
        self, factory, tract, physiology, hce
    ) -> None:
        """At clinical luminal [I] (1000 µM), clarithromycin should
        essentially destroy the simvastatin gut-wall enzyme pool
        and lift fg per segment toward 1.0."""
        drug = factory.create("simvastatin")
        clari = factory.create("clarithromycin")
        clint = hce._scale_clint(drug, physiology)
        exp = GutWallMBIExposure(clari, luminal_conc_um=1000.0)

        fg_base = compute_cumulative_fg(
            drug, tract, exposures=(),
            clint_liver_scaled_L_per_h=clint,
        )
        fg_mbi = compute_cumulative_fg(
            drug, tract, exposures=[exp],
            clint_liver_scaled_L_per_h=clint,
        )
        assert fg_mbi > fg_base
        # Under saturating MBI (e_ss ~ 0.007), per-segment fg ~ 0.99
        # but cumulative product across 5 CYP3A4 segments gives ~0.97
        assert fg_mbi > 0.95  # near-complete escape under saturating MBI

    def test_cumulative_fg_monotone_in_conc(
        self, factory, tract, physiology, hce
    ) -> None:
        """Higher perpetrator concentration must never decrease Fg."""
        drug = factory.create("simvastatin")
        clari = factory.create("clarithromycin")
        clint = hce._scale_clint(drug, physiology)
        fg_at = []
        for conc in [0.0, 1.0, 10.0, 100.0, 1000.0]:
            if conc == 0.0:
                fg = compute_cumulative_fg(
                    drug, tract, exposures=(),
                    clint_liver_scaled_L_per_h=clint,
                )
            else:
                exp = GutWallMBIExposure(clari, luminal_conc_um=conc)
                fg = compute_cumulative_fg(
                    drug, tract, exposures=[exp],
                    clint_liver_scaled_L_per_h=clint,
                )
            fg_at.append(fg)
        for a, b in zip(fg_at, fg_at[1:]):
            assert b >= a - 1e-12

    def test_diagnostic_table_structure(self, factory, tract) -> None:
        drug = factory.create("simvastatin")
        clari = factory.create("clarithromycin")
        exp = GutWallMBIExposure(clari, luminal_conc_um=1000.0)

        diag = diagnose_gut_wall_mbi(drug, tract, exposures=[exp])
        assert len(diag) == len(tract.segments)
        for d in diag:
            assert 0.0 <= d.e_ss <= 1.0
            assert d.fg_baseline <= 1.0
            assert d.fg_under_mbi <= 1.0 + 1e-9
            assert d.fg_lift_ratio >= 0.9999  # MBI never reduces escape

    def test_reversible_perpetrator_has_no_effect(
        self, factory, tract, physiology, hce
    ) -> None:
        """Pure reversible inhibitors (no kinact) must not affect Fg."""
        drug = factory.create("simvastatin")
        keto = factory.create("ketoconazole")
        clint = hce._scale_clint(drug, physiology)
        exp = GutWallMBIExposure(keto, luminal_conc_um=100.0)
        fg_base = compute_cumulative_fg(
            drug, tract, exposures=(),
            clint_liver_scaled_L_per_h=clint,
        )
        fg_with_keto = compute_cumulative_fg(
            drug, tract, exposures=[exp],
            clint_liver_scaled_L_per_h=clint,
        )
        assert fg_with_keto == pytest.approx(fg_base, rel=1e-12)


# ════════════════════════════════════════════════════════════════
# LAYER 3 — PERMEABILITYENGINE INTEGRATION
# ════════════════════════════════════════════════════════════════


class TestPermeabilityEngineIntegration:
    def test_no_mbi_kwarg_matches_no_mbi_empty_list(self, factory) -> None:
        """Passing gut_wall_mbi_exposures=None and =[] should give
        identical results (both mean no-DDI path)."""
        drug = factory.create("simvastatin")
        form = Formulation(formulation_type=FormulationType.IR_TABLET)
        eng = PermeabilityEngine()
        p_none = eng.simulate(drug, form, 40.0)
        p_empty = eng.simulate(drug, form, 40.0, gut_wall_mbi_exposures=[])
        assert p_none.fa == pytest.approx(p_empty.fa, rel=1e-12)
        assert p_none.fg == pytest.approx(p_empty.fg, rel=1e-12)

    def test_simvastatin_fg_lift_under_clarithromycin(self, factory) -> None:
        drug = factory.create("simvastatin")
        clari = factory.create("clarithromycin")
        form = Formulation(formulation_type=FormulationType.IR_TABLET)
        eng = PermeabilityEngine()

        p_base = eng.simulate(drug, form, 40.0)
        exp = GutWallMBIExposure(clari, luminal_conc_um=1000.0)
        p_mbi = eng.simulate(drug, form, 40.0, gut_wall_mbi_exposures=[exp])

        assert p_mbi.fg > p_base.fg
        # Observed lift for simvastatin is ~1.8x after V1.2 calibration.
        assert p_mbi.fg / p_base.fg > 1.5

    def test_midazolam_fg_lift_under_clarithromycin(self, factory) -> None:
        """Midazolam is also a CYP3A4 substrate; gut-wall MBI should
        lift its fg too, though less dramatically than simvastatin
        because midazolam has moderate (not high-CLint) gut extraction."""
        drug = factory.create("midazolam")
        clari = factory.create("clarithromycin")
        form = Formulation(formulation_type=FormulationType.IR_TABLET)
        eng = PermeabilityEngine()

        p_base = eng.simulate(drug, form, 5.0)
        exp = GutWallMBIExposure(clari, luminal_conc_um=1000.0)
        p_mbi = eng.simulate(drug, form, 5.0, gut_wall_mbi_exposures=[exp])

        assert p_mbi.fg > p_base.fg

    def test_non_substrate_unaffected_by_mbi(self, factory) -> None:
        """Caffeine (CYP1A2 substrate, not CYP3A4) should be untouched
        by gut-wall CYP3A4 MBI."""
        drug = factory.create("caffeine")
        clari = factory.create("clarithromycin")
        form = Formulation(formulation_type=FormulationType.IR_TABLET)
        eng = PermeabilityEngine()

        p_base = eng.simulate(drug, form, 100.0)
        exp = GutWallMBIExposure(clari, luminal_conc_um=1000.0)
        p_mbi = eng.simulate(drug, form, 100.0, gut_wall_mbi_exposures=[exp])

        assert p_mbi.fg == pytest.approx(p_base.fg, rel=1e-12)
        assert p_mbi.fa == pytest.approx(p_base.fa, rel=1e-12)


# ════════════════════════════════════════════════════════════════
# LAYER 4 — REGRESSION INVARIANTS
# ════════════════════════════════════════════════════════════════


class TestRegressionInvariants:
    def test_qgut_baseline_bounded_for_all_drugs(
        self, factory, tract, physiology, hce
    ) -> None:
        """For every literature drug, baseline Qgut fg must be
        physiologically bounded. Non-substrates get fg=1; substrates
        get fg in (0, 1)."""
        for name in [
            "midazolam",
            "caffeine",
            "theophylline",
            "warfarin",
            "clarithromycin",
            "ketoconazole",
        ]:
            drug = factory.create(name)
            clint = hce._scale_clint(drug, physiology)
            fg = compute_steady_state_fg_per_segment(
                drug, tract, exposures=(),
                clint_liver_scaled_L_per_h=clint,
            )
            for j, seg in enumerate(tract.segments):
                assert 0.0 <= fg[j] <= 1.0, (
                    f"fg out of bounds for {name} segment {seg.name}"
                )
            # Non-substrates should have all fg = 1
            if not is_gut_cyp3a4_substrate(drug):
                assert np.all(fg == 1.0), f"{name} is non-substrate but fg != 1"

    def test_legacy_path_still_works_without_clint(self, factory, tract) -> None:
        """When clint_liver_scaled is not provided, the legacy linear
        formula should still work (backward compatibility)."""
        drug = factory.create("midazolam")
        fg = compute_steady_state_fg_per_segment(
            drug, tract, exposures=(),
            clint_liver_scaled_L_per_h=None,  # explicitly legacy
        )
        fg_v1 = np.array([
            segment_fg(
                seg,
                is_gut_cyp3a4_substrate(drug),
                DEFAULT_GUT_EXTRACTION_ALPHA,
            )
            for seg in tract.segments
        ])
        assert np.allclose(fg, fg_v1, atol=1e-12)
