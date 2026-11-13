"""
P3 Step 10 — DDI Orchestrator Unit Tests
=========================================
Exercises ``neogen.engine.ddi.ddi_orchestrator`` in isolation from the
PBPK solver. The full PBPK coupling is verified by
``tests/test_v1_p3_ddi_coupling.py`` which contains the acceptance gate.

Scope:
  * ``DDIOrchestrator.build_exposure`` correctness and validation.
  * ``DDIOrchestrator.build_clint_provider`` math:
      - identity behaviour with no perpetrators
      - reversible-only (Step 7) → matches static formula
      - induction-only (Step 9 trace) → matches steady-state product
      - MBI-only (clarithromycin) → suppression
      - mixed perpetrators → multiplicative composition
  * ``well_stirred_clh`` parity with ``HepaticClearanceEngine``.
  * Frozen dataclass invariants.
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
from neogen.core.constants.reference_physiology import build_reference_male
from neogen.engine.ddi import (
    CYPInhibitionEngine,
    DDIOrchestrator,
    EnzymePoolTrace,
    PerpetratorExposure,
    well_stirred_clh,
)
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.hepatic_clearance import HepaticClearanceEngine


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def factory() -> DrugFactory:
    return DrugFactory()


@pytest.fixture(scope="module")
def orch() -> DDIOrchestrator:
    return DDIOrchestrator()


@pytest.fixture(scope="module")
def midazolam(factory: DrugFactory) -> Drug:
    return factory.create("midazolam")


@pytest.fixture(scope="module")
def ketoconazole(factory: DrugFactory) -> Drug:
    return factory.create("ketoconazole")


@pytest.fixture(scope="module")
def rifampicin(factory: DrugFactory) -> Drug:
    return factory.create("rifampicin")


@pytest.fixture(scope="module")
def clarithromycin(factory: DrugFactory) -> Drug:
    return factory.create("clarithromycin")


@pytest.fixture(scope="module")
def midazolam_clint(midazolam: Drug) -> float:
    """Pre-computed baseline whole-liver CLint for the V1 reference male."""
    phys = build_reference_male()
    h = HepaticClearanceEngine().compute(midazolam, phys)
    return h.clint_scaled_L_per_h


# ════════════════════════════════════════════════════════════════
# build_exposure
# ════════════════════════════════════════════════════════════════


class TestBuildExposure:
    def test_reversible_only_no_trace(
        self, orch: DDIOrchestrator, ketoconazole: Drug
    ):
        exp = orch.build_exposure(
            ketoconazole,
            steady_state_conc_um=0.5,
            duration_on_h=24.0,
            include_enzyme_pool=False,
        )
        assert exp.enzyme_trace is None
        assert exp.steady_state_conc_um == 0.5
        assert exp.duration_on_h == 24.0
        assert exp.duration_off_h == 0.0
        assert exp.total_duration_h == 24.0

    def test_with_enzyme_trace(
        self, orch: DDIOrchestrator, rifampicin: Drug
    ):
        exp = orch.build_exposure(
            rifampicin,
            steady_state_conc_um=0.2,
            duration_on_h=14 * 24,
            duration_off_h=7 * 24,
            include_enzyme_pool=True,
        )
        assert exp.enzyme_trace is not None
        assert isinstance(exp.enzyme_trace, EnzymePoolTrace)
        assert exp.total_duration_h == 21 * 24
        # Trace should show real induction at the end of the on-window
        assert exp.enzyme_trace.fold_at(14 * 24, "CYP3A4") > 5.0

    def test_conc_at_step_function(
        self, orch: DDIOrchestrator, ketoconazole: Drug
    ):
        exp = orch.build_exposure(
            ketoconazole, 0.5, duration_on_h=10.0, include_enzyme_pool=False
        )
        assert exp.conc_at(-1.0) == 0.0
        assert exp.conc_at(0.0) == 0.5
        assert exp.conc_at(5.0) == 0.5
        assert exp.conc_at(10.0) == 0.5  # boundary inclusive
        assert exp.conc_at(10.001) == 0.0
        assert exp.conc_at(1000.0) == 0.0

    def test_negative_conc_raises(
        self, orch: DDIOrchestrator, ketoconazole: Drug
    ):
        with pytest.raises(ValueError):
            orch.build_exposure(
                ketoconazole, -0.1, duration_on_h=10.0
            )

    def test_zero_duration_raises(
        self, orch: DDIOrchestrator, ketoconazole: Drug
    ):
        with pytest.raises(ValueError):
            orch.build_exposure(ketoconazole, 0.5, duration_on_h=0.0)

    def test_negative_washout_raises(
        self, orch: DDIOrchestrator, ketoconazole: Drug
    ):
        with pytest.raises(ValueError):
            orch.build_exposure(
                ketoconazole, 0.5, duration_on_h=10.0, duration_off_h=-1.0
            )


# ════════════════════════════════════════════════════════════════
# build_clint_provider — math
# ════════════════════════════════════════════════════════════════


class TestClintProviderMath:
    def test_no_perpetrators_is_identity(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        midazolam_clint: float,
    ):
        provider = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[],
        )
        # Identity: returns base CLint at every time we ask
        for t in (0.0, 1.0, 24.0, 100.0):
            assert math.isclose(provider(t), midazolam_clint, rel_tol=1e-9)

    def test_reversible_only_matches_static_fm_formula(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        ketoconazole: Drug,
        midazolam_clint: float,
    ):
        """For pure reversible Step 7 inhibition with no fold trace,
        the provider should produce
            CLint_eff / CLint_base = (1 - Σ fm) + Σ fm/R
        which is the inverse of the FDA static AUCR formula."""
        exp = orch.build_exposure(
            ketoconazole, 0.5, duration_on_h=24.0, include_enzyme_pool=False
        )
        provider = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        clint_eff = provider(1.0)  # any time inside the on-window

        # Cross-check via Step 7 static engine
        ie = CYPInhibitionEngine()
        state = ie.compute(ketoconazole, 0.5)
        aucr_static = ie.victim_auc_ratio(state, midazolam)
        # CLint scaling = 1 / AUCR
        expected_scale = 1.0 / aucr_static
        actual_scale = clint_eff / midazolam_clint
        assert math.isclose(actual_scale, expected_scale, rel_tol=1e-9)

    def test_reversible_zero_during_washout(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        ketoconazole: Drug,
        midazolam_clint: float,
    ):
        """After the on-window, ketoconazole exposure drops to 0 → no
        inhibition → CLint returns to baseline."""
        exp = orch.build_exposure(
            ketoconazole,
            0.5,
            duration_on_h=24.0,
            include_enzyme_pool=False,
        )
        provider = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        clint_after = provider(48.0)
        assert math.isclose(clint_after, midazolam_clint, rel_tol=1e-9)

    def test_induction_increases_clint(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        rifampicin: Drug,
        midazolam_clint: float,
    ):
        """Rifampicin induction at steady state should INCREASE CLint —
        midazolam fm_3A4 ≈ 0.96 and rifampicin's CYP3A4 fold approaches
        ~5.7× at [I]u = 0.2 µM."""
        exp = orch.build_exposure(
            rifampicin,
            0.2,
            duration_on_h=14 * 24,
            include_enzyme_pool=True,
        )
        provider = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        clint_steady = provider(13 * 24)  # near plateau
        ratio = clint_steady / midazolam_clint
        # Induction expected: scale ≈ (1-0.96) + 0.96 × 5.7 ≈ 5.51
        assert 5.0 <= ratio <= 6.0

    def test_mbi_only_suppresses_clint(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        clarithromycin: Drug,
        midazolam_clint: float,
    ):
        """Clarithromycin MBI at steady state should suppress CLint
        far below baseline. Combined with reversible Ki = 3.3 µM the
        net scaling is dominated by the fold drop to ~3.7%."""
        exp = orch.build_exposure(
            clarithromycin,
            1.0,
            duration_on_h=5 * 24,
            include_enzyme_pool=True,
        )
        provider = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        clint_steady = provider(5 * 24 - 1)
        ratio = clint_steady / midazolam_clint
        # Strong suppression expected: 0.04 - 0.15 range.
        assert 0.02 <= ratio <= 0.15

    def test_clint_floor_never_returns_zero(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        midazolam_clint: float,
    ):
        """Even with extreme MBI, CLint output should not be exactly
        zero — there is a tiny floor to keep the well-stirred recompute
        in the PBPK RHS numerically stable."""
        # Synthetic perpetrator with infinite MBI on CYP3A4
        perp = Drug(
            name="extreme_mbi",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            fu_plasma=0.5,
            metabolism=CYPMetabolismProfile(),
            inhibition=InhibitionProfile(
                kinact_cyp3a4=10.0,  # very fast
                k_i_mbi_cyp3a4=0.01,
            ),
            induction=InductionProfile(),
        )
        exp = orch.build_exposure(
            perp, 5.0, duration_on_h=10 * 24, include_enzyme_pool=True
        )
        provider = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        for t in (1.0, 24.0, 240.0):
            assert provider(t) > 0.0

    def test_combined_perpetrators_multiplicative(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        ketoconazole: Drug,
        rifampicin: Drug,
        midazolam_clint: float,
    ):
        """When two perpetrators act on the same CYP, the per-CYP
        scalings must multiply: induction × inhibition."""
        ket_exp = orch.build_exposure(
            ketoconazole, 0.5, duration_on_h=14 * 24, include_enzyme_pool=False
        )
        rif_exp = orch.build_exposure(
            rifampicin, 0.2, duration_on_h=14 * 24, include_enzyme_pool=True
        )

        # Provider with each perpetrator alone
        prov_ket = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[ket_exp],
        )
        prov_rif = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[rif_exp],
        )
        # Combined provider
        prov_both = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[ket_exp, rif_exp],
        )

        # The COMBINED scaling should be smaller than rifampicin-alone
        # (which boosts CLint) because ketoconazole pulls it back down.
        t = 13 * 24
        s_ket = prov_ket(t) / midazolam_clint
        s_rif = prov_rif(t) / midazolam_clint
        s_both = prov_both(t) / midazolam_clint
        assert s_ket < 1.0  # inhibition lowers CLint
        assert s_rif > 1.0  # induction raises CLint
        assert s_both < s_rif  # combined is below rif-alone
        # And it should be far from either single one (true mix)
        assert abs(s_both - s_rif) > 0.5

    def test_negative_base_clint_raises(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
    ):
        with pytest.raises(ValueError):
            orch.build_clint_provider(
                victim=midazolam,
                base_clint_scaled_L_per_h=-1.0,
                exposures=[],
            )


# ════════════════════════════════════════════════════════════════
# breakdown_at — diagnostic
# ════════════════════════════════════════════════════════════════


class TestCLintBreakdown:
    def test_breakdown_keys_match_cyp_set(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        ketoconazole: Drug,
        midazolam_clint: float,
    ):
        from neogen.engine.ddi import CYP_NAMES

        exp = orch.build_exposure(
            ketoconazole, 0.5, duration_on_h=24.0, include_enzyme_pool=False
        )
        bd = orch.breakdown_at(
            t_h=1.0,
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        cyps_in_breakdown = {cyp for cyp, _ in bd.per_cyp_scaling}
        assert cyps_in_breakdown == set(CYP_NAMES)

    def test_breakdown_cyp3a4_matches_R_inverse(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        ketoconazole: Drug,
        midazolam_clint: float,
    ):
        """For ketoconazole-only the CYP3A4 scaling = 1/R = 1/34.33."""
        exp = orch.build_exposure(
            ketoconazole, 0.5, duration_on_h=24.0, include_enzyme_pool=False
        )
        bd = orch.breakdown_at(
            t_h=1.0,
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        s_3a4 = dict(bd.per_cyp_scaling)["CYP3A4"]
        r_expected = 1.0 + 0.5 / 0.015
        assert math.isclose(s_3a4, 1.0 / r_expected, rel_tol=1e-9)

    def test_breakdown_eff_equals_provider(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        ketoconazole: Drug,
        midazolam_clint: float,
    ):
        """``breakdown_at(t).clint_eff`` should match
        ``provider(t)`` exactly."""
        exp = orch.build_exposure(
            ketoconazole, 0.5, duration_on_h=24.0, include_enzyme_pool=False
        )
        provider = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        bd = orch.breakdown_at(
            t_h=1.0,
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        assert math.isclose(provider(1.0), bd.clint_eff_L_per_h, rel_tol=1e-9)


# ════════════════════════════════════════════════════════════════
# well_stirred_clh helper
# ════════════════════════════════════════════════════════════════


class TestWellStirredClh:
    def test_matches_hepatic_clearance_engine(self, midazolam: Drug):
        """The standalone well-stirred helper used inside the PBPK RHS
        must produce the same CLh as the static
        :class:`HepaticClearanceEngine` for the no-DDI case."""
        phys = build_reference_male()
        h = HepaticClearanceEngine().compute(midazolam, phys)
        clh_dyn = well_stirred_clh(
            h.clint_scaled_L_per_h,
            h.qh_L_per_h,
            midazolam.fu_plasma_safe,
        )
        assert math.isclose(clh_dyn, h.cl_hepatic_L_per_h, rel_tol=1e-3)

    def test_zero_clint_returns_zero(self):
        assert well_stirred_clh(0.0, 90.0, 0.5) == 0.0

    def test_zero_qh_returns_zero(self):
        assert well_stirred_clh(100.0, 0.0, 0.5) == 0.0

    def test_extraction_capped_at_99_percent(self):
        # Extreme CLint should still be bounded
        clh = well_stirred_clh(1e9, 90.0, 0.5)
        assert clh <= 90.0 * 0.99 + 1e-9

    def test_low_extraction_linear_in_clint(self):
        """For very low fu·CLint « Qh, CLh ≈ fu × CLint (linear)."""
        clint = 1.0
        fu = 0.05
        qh = 90.0
        clh = well_stirred_clh(clint, qh, fu)
        assert math.isclose(clh, fu * clint, rel_tol=0.02)


# ════════════════════════════════════════════════════════════════
# Frozen dataclass invariants
# ════════════════════════════════════════════════════════════════


class TestFrozenInvariants:
    def test_perpetrator_exposure_is_frozen(
        self, orch: DDIOrchestrator, ketoconazole: Drug
    ):
        exp = orch.build_exposure(
            ketoconazole, 0.5, duration_on_h=10.0, include_enzyme_pool=False
        )
        with pytest.raises((AttributeError, TypeError)):
            exp.steady_state_conc_um = 999.0  # type: ignore[misc]

    def test_breakdown_is_frozen(
        self,
        orch: DDIOrchestrator,
        midazolam: Drug,
        ketoconazole: Drug,
        midazolam_clint: float,
    ):
        exp = orch.build_exposure(
            ketoconazole, 0.5, duration_on_h=10.0, include_enzyme_pool=False
        )
        bd = orch.breakdown_at(
            t_h=1.0,
            victim=midazolam,
            base_clint_scaled_L_per_h=midazolam_clint,
            exposures=[exp],
        )
        with pytest.raises((AttributeError, TypeError)):
            bd.scaling = 0.5  # type: ignore[misc]
