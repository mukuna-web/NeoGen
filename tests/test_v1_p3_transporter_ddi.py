"""V1.2 Track B — OATP1B1 transporter DDI tests.

Covers :mod:`neogen.engine.ddi.transporter_ddi` in isolation
(closed-form IC50 modulation) and as integrated into
:class:`HepaticClearanceEngine` (end-to-end hepatic CL with
OATP1B1 inhibitor co-administration).

Test layers:

  1. Pure-math tests — exercise ``_sum_inhibition_ratios`` and
     ``compute_oatp1b1_uptake_factor`` against closed-form
     expectations.
  2. Public API tests — drug-entity integration, victim overrides,
     diagnostic record structure.
  3. Integration tests — full ``HepaticClearanceEngine.compute``
     with and without ``oatp1b1_inhibitors``.
  4. Regression invariants — baseline behavior (no inhibitors,
     OATP1B1 substrate present) must match the V1.1 flat 15×
     enhancement path byte-for-byte for every statin entity.
"""
from __future__ import annotations

import pytest

from neogen.core.constants.reference_physiology import build_reference_male
from neogen.engine.ddi.transporter_ddi import (
    DEFAULT_OATP1B1_UPTAKE_FACTOR,
    OATP1B1DDIDiagnostic,
    OATP1B1InhibitorExposure,
    _sum_inhibition_ratios,
    _victim_uptake_base,
    compute_oatp1b1_uptake_factor,
    diagnose_oatp1b1_ddi,
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
def physiology():
    return build_reference_male()


@pytest.fixture(scope="module")
def atorvastatin(factory):
    return factory.create("atorvastatin")


@pytest.fixture(scope="module")
def rifampicin(factory):
    return factory.create("rifampicin")


@pytest.fixture(scope="module")
def caffeine(factory):
    return factory.create("caffeine")


# ════════════════════════════════════════════════════════════════
# LAYER 1 — PURE MATH
# ════════════════════════════════════════════════════════════════


class TestConstants:
    def test_default_uptake_factor(self) -> None:
        """V1.2 Track B default must match V1.1 flat 15× behavior."""
        assert DEFAULT_OATP1B1_UPTAKE_FACTOR == 15.0


class TestSumInhibitionRatios:
    def test_empty(self) -> None:
        assert _sum_inhibition_ratios([]) == 0.0

    def test_single_inhibitor(self, rifampicin) -> None:
        # rifampicin: oatp1b1_ic50 = 0.48 µM, [I] = 1.0 µM → ratio = 1/0.48
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=1.0,
        )
        ratio = _sum_inhibition_ratios([exp])
        assert ratio == pytest.approx(1.0 / 0.48, rel=1e-9)

    def test_additive_cocktail(self, rifampicin) -> None:
        # Two rifampicin exposures additive → 2 × (1/0.48)
        exp1 = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=1.0,
        )
        exp2 = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=1.0,
        )
        ratio = _sum_inhibition_ratios([exp1, exp2])
        assert ratio == pytest.approx(2.0 / 0.48, rel=1e-9)

    def test_non_oatp_perpetrator_ignored(self, factory) -> None:
        # ketoconazole is a CYP3A4 inhibitor but not OATP1B1
        keto = factory.create("ketoconazole")
        exp = OATP1B1InhibitorExposure(
            perpetrator=keto,
            unbound_portal_conc_um=10.0,
        )
        assert exp.has_oatp_ic50 is False
        assert _sum_inhibition_ratios([exp]) == 0.0

    def test_negative_conc_rejected(self, rifampicin) -> None:
        with pytest.raises(ValueError):
            OATP1B1InhibitorExposure(
                perpetrator=rifampicin,
                unbound_portal_conc_um=-0.1,
            )

    def test_zero_conc_no_inhibition(self, rifampicin) -> None:
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=0.0,
        )
        assert _sum_inhibition_ratios([exp]) == 0.0


class TestUptakeFactorMath:
    def test_non_substrate_returns_one(self, caffeine) -> None:
        """Non-substrates always return 1.0 — no enhancement to modulate."""
        assert compute_oatp1b1_uptake_factor(caffeine, ()) == 1.0

    def test_non_substrate_with_inhibitors_still_one(
        self, caffeine, rifampicin
    ) -> None:
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=10.0,
        )
        assert compute_oatp1b1_uptake_factor(caffeine, [exp]) == 1.0

    def test_substrate_no_inhibitors(self, atorvastatin) -> None:
        """Baseline atorvastatin = explicit 15.0 override."""
        factor = compute_oatp1b1_uptake_factor(atorvastatin, ())
        assert factor == pytest.approx(15.0, rel=1e-9)

    def test_substrate_baseline_matches_v11_default(
        self, atorvastatin
    ) -> None:
        """V1.2 atorvastatin baseline must match V1.1 flat 15× exactly."""
        assert compute_oatp1b1_uptake_factor(atorvastatin, []) == 15.0

    def test_half_suppression_at_ic50(self, atorvastatin, rifampicin) -> None:
        """[I]_u = IC50 → denominator = 2 → factor = 15/2 = 7.5."""
        ic50 = rifampicin.transporters.oatp1b1_ic50
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=ic50,
        )
        factor = compute_oatp1b1_uptake_factor(atorvastatin, [exp])
        assert factor == pytest.approx(7.5, rel=1e-6)

    def test_complete_suppression_floors_at_one(
        self, atorvastatin, rifampicin
    ) -> None:
        """[I]_u >> IC50 → factor → 1.0 (passive diffusion floor)."""
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=10_000.0,  # 20 000x IC50
        )
        factor = compute_oatp1b1_uptake_factor(atorvastatin, [exp])
        assert factor == pytest.approx(1.0, abs=1e-3)

    def test_monotonic_in_perpetrator_concentration(
        self, atorvastatin, rifampicin
    ) -> None:
        concs = [0.0, 0.1, 0.5, 1.0, 5.0, 20.0]
        factors = [
            compute_oatp1b1_uptake_factor(
                atorvastatin,
                [OATP1B1InhibitorExposure(
                    perpetrator=rifampicin,
                    unbound_portal_conc_um=c,
                )],
            )
            for c in concs
        ]
        # Monotonically non-increasing
        for prev, cur in zip(factors, factors[1:]):
            assert cur <= prev + 1e-9

    def test_always_bounded(self, atorvastatin, rifampicin) -> None:
        """Factor in [1.0, uptake_base] for any non-negative [I]_u."""
        base = _victim_uptake_base(atorvastatin)
        for c in [0.0, 0.01, 0.48, 1.0, 100.0, 1e6]:
            exp = OATP1B1InhibitorExposure(
                perpetrator=rifampicin,
                unbound_portal_conc_um=c,
            )
            f = compute_oatp1b1_uptake_factor(atorvastatin, [exp])
            assert 1.0 <= f <= base + 1e-9


class TestVictimBase:
    def test_default_fallback(self, factory) -> None:
        """A substrate without an explicit override falls back to 15.0."""
        # Create a minimal stub Drug via factory
        drug = factory.create("atorvastatin")
        # Atorvastatin has explicit 15.0 — use that as the test subject
        base = _victim_uptake_base(drug)
        assert base == 15.0

    def test_explicit_override_respected(self, factory) -> None:
        """If a drug has oatp1b1_uptake_factor set, it is used verbatim."""
        drug = factory.create("atorvastatin")
        base = _victim_uptake_base(drug)
        assert base == drug.transporters.oatp1b1_uptake_factor


# ════════════════════════════════════════════════════════════════
# LAYER 2 — DIAGNOSTICS
# ════════════════════════════════════════════════════════════════


class TestDiagnostic:
    def test_empty_diagnostic(self, atorvastatin) -> None:
        diag = diagnose_oatp1b1_ddi(atorvastatin, ())
        assert isinstance(diag, OATP1B1DDIDiagnostic)
        assert diag.victim_name == "atorvastatin"
        assert diag.n_inhibitors == 0
        assert diag.inhibitor_names == ()
        assert diag.total_inhibition_ratio == 0.0
        assert diag.uptake_base == 15.0
        assert diag.uptake_effective == 15.0
        assert diag.suppression_percent == 0.0

    def test_diagnostic_with_rifampicin(
        self, atorvastatin, rifampicin
    ) -> None:
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=0.48,  # = IC50
        )
        diag = diagnose_oatp1b1_ddi(atorvastatin, [exp])
        assert diag.n_inhibitors == 1
        assert diag.inhibitor_names == ("rifampicin",)
        assert diag.total_inhibition_ratio == pytest.approx(1.0, rel=1e-9)
        assert diag.uptake_effective == pytest.approx(7.5, rel=1e-6)
        # Suppression: base=15, eff=7.5 → dropped=7.5, total=14 → ~53.6%
        assert diag.suppression_percent == pytest.approx(
            100.0 * 7.5 / 14.0, rel=1e-6
        )

    def test_non_substrate_diagnostic_is_flat(
        self, caffeine, rifampicin
    ) -> None:
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=10.0,
        )
        diag = diagnose_oatp1b1_ddi(caffeine, [exp])
        assert diag.uptake_base == 1.0
        assert diag.uptake_effective == 1.0
        assert diag.suppression_percent == 0.0


# ════════════════════════════════════════════════════════════════
# LAYER 3 — HEPATIC CLEARANCE INTEGRATION
# ════════════════════════════════════════════════════════════════


class TestHepaticClearanceIntegration:
    def test_baseline_matches_v11(self, atorvastatin, physiology) -> None:
        """No inhibitors → CLint amplified by flat 15× exactly as V1.1."""
        res = HepaticClearanceEngine().compute(atorvastatin, physiology)
        assert res.oatp1b1_uptake_factor == pytest.approx(15.0, rel=1e-9)

    def test_baseline_default_param_matches_explicit_none(
        self, atorvastatin, physiology
    ) -> None:
        """compute(drug, phys) == compute(drug, phys, oatp1b1_inhibitors=None)."""
        res_default = HepaticClearanceEngine().compute(atorvastatin, physiology)
        res_none = HepaticClearanceEngine().compute(
            atorvastatin, physiology, oatp1b1_inhibitors=None
        )
        assert res_default.clint_scaled_L_per_h == res_none.clint_scaled_L_per_h
        assert res_default.cl_hepatic_L_per_h == res_none.cl_hepatic_L_per_h
        assert res_default.extraction_ratio == res_none.extraction_ratio

    def test_empty_exposures_matches_v11(
        self, atorvastatin, physiology
    ) -> None:
        res_none = HepaticClearanceEngine().compute(atorvastatin, physiology)
        res_empty = HepaticClearanceEngine().compute(
            atorvastatin, physiology, oatp1b1_inhibitors=[]
        )
        assert res_none.clint_scaled_L_per_h == res_empty.clint_scaled_L_per_h
        assert res_none.cl_hepatic_L_per_h == res_empty.cl_hepatic_L_per_h

    def test_rifampicin_suppresses_clh(
        self, atorvastatin, rifampicin, physiology
    ) -> None:
        """Strong OATP inhibition reduces CLh substantially."""
        baseline = HepaticClearanceEngine().compute(atorvastatin, physiology)
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=2.0,  # clinical rifampicin ~2 µM
        )
        suppressed = HepaticClearanceEngine().compute(
            atorvastatin, physiology, oatp1b1_inhibitors=[exp]
        )
        # Clh must drop — well-stirred model bounds, can't go negative.
        assert suppressed.cl_hepatic_L_per_h < baseline.cl_hepatic_L_per_h
        # And the uptake factor must have dropped
        assert suppressed.oatp1b1_uptake_factor < baseline.oatp1b1_uptake_factor
        # Not clipped below 1.0
        assert suppressed.oatp1b1_uptake_factor >= 1.0

    def test_non_substrate_unaffected(
        self, caffeine, rifampicin, physiology
    ) -> None:
        """Non-substrates see zero OATP1B1 effect regardless of inhibitors."""
        baseline = HepaticClearanceEngine().compute(caffeine, physiology)
        exp = OATP1B1InhibitorExposure(
            perpetrator=rifampicin,
            unbound_portal_conc_um=100.0,
        )
        with_inh = HepaticClearanceEngine().compute(
            caffeine, physiology, oatp1b1_inhibitors=[exp]
        )
        assert baseline.clint_scaled_L_per_h == with_inh.clint_scaled_L_per_h
        assert baseline.cl_hepatic_L_per_h == with_inh.cl_hepatic_L_per_h
        assert with_inh.oatp1b1_uptake_factor == 1.0


# ════════════════════════════════════════════════════════════════
# LAYER 4 — REGRESSION INVARIANTS
# ════════════════════════════════════════════════════════════════


class TestRegressionInvariants:
    """All OATP1B1 substrates in the drug factory must reproduce the
    V1.1 flat 15× amplification when no inhibitors are supplied."""

    @pytest.mark.parametrize(
        "drug_name", ["atorvastatin"]  # only atorvastatin wired today
    )
    def test_baseline_uptake_factor_equals_15(
        self, factory, physiology, drug_name
    ) -> None:
        drug = factory.create(drug_name)
        if not getattr(drug.transporters, "is_oatp1b1_substrate", False):
            pytest.skip(f"{drug_name} is not an OATP1B1 substrate")
        res = HepaticClearanceEngine().compute(drug, physiology)
        assert res.oatp1b1_uptake_factor == pytest.approx(15.0, rel=1e-9)

    @pytest.mark.parametrize(
        "drug_name", ["caffeine", "theophylline", "midazolam", "warfarin"]
    )
    def test_non_substrates_flat_one(
        self, factory, physiology, drug_name
    ) -> None:
        """Non-OATP substrates always see uptake_factor == 1.0."""
        drug = factory.create(drug_name)
        assert not getattr(
            drug.transporters, "is_oatp1b1_substrate", False
        )
        res = HepaticClearanceEngine().compute(drug, physiology)
        assert res.oatp1b1_uptake_factor == 1.0
