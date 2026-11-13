"""
P2 acceptance gate — Step 6 of the V1 execution plan.

Runs the 5-drug clinical reference panel and asserts that at least 4 of 5
drugs land within a 2× tolerance window of published Cmax AND AUC.

Reference drugs (selected in the plan):
    paracetamol   — BCS I, neutral, high-solubility / high-permeability
    ibuprofen     — BCS II, acid, low-solubility / high-permeability
    metformin     — BCS III, base, high-solubility / low-permeability (OCT2)
    atorvastatin  — BCS II, acid, low-solubility + OATP1B1 uptake
    midazolam     — BCS I, base, CYP3A4 substrate (gut + hepatic first-pass)

Reference Cmax and AUC ranges come from published clinical PK studies
(Goodman & Gilman, FDA labels, Simcyp compound files). The 2× tolerance
matches the plan's "≥4 of 5 within 2-fold" gate for P2.

This gate test is HARD — it gates Phase 3 work. If it fails, the DDI
layer should not be built on top of a miscalibrated absorption engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pytest

from neogen.core.constants.reference_physiology import build_reference_male
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.orchestrator import SimulationOrchestrator


# ════════════════════════════════════════════════════════════════
# REFERENCE PANEL
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ClinicalReference:
    drug_name: str
    dose_mg: float
    cmax_range_mg_per_L: Tuple[float, float]
    auc_inf_range_mg_h_per_L: Tuple[float, float]
    notes: str


REFERENCE_PANEL: Tuple[ClinicalReference, ...] = (
    ClinicalReference(
        drug_name="paracetamol",
        dose_mg=1000.0,
        cmax_range_mg_per_L=(8.0, 20.0),
        auc_inf_range_mg_h_per_L=(25.0, 80.0),
        notes="Rawlins 1977; Prescott 1980 — fast, complete absorption",
    ),
    ClinicalReference(
        drug_name="ibuprofen",
        dose_mg=400.0,
        cmax_range_mg_per_L=(20.0, 40.0),
        auc_inf_range_mg_h_per_L=(60.0, 180.0),
        notes="Davies 1998 — BCS II weak acid, 99% protein bound",
    ),
    ClinicalReference(
        drug_name="metformin",
        dose_mg=500.0,
        cmax_range_mg_per_L=(1.0, 2.0),
        auc_inf_range_mg_h_per_L=(6.0, 15.0),
        notes="Graham 2011 — OCT2/MATE1 substrate, 100% renal",
    ),
    ClinicalReference(
        drug_name="atorvastatin",
        dose_mg=40.0,
        cmax_range_mg_per_L=(0.005, 0.015),
        auc_inf_range_mg_h_per_L=(0.03, 0.1),
        notes="Lennernäs 2003 — OATP1B1 + CYP3A4 gut extraction",
    ),
    ClinicalReference(
        drug_name="midazolam",
        dose_mg=7.5,
        cmax_range_mg_per_L=(0.03, 0.08),
        auc_inf_range_mg_h_per_L=(0.10, 0.40),
        notes="Thummel 1996 — benchmark CYP3A4 victim drug",
    ),
)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _within_2x(value: float, lo: float, hi: float) -> bool:
    """Inclusive 2× envelope around a published clinical range."""
    return (lo / 2.0) <= value <= (hi * 2.0)


@dataclass(frozen=True)
class GateResult:
    drug_name: str
    cmax: float
    auc: float
    cmax_pass: bool
    auc_pass: bool

    @property
    def passed(self) -> bool:
        return self.cmax_pass and self.auc_pass


def _run_panel() -> Tuple[GateResult, ...]:
    orch = SimulationOrchestrator()
    factory = DrugFactory()
    scheduler = DoseScheduler()
    physiology = build_reference_male()

    results = []
    for ref in REFERENCE_PANEL:
        drug = factory.create(ref.drug_name)
        schedule = scheduler.single_oral(ref.drug_name, dose_mg=ref.dose_mg)
        result = orch.run(drug, schedule, physiology)

        cmax = result.pk_metrics.cmax_mg_per_L
        auc = result.pk_metrics.auc_0_inf_mg_h_per_L

        results.append(
            GateResult(
                drug_name=ref.drug_name,
                cmax=cmax,
                auc=auc,
                cmax_pass=_within_2x(cmax, *ref.cmax_range_mg_per_L),
                auc_pass=_within_2x(auc, *ref.auc_inf_range_mg_h_per_L),
            )
        )
    return tuple(results)


# ════════════════════════════════════════════════════════════════
# THE GATE
# ════════════════════════════════════════════════════════════════


class TestP2AcceptanceGate:
    """Phase 2 hard gate: ≥4/5 within 2× of published clinical Cmax AND AUC."""

    @pytest.fixture(scope="class")
    def panel_results(self) -> Tuple[GateResult, ...]:
        return _run_panel()

    def test_gate_at_least_4_of_5_pass(self, panel_results):
        passed = [r for r in panel_results if r.passed]
        summary = "\n".join(
            f"  {r.drug_name:13s} Cmax={r.cmax:10.4f} [{'OK' if r.cmax_pass else 'FAIL'}]"
            f"  AUC={r.auc:10.4f} [{'OK' if r.auc_pass else 'FAIL'}]"
            for r in panel_results
        )
        assert len(passed) >= 4, (
            f"P2 acceptance gate requires ≥4/5 drugs within 2× tolerance.\n"
            f"Observed {len(passed)}/5 passing:\n{summary}"
        )

    def test_paracetamol_in_gate(self, panel_results):
        r = next(x for x in panel_results if x.drug_name == "paracetamol")
        assert r.cmax_pass, f"paracetamol Cmax failed: {r.cmax:.2f}"
        assert r.auc_pass, f"paracetamol AUC failed: {r.auc:.2f}"

    def test_ibuprofen_in_gate(self, panel_results):
        r = next(x for x in panel_results if x.drug_name == "ibuprofen")
        assert r.cmax_pass, f"ibuprofen Cmax failed: {r.cmax:.2f}"
        assert r.auc_pass, f"ibuprofen AUC failed: {r.auc:.2f}"

    def test_metformin_in_gate(self, panel_results):
        r = next(x for x in panel_results if x.drug_name == "metformin")
        assert r.cmax_pass, f"metformin Cmax failed: {r.cmax:.4f}"
        assert r.auc_pass, f"metformin AUC failed: {r.auc:.4f}"

    def test_midazolam_in_gate(self, panel_results):
        r = next(x for x in panel_results if x.drug_name == "midazolam")
        assert r.cmax_pass, f"midazolam Cmax failed: {r.cmax:.4f}"
        assert r.auc_pass, f"midazolam AUC failed: {r.auc:.4f}"

    def test_atorvastatin_known_limitation(self, panel_results):
        """Atorvastatin is documented as failing the 2× window in V1 because
        the gut CYP3A4 first-pass model uses a linear α formulation that
        cannot reach the published Fg ≈ 0.24. This test DOCUMENTS the gap
        and will be converted to a passing assertion once the Qgut model
        lands in V2.

        If atorvastatin ever starts passing, this test will flip and force
        us to either:
          (a) celebrate and convert it to a pass-assertion, or
          (b) investigate whether a regression in another drug accidentally
              brought atorvastatin into range.
        """
        r = next(x for x in panel_results if x.drug_name == "atorvastatin")
        # Expected failure: at least one of Cmax / AUC is outside the window.
        if r.passed:
            pytest.fail(
                "Atorvastatin now passes the acceptance gate — promote this "
                "test to a pass-assertion and update Step 6 status."
            )


# ════════════════════════════════════════════════════════════════
# PROVENANCE SMOKE (prevents silent calibration drift)
# ════════════════════════════════════════════════════════════════


class TestReferenceDataIntegrity:
    """Sanity checks on the calibration panel itself."""

    def test_all_refs_have_positive_ranges(self):
        for ref in REFERENCE_PANEL:
            assert ref.cmax_range_mg_per_L[0] < ref.cmax_range_mg_per_L[1]
            assert ref.auc_inf_range_mg_h_per_L[0] < ref.auc_inf_range_mg_h_per_L[1]
            assert ref.dose_mg > 0

    def test_panel_covers_required_drugs(self):
        names = {r.drug_name for r in REFERENCE_PANEL}
        required = {
            "paracetamol", "ibuprofen", "metformin",
            "atorvastatin", "midazolam",
        }
        assert required.issubset(names)
