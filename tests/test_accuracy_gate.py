"""Accuracy CI gate — fails the build if honest-mode accuracy regresses.

Run with: NEOGEN_SUPPRESS_KNOWN_DRUGS=1 pytest tests/test_accuracy_gate.py -v

The thresholds are locked to the honest numbers measured in
research_pipeline/metrics/dosed_benchmark.json as of 2026-04-22. Raising
the floors is welcome; lowering them requires a dated CHANGELOG entry.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Floors measured 2026-04-22 on the 30-case FDA DDI benchmark with lookup suppressed.
MIN_COVERAGE = 1.00
MIN_EXACT_SEVERITY = 0.55  # currently 0.567
MIN_ADJACENT_SEVERITY = 1.00  # currently 1.0
MIN_WITHIN_2FOLD = 0.55  # currently 0.567
MAX_GMFE = 2.10  # currently 2.0


@pytest.fixture(autouse=True)
def _suppress_lookup(monkeypatch):
    monkeypatch.setenv("NEOGEN_SUPPRESS_KNOWN_DRUGS", "1")


def _run_dosed_benchmark() -> dict:
    from neogen.simulation.ddi_dose_aware import (
        DoseAwareDDIPredictor, DoseContext, _severity_from_aucr,
    )
    from neogen.validation.datasets.ddi_benchmark_v1 import build_ddi_benchmark_v1

    predictor = DoseAwareDDIPredictor()
    cases = build_ddi_benchmark_v1()

    def severity_distance(a: str, b: str) -> int:
        order = {"none": 0, "minor": 1, "moderate": 2, "major": 3, "disagreement": 2}
        return abs(order.get(a.lower(), 0) - order.get(b.lower(), 0))

    exact = adj = covered = 0
    fold_errors = []
    for c in cases:
        ctx = DoseContext(
            perpetrator_ss_conc_um=c.perpetrator_steady_state_conc_um,
            perpetrator_duration_h=c.perpetrator_duration_on_h,
            victim_dose_mg=c.victim_dose_mg,
            victim_route=c.victim_route,
        )
        p = predictor.predict(c.perpetrator_drug_name, c.victim_drug_name, ctx)
        truth = _severity_from_aucr(c.observed_aucr)
        d = severity_distance(p.predicted_severity, truth)
        if p.confidence > 0:
            covered += 1
            fe = max(p.predicted_aucr / c.observed_aucr, c.observed_aucr / p.predicted_aucr)
            fold_errors.append(fe)
        if d == 0:
            exact += 1
        if d <= 1:
            adj += 1

    n = len(cases)
    gmfe = (
        math.exp(sum(math.log(x) for x in fold_errors) / len(fold_errors))
        if fold_errors else float("inf")
    )
    within_2fold = sum(1 for x in fold_errors if x <= 2.0) / len(fold_errors) if fold_errors else 0.0

    return {
        "n": n,
        "coverage": covered / n,
        "exact_severity": exact / n,
        "adjacent_severity": adj / n,
        "gmfe": gmfe,
        "within_2fold": within_2fold,
    }


@pytest.mark.integration
def test_honest_ddi_accuracy_gates():
    m = _run_dosed_benchmark()
    print(
        "\nHONEST DDI BENCHMARK  coverage={coverage:.3f}  exact={exact_severity:.3f}  "
        "adjacent={adjacent_severity:.3f}  within2fold={within_2fold:.3f}  "
        "gmfe={gmfe:.3f}".format(**m)
    )
    assert m["coverage"] >= MIN_COVERAGE, f"coverage regressed: {m['coverage']:.3f} < {MIN_COVERAGE}"
    assert m["exact_severity"] >= MIN_EXACT_SEVERITY, (
        f"exact severity regressed: {m['exact_severity']:.3f} < {MIN_EXACT_SEVERITY}"
    )
    assert m["adjacent_severity"] >= MIN_ADJACENT_SEVERITY, (
        f"adjacent severity regressed: {m['adjacent_severity']:.3f} < {MIN_ADJACENT_SEVERITY}"
    )
    assert m["within_2fold"] >= MIN_WITHIN_2FOLD, (
        f"within-2-fold regressed: {m['within_2fold']:.3f} < {MIN_WITHIN_2FOLD}"
    )
    assert m["gmfe"] <= MAX_GMFE, f"GMFE regressed: {m['gmfe']:.3f} > {MAX_GMFE}"


@pytest.mark.unit
def test_lookup_suppressed_returns_none():
    """Sanity check: suppress flag actually blocks lookups."""
    from neogen.simulation.known_drugs import lookup_drug
    assert lookup_drug("midazolam") is None, "lookup not suppressed despite env flag"
    assert lookup_drug("atorvastatin") is None


@pytest.mark.unit
def test_holdout_mode_blocks_only_validation_drugs(monkeypatch):
    monkeypatch.delenv("NEOGEN_SUPPRESS_KNOWN_DRUGS", raising=False)
    monkeypatch.setenv("NEOGEN_HOLDOUT_MODE", "1")
    from importlib import reload
    import neogen.simulation.known_drugs as kd_mod
    reload(kd_mod)
    # Holdout should be suppressed
    assert kd_mod.lookup_drug("midazolam") is None
    # Non-holdout drug should still work (pick one guaranteed not in holdout)
    non_holdout = next(
        (name for name in kd_mod.KNOWN_DRUGS if name not in kd_mod.HOLDOUT_DRUGS),
        None,
    )
    assert non_holdout is not None
    assert kd_mod.lookup_drug(non_holdout) is not None
