"""Automated sensitivity analysis for PBPK simulations.

For a given drug and dose, perturb key ADME parameters (CLint, fu, Peff)
by x0.5 and x2.0, rerun the simulation, and compute the relative change
in Cmax, AUC, and t1/2.  Returns a tornado-plot-ready dict.

This is the standard one-at-a-time (OAT) sensitivity analysis used in
regulatory PBPK submissions (FDA Guidance 2018, EMA Guideline 2018).

Usage
-----
::

    from neogen.validation.sensitivity import run_sensitivity_analysis
    from neogen.engine.drug_factory import DrugFactory

    factory = DrugFactory()
    drug = factory.create("midazolam")

    result = run_sensitivity_analysis(drug, dose_mg=5.0, route="ORAL")

    # Tornado-plot data
    for param, perturbations in result.items():
        for direction, changes in perturbations.items():
            print(f"{param} {direction}: AUC change = {changes['auc_change']:.2%}")
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, Optional

from neogen.core.entities.drug import CYPMetabolismProfile, Drug

logger = logging.getLogger(__name__)

# Parameters to perturb and their Drug field names
_PERTURBATION_PARAMS = {
    "CLint": "clint_microsomal",
    "fu": "fu_plasma",
    "Peff": "peff_jejunum",
}

# Default perturbation factors
_LOW_FACTOR = 0.5
_HIGH_FACTOR = 2.0


def _run_single_sim(drug: Drug, dose_mg: float, route: str) -> Optional[dict]:
    """Run a single simulation and extract PK metrics.

    Returns dict with cmax, auc, half_life or None on failure.
    """
    try:
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.engine.dose_scheduler import DoseScheduler
        from neogen.engine.orchestrator_v2 import OrchestratorV2

        engine = OrchestratorV2()
        scheduler = DoseScheduler()
        physiology = build_reference_male()

        if route.upper() == "IV":
            schedule = scheduler.iv_bolus(drug.name, dose_mg=dose_mg)
        else:
            schedule = scheduler.single_oral(drug.name, dose_mg=dose_mg)

        result = engine.run(drug, schedule, physiology)
        pk = result.pk_result.pk_metrics

        return {
            "cmax": pk.cmax_mg_per_L,
            "auc": pk.auc_0_inf_mg_h_per_L,
            "half_life": pk.half_life_h,
        }
    except Exception as exc:
        logger.warning("Sensitivity sim failed: %s", exc)
        return None


def _compute_changes(
    baseline: dict, perturbed: dict
) -> dict:
    """Compute relative changes between baseline and perturbed PK metrics."""
    changes = {}
    for metric in ("cmax", "auc", "half_life"):
        base_val = baseline.get(metric, 0.0)
        pert_val = perturbed.get(metric, 0.0)
        if base_val and base_val > 0:
            changes[f"{metric}_change"] = (pert_val - base_val) / base_val
        else:
            changes[f"{metric}_change"] = 0.0
    return changes


def run_sensitivity_analysis(
    drug: Drug,
    dose_mg: float,
    route: str = "ORAL",
    low_factor: float = _LOW_FACTOR,
    high_factor: float = _HIGH_FACTOR,
) -> Dict[str, Dict[str, dict]]:
    """Run OAT sensitivity analysis on a drug.

    Parameters
    ----------
    drug
        Drug entity (from DrugFactory).
    dose_mg
        Dose in mg.
    route
        "IV" or "ORAL".
    low_factor
        Multiplicative factor for the low perturbation (default 0.5).
    high_factor
        Multiplicative factor for the high perturbation (default 2.0).

    Returns
    -------
    dict
        Tornado-plot-ready structure::

            {
                'CLint_low': {'cmax_change': -0.15, 'auc_change': -0.20, ...},
                'CLint_high': {'cmax_change': +0.18, 'auc_change': +0.25, ...},
                'fu_low': {...},
                'fu_high': {...},
                'Peff_low': {...},
                'Peff_high': {...},
            }

        Change values are relative (e.g., -0.15 means 15% decrease).
        Missing entries indicate the parameter was not available for
        perturbation or the simulation failed.
    """
    logger.info(
        "Starting sensitivity analysis for %s (%.1f mg %s)",
        drug.name, dose_mg, route,
    )

    # Baseline simulation
    baseline = _run_single_sim(drug, dose_mg, route)
    if baseline is None:
        logger.warning("Baseline simulation failed for %s", drug.name)
        return {}

    results = {}

    for param_label, field_name in _PERTURBATION_PARAMS.items():
        base_value = getattr(drug, field_name, None)
        if base_value is None or base_value <= 0:
            logger.debug(
                "Skipping %s perturbation (value=%s)", param_label, base_value
            )
            continue

        # Low perturbation
        try:
            drug_low = drug.with_overrides(**{field_name: base_value * low_factor})
            pk_low = _run_single_sim(drug_low, dose_mg, route)
            if pk_low is not None:
                results[f"{param_label}_low"] = _compute_changes(baseline, pk_low)
        except Exception as exc:
            logger.debug("%s low perturbation failed: %s", param_label, exc)

        # High perturbation
        try:
            drug_high = drug.with_overrides(**{field_name: base_value * high_factor})
            pk_high = _run_single_sim(drug_high, dose_mg, route)
            if pk_high is not None:
                results[f"{param_label}_high"] = _compute_changes(baseline, pk_high)
        except Exception as exc:
            logger.debug("%s high perturbation failed: %s", param_label, exc)

    logger.info(
        "Sensitivity analysis complete for %s: %d perturbations",
        drug.name, len(results),
    )

    return results


def sensitivity_summary(results: Dict[str, Dict[str, dict]]) -> dict:
    """Summarise sensitivity analysis results.

    Returns a dict mapping each parameter to its maximum absolute AUC
    impact, useful for ranking parameter importance.
    """
    summary = {}
    params_seen = set()

    for key, changes in results.items():
        # Extract parameter name (e.g., "CLint" from "CLint_low")
        param = key.rsplit("_", 1)[0]
        params_seen.add(param)

        auc_change = abs(changes.get("auc_change", 0.0))
        if param not in summary or auc_change > summary[param]:
            summary[param] = auc_change

    return summary
