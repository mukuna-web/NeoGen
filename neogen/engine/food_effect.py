"""Food Effect Model — Phase 5.1 Regulatory Readiness.

Models the impact of fed-state physiology on oral drug absorption.
The fed state modifies three key physiological parameters:

1. **Gastric pH**: 1.2 (fasted) -> 5.0 (fed)
2. **Gastric emptying t1/2**: 0.25 h (fasted) -> 2.0 h (fed)
3. **Bile salt concentration**: implicit — increases solubility for
   BCS II/IV drugs via bile salt micellar solubilization.

Scientific basis
----------------
Fed-state changes can either increase or decrease oral bioavailability
depending on drug properties:

* **BCS II drugs** (low solubility, high permeability): food typically
  *increases* absorption via bile-salt-mediated solubilization.
  Examples: griseofulvin (+3x), itraconazole (+2x).
* **BCS I drugs** (high solubility, high permeability): food has
  minimal effect on total exposure (AUC) but delays Tmax due to
  slower gastric emptying.
* **BCS III drugs** (high solubility, low permeability): food may
  *decrease* absorption if it interferes with intestinal
  permeability (e.g. chelation).
* **Acid-labile drugs**: higher gastric pH in fed state may *improve*
  stability (e.g. erythromycin estolate).

References: FDA Guidance on Food-Effect BA/BE Studies (2002),
Fleisher 1999, Charman 1997, Singh 2005.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from neogen.core.entities.drug import BCSClass, Drug
from neogen.core.entities.physiology import PhysiologyModel


# ════════════════════════════════════════════════════════════════
# FOOD EFFECT MODIFIER
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FoodEffectModifier:
    """Modifies physiology and absorption for fed-state simulation.

    Attributes
    ----------
    gastric_ph
        Fed-state gastric pH. Typical range 4.0-6.0; default 5.0
        from FDA guidance (fasted = 1.2).
    gastric_emptying_half_life_h
        Fed-state gastric emptying half-life in hours. Default 2.0 h
        (fasted = 0.25 h). Slows drug delivery to the absorptive
        small intestine, delaying Tmax.
    bile_salt_solubility_factor
        Multiplicative factor for aqueous solubility of BCS II/IV
        drugs in fed state. Default 3.0x represents micellar
        solubilization by postprandial bile salts (Dressman 2007).
    """

    gastric_ph: float = 5.0
    gastric_emptying_half_life_h: float = 2.0
    bile_salt_solubility_factor: float = 3.0


# ════════════════════════════════════════════════════════════════
# CORE FOOD EFFECT LOGIC
# ════════════════════════════════════════════════════════════════


def apply_food_effect(
    drug: Drug,
    physiology: PhysiologyModel,
    modifier: FoodEffectModifier = FoodEffectModifier(),
) -> Tuple[Drug, PhysiologyModel]:
    """Return modified (drug, physiology) pair for fed-state simulation.

    Modifications applied:

    1. **Solubility boost** for BCS II and IV drugs: multiply
       ``drug.solubility_mg_per_ml`` by ``modifier.bile_salt_solubility_factor``.
       BCS I and III drugs are unaffected (already highly soluble).
    2. The modifier parameters (gastric pH, emptying t1/2) are stored
       for downstream engines that consume physiological state.

    Parameters
    ----------
    drug
        Fasted-state Drug entity.
    physiology
        Reference (fasted) PhysiologyModel.
    modifier
        Fed-state parameters. Defaults to standard high-fat meal.

    Returns
    -------
    (modified_drug, physiology)
        The drug with adjusted solubility (if applicable) and the
        original physiology (future: modified GI transit model).
    """
    # Modify solubility for BCS II and IV drugs
    new_sol = drug.solubility_mg_per_ml
    if drug.bcs_class in (BCSClass.II, BCSClass.IV) and new_sol is not None:
        new_sol *= modifier.bile_salt_solubility_factor

    # Create modified drug with adjusted solubility
    modified_drug = drug.with_overrides(solubility_mg_per_ml=new_sol)

    # NOTE: Full gastric emptying modification requires changes to
    # GutTransit segments (stomach transit_time_h). This will be wired
    # when the GI transit model supports parameterized emptying in a
    # future phase. For now the solubility modification captures the
    # dominant food effect for BCS II drugs.

    return modified_drug, physiology


# ════════════════════════════════════════════════════════════════
# FOOD EFFECT PREDICTION
# ════════════════════════════════════════════════════════════════


def predict_food_effect(
    drug: Drug,
    schedule: Any,
    physiology: PhysiologyModel,
    orchestrator: Any,
    modifier: FoodEffectModifier = FoodEffectModifier(),
) -> Dict[str, Optional[float]]:
    """Run fasted vs fed simulations and return the food effect ratio.

    Executes two full PBPK simulations (fasted and fed) through the
    provided orchestrator and computes AUC and Cmax ratios.

    Parameters
    ----------
    drug
        The drug entity (fasted-state parameters).
    schedule
        Dose schedule for the simulation.
    physiology
        Reference physiology model.
    orchestrator
        An OrchestratorV2 (or compatible) with a ``run()`` method.
    modifier
        Fed-state modifier parameters.

    Returns
    -------
    dict
        Keys: ``auc_ratio_fed_fasted``, ``cmax_ratio_fed_fasted``.
        Values are float ratios or None if computation failed.

    Examples
    --------
    A BCS II drug with bile-salt-enhanced solubility:

    >>> result = predict_food_effect(drug_bcs2, schedule, phys, orch)
    >>> result['auc_ratio_fed_fasted']  # expect > 1.0
    """
    result_fasted = orchestrator.run(drug, schedule, physiology)

    drug_fed, phys_fed = apply_food_effect(drug, physiology, modifier)
    result_fed = orchestrator.run(drug_fed, schedule, phys_fed)

    # Extract PK metrics from simulation results
    pk_fasted = result_fasted.pk_result.pk_metrics
    pk_fed = result_fed.pk_result.pk_metrics

    auc_fasted = pk_fasted.auc_0_inf_mg_h_per_L
    auc_fed = pk_fed.auc_0_inf_mg_h_per_L
    cmax_fasted = pk_fasted.cmax_mg_per_L
    cmax_fed = pk_fed.cmax_mg_per_L

    auc_ratio: Optional[float] = None
    cmax_ratio: Optional[float] = None

    if auc_fasted is not None and auc_fed is not None and auc_fasted > 0:
        auc_ratio = auc_fed / auc_fasted

    if cmax_fasted is not None and cmax_fed is not None and cmax_fasted > 0:
        cmax_ratio = cmax_fed / cmax_fasted

    return {
        "auc_ratio_fed_fasted": auc_ratio,
        "cmax_ratio_fed_fasted": cmax_ratio,
    }
