"""Enterohepatic Recirculation (EHR) — Phase 6.3 (Scaffold Only).

Defines data structures and known drug profiles for enterohepatic
recirculation modeling. EHR occurs when a drug is excreted into bile
by the liver, released into the duodenum via gallbladder emptying,
and reabsorbed from the gut back into portal circulation.

Clinical significance
---------------------
EHR produces characteristic secondary plasma peaks (often at 4-8 h
post-dose) and can significantly increase total drug exposure (AUC)
by recycling drug that would otherwise be eliminated. Drugs with
significant EHR include:

* **Mycophenolate mofetil** — 30% biliary excretion, 80% gut
  reabsorption. The characteristic secondary peak at ~6 h is a
  hallmark of EHR (Bullingham 1998).
* **Ezetimibe** — 60% biliary excretion, 90% reabsorption via
  glucuronide hydrolysis in the gut. Long effective half-life
  despite rapid hepatic clearance (Kosoglou 2005).
* **Morphine** — 10% biliary excretion of morphine-3-glucuronide,
  50% reabsorption after gut bacterial deconjugation
  (Hasselstrom 1990).

Implementation status
---------------------
This is a **scaffold only**. The actual ODE integration would require
adding a bile compartment to the PBPK solver with:

1. Biliary excretion flux: ``J_bile = f_bile * CLh * C_liver``
2. Gallbladder accumulation and timed emptying
3. Gut lumen reintroduction and reabsorption

This is Phase 7+ scope. The current module defines the data contract
so that literature parameters can be collected and tested before the
ODE extension is built.

References: Roberts 2002, Gao 2014, FDA Guidance on Food Effect
BA/BE Studies 2002.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ════════════════════════════════════════════════════════════════
# EHR PROFILE
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EHRProfile:
    """Enterohepatic recirculation parameters for a single drug.

    Attributes
    ----------
    biliary_excretion_fraction
        Fraction of hepatic clearance directed to bile (0-1).
        This is the fraction of drug leaving the liver that goes
        into the bile duct rather than back to systemic circulation.
    gut_reabsorption_fraction
        Fraction of biliary-excreted drug reabsorbed from the gut
        (0-1). Depends on gut bacterial deconjugation (for
        glucuronide conjugates) and passive permeability of the
        parent drug.
    gallbladder_emptying_time_h
        Time of bile acid release post-dose (hours). Typically
        corresponds to postprandial gallbladder emptying (~4 h for
        a meal-coincident dose). This determines when the secondary
        peak appears in the plasma concentration profile.
    """

    biliary_excretion_fraction: float = 0.0
    gut_reabsorption_fraction: float = 0.0
    gallbladder_emptying_time_h: float = 4.0

    @property
    def effective_recycling_fraction(self) -> float:
        """Net fraction of hepatic CL that is recycled back to systemic.

        Product of biliary excretion and gut reabsorption fractions.
        This represents the steady-state amplification of exposure:

            AUC_with_EHR / AUC_without_EHR ~ 1 / (1 - f_recycle)

        For mycophenolate: 0.30 * 0.80 = 0.24 -> AUC amplified ~1.3x
        For ezetimibe: 0.60 * 0.90 = 0.54 -> AUC amplified ~2.2x
        """
        return self.biliary_excretion_fraction * self.gut_reabsorption_fraction

    @property
    def has_significant_ehr(self) -> bool:
        """Whether EHR is pharmacokinetically significant.

        Threshold: effective recycling > 5% of hepatic clearance.
        Below this, the secondary peak is negligible and standard
        models are adequate.
        """
        return self.effective_recycling_fraction > 0.05


# ════════════════════════════════════════════════════════════════
# KNOWN EHR DRUGS
# ════════════════════════════════════════════════════════════════

EHR_DRUGS: Dict[str, EHRProfile] = {
    "mycophenolate": EHRProfile(
        biliary_excretion_fraction=0.30,
        gut_reabsorption_fraction=0.80,
        gallbladder_emptying_time_h=6.0,
    ),
    "ezetimibe": EHRProfile(
        biliary_excretion_fraction=0.60,
        gut_reabsorption_fraction=0.90,
        gallbladder_emptying_time_h=4.0,
    ),
    "morphine": EHRProfile(
        biliary_excretion_fraction=0.10,
        gut_reabsorption_fraction=0.50,
        gallbladder_emptying_time_h=4.0,
    ),
    "warfarin": EHRProfile(
        biliary_excretion_fraction=0.05,
        gut_reabsorption_fraction=0.90,
        gallbladder_emptying_time_h=4.0,
    ),
    "indomethacin": EHRProfile(
        biliary_excretion_fraction=0.15,
        gut_reabsorption_fraction=0.60,
        gallbladder_emptying_time_h=4.0,
    ),
}


def get_ehr_profile(drug_name: str) -> Optional[EHRProfile]:
    """Look up known EHR parameters for a drug.

    Parameters
    ----------
    drug_name
        Drug name (case-insensitive).

    Returns
    -------
    EHRProfile or None
        Known EHR parameters, or None if the drug has no published
        EHR data in the registry.
    """
    return EHR_DRUGS.get(drug_name.lower())


def estimate_auc_amplification(profile: EHRProfile) -> float:
    """Estimate the steady-state AUC amplification from EHR.

    Uses the geometric series approximation:

        AUC_ratio = 1 / (1 - f_recycle)

    where f_recycle = biliary_excretion * gut_reabsorption.

    This is an upper bound — actual amplification depends on the
    timing of gallbladder emptying relative to drug elimination.

    Parameters
    ----------
    profile
        EHR parameters for the drug.

    Returns
    -------
    float
        Estimated fold-increase in AUC due to EHR. >= 1.0.
    """
    f_recycle = profile.effective_recycling_fraction
    if f_recycle >= 1.0:
        return float("inf")  # pathological — should not happen
    return 1.0 / (1.0 - f_recycle)
