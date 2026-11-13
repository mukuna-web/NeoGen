"""Tier 6 — DDI Screening at Scale (static model).

Performs rapid DDI screening using the static mechanistic model
(no PBPK simulation) to classify drug pairs as interacting or
non-interacting based on CYP inhibition/induction parameters and
victim fm profiles.

The static DDI model
--------------------

For reversible inhibition::

    R = 1 + [I]_u / Ki
    AUCR_static = 1 / (fm_affected / R + (1 - fm_affected))

For mechanism-based inhibition (MBI)::

    R = kinact * [I]_u / (k_deg * (K_I + [I]_u))
    where k_deg = ln(2) / t_half_enzyme (CYP3A4 ~ 36 h -> 0.019 /h)

For induction::

    R = 1 / (1 + d_max * [I]_u / (EC50 + [I]_u))
    where d_max = Emax (fold induction)

Interaction classification
--------------------------

A pair is predicted as "interacting" if::

    AUCR_static > 1.25  (inhibition — AUC increase)
    or AUCR_static < 0.80  (induction — AUC decrease)

These thresholds follow the FDA DDI guidance (2020) for "no effect"
boundaries.

Performance metrics
-------------------

Sensitivity, specificity, PPV, NPV, and F1 are computed against a
binary ground truth derived from severity labels (if available) or
observed AUCR values.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from neogen.core.entities.drug import Drug

logger = logging.getLogger(__name__)

# CYP3A4 degradation half-life ~ 36 h (Obach 2007)
_CYP3A4_KDEG = math.log(2) / 36.0  # ~ 0.019 /h

# CYP isoforms checked for DDI
_CYP_ISOFORMS = [
    "CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19",
    "CYP1A2", "CYP2C8", "CYP2B6", "CYP2E1",
]


@dataclass(frozen=True)
class StaticDDIResult:
    """Result for a single drug pair from the static DDI model.

    Attributes
    ----------
    perpetrator
        Name of the perpetrator drug.
    victim
        Name of the victim drug.
    aucr_static
        Predicted AUC ratio from the static model.
    predicted_interaction
        True if AUCR > 1.25 or < 0.80.
    mechanism
        Description of the dominant DDI mechanism.
    dominant_cyp
        The CYP isoform mediating the interaction.
    """

    perpetrator: str
    victim: str
    aucr_static: float
    predicted_interaction: bool
    mechanism: str = ""
    dominant_cyp: str = ""


@dataclass
class ScreeningMetrics:
    """Confusion-matrix-derived metrics for DDI screening.

    Attributes
    ----------
    n_pairs
        Total number of pairs screened.
    true_positives, false_positives, true_negatives, false_negatives
        Confusion matrix counts.
    sensitivity, specificity, ppv, npv, f1
        Derived metrics.
    """

    n_pairs: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def sensitivity(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0

    @property
    def ppv(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def npv(self) -> float:
        denom = self.true_negatives + self.false_negatives
        return self.true_negatives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.ppv
        r = self.sensitivity
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_static_aucr(
    perpetrator: Drug,
    victim: Drug,
    perpetrator_conc_um: float = 1.0,
) -> StaticDDIResult:
    """Compute static DDI AUCR for a perpetrator-victim pair.

    Parameters
    ----------
    perpetrator
        Drug entity with inhibition/induction parameters.
    victim
        Drug entity with CYP metabolism profile (fm values).
    perpetrator_conc_um
        Unbound perpetrator concentration at the interaction site (uM).
        Default 1.0 uM if not provided.

    Returns
    -------
    StaticDDIResult
        With the predicted static AUCR and interaction classification.
    """
    best_aucr = 1.0
    best_mechanism = "none"
    best_cyp = ""

    iu = perpetrator_conc_um * perpetrator.fu_plasma_safe

    # Check reversible inhibition for each CYP
    for cyp in _CYP_ISOFORMS:
        ki = perpetrator.inhibition.ki_for(cyp)
        fm = victim.metabolism.fm_for(cyp)

        if ki is not None and ki > 0 and fm > 0.05:
            r = 1.0 + iu / ki
            aucr = 1.0 / (fm / r + (1.0 - fm))

            if aucr > best_aucr:
                best_aucr = aucr
                best_mechanism = f"reversible inhibition ({cyp})"
                best_cyp = cyp

    # Check MBI (currently only CYP3A4)
    if (
        perpetrator.inhibition.kinact_cyp3a4 is not None
        and perpetrator.inhibition.k_i_mbi_cyp3a4 is not None
    ):
        kinact = perpetrator.inhibition.kinact_cyp3a4
        ki_mbi = perpetrator.inhibition.k_i_mbi_cyp3a4
        fm_3a4 = victim.metabolism.cyp3a4

        if fm_3a4 > 0.05 and ki_mbi > 0:
            # MBI: effective enzyme loss ratio
            r_mbi = 1.0 + kinact * iu / (_CYP3A4_KDEG * (ki_mbi + iu))
            aucr_mbi = 1.0 / (fm_3a4 / r_mbi + (1.0 - fm_3a4))

            if aucr_mbi > best_aucr:
                best_aucr = aucr_mbi
                best_mechanism = "mechanism-based inhibition (CYP3A4)"
                best_cyp = "CYP3A4"

    # Check induction (CYP3A4, CYP2B6, CYP1A2, CYP2C9)
    inducible_cyps = ["CYP3A4", "CYP2B6", "CYP1A2", "CYP2C9"]
    for cyp in inducible_cyps:
        indmax, indc50 = perpetrator.induction.params_for(cyp)
        fm = victim.metabolism.fm_for(cyp)

        if indmax is not None and indc50 is not None and fm > 0.05:
            if indc50 > 0:
                fold_increase = 1.0 + indmax * iu / (indc50 + iu)
                # Induction reduces AUC
                aucr_ind = 1.0 / (fm * fold_increase + (1.0 - fm))

                if aucr_ind < 1.0 and (1.0 / aucr_ind) > (1.0 / best_aucr if best_aucr < 1.0 else best_aucr):
                    best_aucr = aucr_ind
                    best_mechanism = f"induction ({cyp})"
                    best_cyp = cyp

    predicted_interaction = best_aucr > 1.25 or best_aucr < 0.80

    return StaticDDIResult(
        perpetrator=perpetrator.name,
        victim=victim.name,
        aucr_static=round(best_aucr, 4),
        predicted_interaction=predicted_interaction,
        mechanism=best_mechanism,
        dominant_cyp=best_cyp,
    )


def screen_drug_pairs(
    pairs: Sequence[Tuple[Drug, Drug]],
    conc_um: float = 1.0,
    observed_interactions: Optional[Dict[Tuple[str, str], bool]] = None,
) -> Tuple[List[StaticDDIResult], Optional[ScreeningMetrics]]:
    """Screen a batch of drug pairs for DDI potential.

    Parameters
    ----------
    pairs
        Sequence of (perpetrator, victim) Drug tuples.
    conc_um
        Default unbound perpetrator concentration (uM).
    observed_interactions
        Optional dict mapping (perpetrator_name, victim_name) -> bool
        indicating whether the pair has a known clinical interaction.
        When provided, confusion matrix metrics are computed.

    Returns
    -------
    (results, metrics)
        List of StaticDDIResult for each pair, and ScreeningMetrics
        if observed_interactions was provided.
    """
    results = []
    metrics = ScreeningMetrics() if observed_interactions is not None else None

    for perpetrator, victim in pairs:
        result = compute_static_aucr(perpetrator, victim, conc_um)
        results.append(result)

        if observed_interactions is not None and metrics is not None:
            key = (perpetrator.name, victim.name)
            if key in observed_interactions:
                metrics.n_pairs += 1
                observed = observed_interactions[key]
                predicted = result.predicted_interaction

                if predicted and observed:
                    metrics.true_positives += 1
                elif predicted and not observed:
                    metrics.false_positives += 1
                elif not predicted and not observed:
                    metrics.true_negatives += 1
                elif not predicted and observed:
                    metrics.false_negatives += 1

    return results, metrics
