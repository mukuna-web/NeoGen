"""Interaction entities — DDI mechanism records.

These capture the RESULT of DDI resolution: what happened, which enzyme,
what magnitude, and what clinical significance. They are OUTPUT entities
produced by the DDIResolver (P3) and consumed by the ReportGenerator.

Architecture reference: Layer 7 — DDI Resolution (Blueprint p.4)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DDISeverity(Enum):
    """Clinical DDI severity classification.

    CONTRAINDICATED: Must not co-administer (e.g., ketoconazole + simvastatin)
    MAJOR: Significant risk, monitor closely or adjust dose
    MODERATE: May need dose adjustment
    MINOR: Clinically insignificant, no action needed
    """

    CONTRAINDICATED = "contraindicated"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    NONE = "none"

    @staticmethod
    def from_auc_ratio(auc_ratio: float) -> DDISeverity:
        """Classify DDI severity from predicted AUC fold-change.

        Based on FDA DDI guidance (2020) thresholds:
          ≥5-fold: Contraindicated (strong inhibition)
          ≥2-fold: Major (moderate inhibition)
          ≥1.25-fold: Moderate (weak inhibition)
          <1.25-fold: Minor / None
        """
        if auc_ratio >= 5.0:
            return DDISeverity.CONTRAINDICATED
        if auc_ratio >= 2.0:
            return DDISeverity.MAJOR
        if auc_ratio >= 1.25:
            return DDISeverity.MODERATE
        return DDISeverity.MINOR


class DDIMechanism(Enum):
    """The molecular mechanism driving the DDI."""

    CYP_COMPETITIVE_INHIBITION = "cyp_competitive_inhibition"
    CYP_MECHANISM_BASED_INHIBITION = "cyp_mechanism_based_inhibition"
    CYP_INDUCTION = "cyp_induction"
    PGP_INHIBITION_GUT = "pgp_inhibition_gut"
    PGP_INHIBITION_RENAL = "pgp_inhibition_renal"
    OATP1B1_INHIBITION = "oatp1b1_inhibition"
    OCT2_INHIBITION = "oct2_inhibition"
    PROTEIN_BINDING_DISPLACEMENT = "protein_binding_displacement"
    PD_ADDITIVE = "pd_additive"
    PD_SYNERGISTIC = "pd_synergistic"
    PD_ANTAGONISTIC = "pd_antagonistic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InhibitionEvent:
    """Record of a CYP inhibition DDI.

    Produced by the DDIResolver when Drug B inhibits a CYP enzyme
    that metabolizes Drug A.

    The fm rule quantifies the magnitude:
      AUC_ratio = 1 / (1 - fm + fm / (1 + [I]u / Ki))

    Where:
      fm = fraction of Drug A metabolized by the inhibited enzyme
      [I]u = unbound inhibitor concentration at the enzyme
      Ki = inhibition constant
    """

    perpetrator: str  # Drug doing the inhibiting
    victim: str  # Drug being affected
    enzyme: str  # e.g., "CYP3A4"
    mechanism: DDIMechanism = DDIMechanism.CYP_COMPETITIVE_INHIBITION

    # Key parameters
    fm: float = 0.0  # Fraction metabolized via this enzyme
    ki_um: Optional[float] = None  # Ki (µM)
    inhibitor_conc_unbound_um: Optional[float] = None  # [I]u at enzyme (µM)

    # MBI-specific
    kinact_per_min: Optional[float] = None
    ki_mbi_um: Optional[float] = None

    # Predicted outcome
    auc_ratio: float = 1.0  # Predicted fold-change in victim AUC
    cmax_ratio: float = 1.0  # Predicted fold-change in victim Cmax

    # Classification
    severity: DDISeverity = DDISeverity.NONE

    # Clinical context
    clinical_consequence: str = ""
    recommendation: str = ""

    @property
    def is_clinically_significant(self) -> bool:
        return self.auc_ratio >= 1.25

    def __repr__(self) -> str:
        return (
            f"Inhibition({self.perpetrator} → {self.victim} via {self.enzyme}, "
            f"AUC ×{self.auc_ratio:.1f}, {self.severity.value})"
        )


@dataclass(frozen=True)
class InductionEvent:
    """Record of a CYP induction DDI.

    Induction increases enzyme synthesis → faster victim metabolism → lower AUC.
    Onset is SLOW (days to weeks) because new enzyme protein must be synthesized.

    AUC_ratio < 1.0 for induction (drug is cleared faster).
    """

    perpetrator: str
    victim: str
    enzyme: str
    mechanism: DDIMechanism = DDIMechanism.CYP_INDUCTION

    # Induction parameters
    indmax: float = 0.0  # Maximal fold-induction
    indc50_um: Optional[float] = None  # Concentration for half-max induction
    inducer_conc_unbound_um: Optional[float] = None

    # Enzyme turnover (needed for time-to-new-steady-state)
    kdeg_per_h: float = 0.0193  # Enzyme degradation rate

    # Predicted outcome
    auc_ratio: float = 1.0  # < 1.0 for induction
    cmax_ratio: float = 1.0

    severity: DDISeverity = DDISeverity.NONE
    clinical_consequence: str = ""
    recommendation: str = ""

    @property
    def time_to_steady_state_h(self) -> float:
        """Time for enzyme to reach new steady-state after induction.

        Approximately 4-5 × enzyme half-life.
        """
        if self.kdeg_per_h > 0:
            t_half = 0.693 / self.kdeg_per_h
            return 4.0 * t_half
        return float("inf")

    def __repr__(self) -> str:
        return (
            f"Induction({self.perpetrator} → {self.victim} via {self.enzyme}, "
            f"AUC ×{self.auc_ratio:.2f}, {self.severity.value})"
        )


@dataclass(frozen=True)
class TransporterInhibition:
    """Record of a transporter-mediated DDI.

    P-gp inhibition at gut: increases absorption (higher fa) → higher AUC
    OATP1B1 inhibition at liver: decreases hepatic uptake → higher systemic
    OCT2 inhibition at kidney: decreases renal secretion → higher AUC
    """

    perpetrator: str
    victim: str
    transporter: str  # "P-gp", "OATP1B1", "OCT2"
    location: str  # "gut", "liver", "kidney"
    mechanism: DDIMechanism = DDIMechanism.PGP_INHIBITION_GUT

    ic50_um: Optional[float] = None
    inhibitor_conc_um: Optional[float] = None

    auc_ratio: float = 1.0
    severity: DDISeverity = DDISeverity.NONE
    clinical_consequence: str = ""

    def __repr__(self) -> str:
        return (
            f"TransporterDDI({self.perpetrator} → {self.victim}, "
            f"{self.transporter}@{self.location}, "
            f"AUC ×{self.auc_ratio:.1f})"
        )
