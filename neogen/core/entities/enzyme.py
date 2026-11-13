"""Enzyme, EnzymePool, and CYP/Transporter profile entities.

These represent the metabolic and transport machinery of the human body.
The EnzymeKineticsEngine (P3) will use EnzymePool to track enzyme
amounts that change over time due to inhibition, induction, and turnover.

Architecture reference: Layer 5 — Metabolism (Blueprint p.4)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional


class EnzymeLocation(Enum):
    """Where this enzyme is expressed."""

    LIVER = "liver"
    GUT_WALL = "gut_wall"
    KIDNEY = "kidney"
    LUNG = "lung"
    BRAIN = "brain"
    INTESTINE = "intestine"


class EnzymeFamily(Enum):
    """Enzyme superfamily classification."""

    CYP = "cyp"
    UGT = "ugt"
    SULT = "sult"
    NAT = "nat"
    GST = "gst"
    ALDH = "aldh"
    FMO = "fmo"
    MAO = "mao"


@dataclass(frozen=True)
class Enzyme:
    """A single metabolic enzyme isoform with kinetic parameters.

    This is the STATIC definition of an enzyme. For simulation,
    EnzymePool wraps this with a time-varying abundance that changes
    due to inhibition and induction.

    Key parameters for DDI prediction:
    - km: Michaelis constant (µM) — affinity for substrate
    - vmax_pmol_per_min_per_pmol: Catalytic rate per pmol enzyme
    - abundance_pmol_per_mg: Expression level in tissue
    - kdeg: Degradation rate constant (h⁻¹) — critical for MBI/induction
    """

    name: str  # e.g. "CYP3A4"
    family: EnzymeFamily = EnzymeFamily.CYP

    # Kinetic parameters (substrate-independent, isoform-level)
    km_reference: float = 10.0  # µM — typical Km (varies per substrate)
    vmax_pmol_per_min_per_pmol: float = 1.0  # Turnover number

    # Expression
    abundance_liver_pmol_per_mg: float = 0.0  # pmol/mg microsomal protein
    abundance_gut_pmol_per_mg: float = 0.0  # pmol/mg enterocyte protein

    # Turnover (for time-dependent DDI)
    kdeg_liver: float = 0.0193  # h⁻¹ (default ~36h half-life)
    kdeg_gut: float = 0.029  # h⁻¹ (gut turnover ~24h, faster than liver)

    # Locations
    locations: tuple[EnzymeLocation, ...] = (
        EnzymeLocation.LIVER,
        EnzymeLocation.GUT_WALL,
    )

    # Inducibility
    is_inducible: bool = True  # CYP2D6 is NOT inducible (unique)
    induction_receptor: Optional[str] = None  # PXR, AhR, CAR

    # Polymorphism
    is_polymorphic: bool = False
    poor_metabolizer_frequency: float = 0.0  # Population fraction

    @property
    def half_life_liver_h(self) -> float:
        """Enzyme half-life in liver (hours)."""
        if self.kdeg_liver > 0:
            return 0.693 / self.kdeg_liver
        return float("inf")

    @property
    def half_life_gut_h(self) -> float:
        """Enzyme half-life in gut wall (hours)."""
        if self.kdeg_gut > 0:
            return 0.693 / self.kdeg_gut
        return float("inf")

    @property
    def synthesis_rate_liver(self) -> float:
        """Steady-state synthesis rate = kdeg × baseline abundance."""
        return self.kdeg_liver * self.abundance_liver_pmol_per_mg

    @property
    def synthesis_rate_gut(self) -> float:
        """Steady-state synthesis rate in gut."""
        return self.kdeg_gut * self.abundance_gut_pmol_per_mg

    def __repr__(self) -> str:
        return (
            f"Enzyme({self.name}, liver={self.abundance_liver_pmol_per_mg:.0f} "
            f"pmol/mg, gut={self.abundance_gut_pmol_per_mg:.0f} pmol/mg, "
            f"kdeg_liver={self.kdeg_liver:.4f} h⁻¹)"
        )


@dataclass
class EnzymePool:
    """Mutable pool of enzyme that changes during simulation.

    At t=0, amount = baseline abundance.
    During simulation:
      dE/dt = ksyn × (1 + induction_factor) - kdeg × E - kinact × [I] × E / (KI + [I])

    This is what makes DDI TIME-DEPENDENT. Mechanism-based inhibitors
    destroy enzyme (takes days to recover). Inducers increase synthesis
    (takes days to reach new steady-state).
    """

    enzyme: Enzyme
    location: EnzymeLocation

    # Current amount (changes during simulation)
    current_amount_pmol_per_mg: float = 0.0

    # Tracking
    baseline_amount: float = 0.0
    inhibition_factor: float = 0.0  # 0 = no inhibition, 1 = fully inhibited
    induction_factor: float = 0.0  # 0 = no induction, 1 = 2x baseline

    def __post_init__(self):
        if self.location == EnzymeLocation.LIVER:
            if self.current_amount_pmol_per_mg == 0.0:
                self.current_amount_pmol_per_mg = (
                    self.enzyme.abundance_liver_pmol_per_mg
                )
            if self.baseline_amount == 0.0:
                self.baseline_amount = self.enzyme.abundance_liver_pmol_per_mg
        elif self.location == EnzymeLocation.GUT_WALL:
            if self.current_amount_pmol_per_mg == 0.0:
                self.current_amount_pmol_per_mg = (
                    self.enzyme.abundance_gut_pmol_per_mg
                )
            if self.baseline_amount == 0.0:
                self.baseline_amount = self.enzyme.abundance_gut_pmol_per_mg

    @property
    def fraction_remaining(self) -> float:
        """Current enzyme amount as fraction of baseline."""
        if self.baseline_amount > 0:
            return self.current_amount_pmol_per_mg / self.baseline_amount
        return 1.0

    @property
    def effective_activity(self) -> float:
        """Effective enzyme activity accounting for competitive inhibition.

        Competitive inhibition is REVERSIBLE and instantaneous:
        v_effective = v_baseline / (1 + [I]/Ki)

        This is separate from MBI (which destroys enzyme protein).
        """
        return self.current_amount_pmol_per_mg * (1.0 - self.inhibition_factor)


@dataclass(frozen=True)
class CYPProfile:
    """Complete CYP interaction profile for a drug.

    Combines substrate (fm), inhibitor (Ki), and inducer data
    into a single view per drug.
    """

    drug_name: str
    substrates: tuple[str, ...] = ()  # Which CYPs metabolize this drug
    inhibits: tuple[str, ...] = ()  # Which CYPs this drug inhibits
    induces: tuple[str, ...] = ()  # Which CYPs this drug induces

    # Fraction metabolized per isoform
    fm_values: dict[str, float] = field(default_factory=dict)

    # Inhibition constants
    ki_values: dict[str, float] = field(default_factory=dict)  # µM

    # MBI parameters
    kinact_values: dict[str, float] = field(default_factory=dict)  # min⁻¹
    ki_mbi_values: dict[str, float] = field(default_factory=dict)  # µM

    # Induction parameters
    indmax_values: dict[str, float] = field(default_factory=dict)
    indc50_values: dict[str, float] = field(default_factory=dict)  # µM


@dataclass(frozen=True)
class TransporterProfile:
    """Transporter interaction profile for a drug.

    P-gp, OATP1B1, OCT2, BCRP interactions at the drug level.
    """

    drug_name: str
    pgp_substrate: bool = False
    pgp_inhibitor: bool = False
    pgp_ic50: Optional[float] = None

    oatp1b1_substrate: bool = False
    oatp1b1_inhibitor: bool = False
    oatp1b1_ic50: Optional[float] = None

    oct2_substrate: bool = False
    oct2_inhibitor: bool = False
    oct2_ic50: Optional[float] = None

    bcrp_substrate: bool = False
    bcrp_inhibitor: bool = False
    bcrp_ic50: Optional[float] = None
