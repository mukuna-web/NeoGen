"""Patient entity — individual physiology, organ function, and genotype.

The patient profile determines how drug PK varies from the reference human.
Age, weight, sex, organ function, and CYP genotype all modify the simulation.

Architecture reference: Core Data Entities (Blueprint p.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


class CYPPhenotype(Enum):
    """CYP metabolizer phenotype — determines enzyme activity scaling.

    Poor metabolizers: ~0% activity (homozygous loss-of-function)
    Intermediate: ~50% activity
    Normal (Extensive): 100% activity (reference)
    Ultra-rapid: >100% activity (gene duplication)
    """

    POOR = "poor_metabolizer"
    INTERMEDIATE = "intermediate_metabolizer"
    NORMAL = "normal_metabolizer"
    ULTRA_RAPID = "ultra_rapid_metabolizer"

    @property
    def activity_factor(self) -> float:
        """Multiplicative factor applied to enzyme Vmax."""
        return {
            CYPPhenotype.POOR: 0.0,
            CYPPhenotype.INTERMEDIATE: 0.5,
            CYPPhenotype.NORMAL: 1.0,
            CYPPhenotype.ULTRA_RAPID: 2.0,
        }[self]


@dataclass(frozen=True)
class GeneticProfile:
    """CYP pharmacogenomic profile.

    CYP2D6 polymorphism is the most clinically important:
      - Poor metabolizers: codeine → no morphine (no analgesia)
      - Ultra-rapid: codeine → excess morphine (respiratory depression)

    CYP2C19 polymorphism affects clopidogrel activation:
      - Poor metabolizers: no active metabolite → no antiplatelet effect
    """

    cyp2d6: CYPPhenotype = CYPPhenotype.NORMAL
    cyp2c19: CYPPhenotype = CYPPhenotype.NORMAL
    cyp2c9: CYPPhenotype = CYPPhenotype.NORMAL
    cyp2b6: CYPPhenotype = CYPPhenotype.NORMAL
    cyp1a2: CYPPhenotype = CYPPhenotype.NORMAL
    cyp3a4: CYPPhenotype = CYPPhenotype.NORMAL

    def activity_factor_for(self, enzyme_name: str) -> float:
        """Get activity scaling factor for a specific CYP enzyme."""
        mapping = {
            "CYP2D6": self.cyp2d6,
            "CYP2C19": self.cyp2c19,
            "CYP2C9": self.cyp2c9,
            "CYP2B6": self.cyp2b6,
            "CYP1A2": self.cyp1a2,
            "CYP3A4": self.cyp3a4,
        }
        phenotype = mapping.get(enzyme_name.upper(), CYPPhenotype.NORMAL)
        return phenotype.activity_factor


@dataclass(frozen=True)
class OrganFunction:
    """Organ function scores — used to scale clearance in impairment.

    Each score is 0.0 (no function) to 1.0 (normal function).
    These map to clinical staging:
      Hepatic: Child-Pugh A (~0.7), B (~0.5), C (~0.3)
      Renal: eGFR-based (mild 0.7, moderate 0.5, severe 0.3, dialysis 0.1)
      Cardiac: Ejection fraction / output scaling
    """

    hepatic: float = 1.0  # 1.0 = normal, <1.0 = impaired
    renal: float = 1.0  # 1.0 = normal (GFR 120), 0.5 = moderate (GFR 60)
    cardiac: float = 1.0  # 1.0 = normal CO, <1.0 = reduced
    pulmonary: float = 1.0

    @property
    def gfr_ml_per_min(self) -> float:
        """Estimated GFR from renal function score."""
        return self.renal * 120.0  # Reference GFR = 120 mL/min

    @property
    def hepatic_blood_flow_fraction(self) -> float:
        """Liver blood flow as fraction of normal (reduced in cirrhosis)."""
        return max(self.hepatic, 0.1)

    def __post_init__(self):
        # Validate ranges via object.__setattr__ since frozen
        for attr in ("hepatic", "renal", "cardiac", "pulmonary"):
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"OrganFunction.{attr} must be 0.0-1.0, got {val}"
                )


@dataclass(frozen=True)
class PatientProfile:
    """Complete patient profile for individualized PBPK simulation.

    Default values represent the ICRP reference male (70 kg, 30 years).
    The PhysiologyModel uses these values to scale organ volumes and
    blood flows allometrically.
    """

    weight_kg: float = 70.0
    height_cm: float = 176.0
    age_years: float = 30.0
    sex: Sex = Sex.MALE

    organ_function: OrganFunction = field(default_factory=OrganFunction)
    genetics: GeneticProfile = field(default_factory=GeneticProfile)

    # Lifestyle factors
    is_smoker: bool = False  # Induces CYP1A2 (theophylline, caffeine)
    is_fasted: bool = True  # Fed state changes GI transit + bile

    # Optional: measured lab values override predictions
    measured_albumin_g_per_dl: Optional[float] = None  # Normal: 3.5-5.0
    measured_aag_mg_per_dl: Optional[float] = None  # Alpha-1 acid glycoprotein

    @property
    def bmi(self) -> float:
        """Body mass index."""
        height_m = self.height_cm / 100.0
        return self.weight_kg / (height_m * height_m)

    @property
    def bsa_m2(self) -> float:
        """Body surface area (Du Bois formula)."""
        return 0.007184 * (self.weight_kg**0.425) * (self.height_cm**0.725)

    @property
    def lean_body_mass_kg(self) -> float:
        """Lean body mass (Janmahasatian formula)."""
        if self.sex == Sex.MALE:
            return (
                9270.0
                * self.weight_kg
                / (6680.0 + 216.0 * self.bmi)
            )
        return (
            9270.0
            * self.weight_kg
            / (8780.0 + 244.0 * self.bmi)
        )

    @property
    def is_pediatric(self) -> bool:
        return self.age_years < 18

    @property
    def is_elderly(self) -> bool:
        return self.age_years >= 65

    @property
    def is_obese(self) -> bool:
        return self.bmi >= 30

    def __repr__(self) -> str:
        return (
            f"Patient({self.sex.value}, {self.age_years:.0f}y, "
            f"{self.weight_kg:.0f}kg, "
            f"hepatic={self.organ_function.hepatic:.1f}, "
            f"renal={self.organ_function.renal:.1f})"
        )
