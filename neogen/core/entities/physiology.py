"""Physiology entities — organ compartments and the whole-body model.

The PhysiologyModel represents a 14-compartment human body with organ
volumes, blood flows, and tissue composition. It is initialized from
ICRP reference values and scaled to individual patients.

Architecture reference: Layer 4 — Distribution (Blueprint p.4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TissueComposition:
    """Tissue composition fractions used for Rodgers-Rowland Kp prediction.

    These fractions determine how much drug partitions into each tissue.
    Values from Rodgers & Rowland (2006) and ICRP Publication 89.

    Rodgers-Rowland Kp depends on:
      - Neutral lipid fraction → lipophilic drug partitioning
      - Phospholipid fraction → amphiphilic drug partitioning
      - Water fraction → hydrophilic drug partitioning
      - Acidic phospholipid fraction → basic drug ion-pairing
    """

    fraction_water: float = 0.7
    fraction_neutral_lipid: float = 0.01
    fraction_phospholipid: float = 0.005
    fraction_acidic_phospholipid: float = 0.0
    fraction_protein: float = 0.1
    ph: float = 7.4  # Intracellular pH


@dataclass(frozen=True)
class OrganCompartment:
    """A single organ compartment in the PBPK model.

    Each organ has:
      - Volume (L) — how much drug it can hold
      - Blood flow (L/h) — how fast drug is delivered/removed
      - Tissue composition — determines Kp (partition coefficient)
      - Optional: metabolic capacity (liver, gut wall)
      - Optional: filtration (kidney)

    Mass balance ODE per organ:
      Vt × dCt/dt = Qt × (Cart - Ct/Kpt)

    For liver, add: hepatic artery + portal vein - clearance
    For kidney, add: filtration + secretion - reabsorption
    """

    name: str
    volume_L: float  # Organ volume in liters
    blood_flow_L_per_h: float  # Blood flow in L/h
    composition: TissueComposition = field(
        default_factory=TissueComposition
    )

    # Whether this organ metabolizes drugs
    has_metabolic_capacity: bool = False

    # Microsomal protein per gram liver (for IVIVE scaling)
    mppgl: Optional[float] = None  # mg microsomal protein / g liver

    # Hepatocytes per gram liver
    hpgl: Optional[float] = None  # million hepatocytes / g liver

    # Liver density for scaling (g/mL)
    density_g_per_ml: float = 1.0

    def __repr__(self) -> str:
        return (
            f"Organ({self.name}, V={self.volume_L:.2f}L, "
            f"Q={self.blood_flow_L_per_h:.1f}L/h)"
        )


@dataclass
class PhysiologyModel:
    """14-compartment whole-body physiology model.

    Initialized from ICRP reference values, then scaled to patient.
    The reference_physiology.py constants module provides the canonical
    values used for initialization.

    Compartments:
      1. Plasma (venous blood)
      2. Lung
      3. Liver (receives hepatic artery + portal vein)
      4. Gut (GI lumen + gut wall)
      5. Kidney
      6. Brain
      7. Heart
      8. Muscle
      9. Fat (adipose)
      10. Skin
      11. Bone
      12. Spleen
      13. Pancreas
      14. Rest (remaining tissues)
    """

    plasma: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="plasma", volume_L=3.0, blood_flow_L_per_h=0.0
    ))
    lung: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="lung", volume_L=0.5, blood_flow_L_per_h=390.0,
        composition=TissueComposition(
            fraction_water=0.626, fraction_neutral_lipid=0.003,
            fraction_phospholipid=0.009, fraction_acidic_phospholipid=0.0036,
            ph=7.4
        )
    ))
    liver: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="liver", volume_L=1.8, blood_flow_L_per_h=90.0,
        has_metabolic_capacity=True, mppgl=40.0, hpgl=120.0,
        density_g_per_ml=1.08,
        composition=TissueComposition(
            fraction_water=0.751, fraction_neutral_lipid=0.014,
            fraction_phospholipid=0.024, fraction_acidic_phospholipid=0.0049,
            ph=7.2
        )
    ))
    gut: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="gut", volume_L=1.2, blood_flow_L_per_h=60.0,
        has_metabolic_capacity=True,
        composition=TissueComposition(
            fraction_water=0.744, fraction_neutral_lipid=0.038,
            fraction_phospholipid=0.011, fraction_acidic_phospholipid=0.0033,
            ph=7.0
        )
    ))
    kidney: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="kidney", volume_L=0.31, blood_flow_L_per_h=76.0,
        composition=TissueComposition(
            fraction_water=0.783, fraction_neutral_lipid=0.012,
            fraction_phospholipid=0.024, fraction_acidic_phospholipid=0.0044,
            ph=7.4
        )
    ))
    brain: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="brain", volume_L=1.4, blood_flow_L_per_h=47.0,
        composition=TissueComposition(
            fraction_water=0.774, fraction_neutral_lipid=0.051,
            fraction_phospholipid=0.056, fraction_acidic_phospholipid=0.0044,
            ph=7.0
        )
    ))
    heart: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="heart", volume_L=0.33, blood_flow_L_per_h=15.0,
        composition=TissueComposition(
            fraction_water=0.758, fraction_neutral_lipid=0.014,
            fraction_phospholipid=0.011, fraction_acidic_phospholipid=0.0044,
            ph=7.0
        )
    ))
    muscle: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="muscle", volume_L=28.0, blood_flow_L_per_h=66.0,
        composition=TissueComposition(
            fraction_water=0.756, fraction_neutral_lipid=0.01,
            fraction_phospholipid=0.009, fraction_acidic_phospholipid=0.0044,
            ph=7.0
        )
    ))
    fat: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="fat", volume_L=14.0, blood_flow_L_per_h=18.0,
        composition=TissueComposition(
            fraction_water=0.174, fraction_neutral_lipid=0.853,
            fraction_phospholipid=0.002, fraction_acidic_phospholipid=0.0004,
            ph=7.0
        )
    ))
    skin: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="skin", volume_L=3.0, blood_flow_L_per_h=18.0,
        composition=TissueComposition(
            fraction_water=0.718, fraction_neutral_lipid=0.028,
            fraction_phospholipid=0.011, fraction_acidic_phospholipid=0.0033,
            ph=7.0
        )
    ))
    bone: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="bone", volume_L=5.5, blood_flow_L_per_h=6.0,
        composition=TissueComposition(
            fraction_water=0.439, fraction_neutral_lipid=0.074,
            fraction_phospholipid=0.011, fraction_acidic_phospholipid=0.001,
            ph=7.0
        )
    ))
    spleen: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="spleen", volume_L=0.18, blood_flow_L_per_h=9.0,
        composition=TissueComposition(
            fraction_water=0.788, fraction_neutral_lipid=0.008,
            fraction_phospholipid=0.013, fraction_acidic_phospholipid=0.0044,
            ph=7.0
        )
    ))
    pancreas: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="pancreas", volume_L=0.1, blood_flow_L_per_h=6.0,
        composition=TissueComposition(
            fraction_water=0.641, fraction_neutral_lipid=0.041,
            fraction_phospholipid=0.011, fraction_acidic_phospholipid=0.0033,
            ph=7.0
        )
    ))
    rest: OrganCompartment = field(default_factory=lambda: OrganCompartment(
        name="rest", volume_L=5.0, blood_flow_L_per_h=9.0,
        composition=TissueComposition(
            fraction_water=0.7, fraction_neutral_lipid=0.02,
            fraction_phospholipid=0.01, fraction_acidic_phospholipid=0.003,
            ph=7.0
        )
    ))

    @property
    def cardiac_output_L_per_h(self) -> float:
        """Total cardiac output — sum of all organ blood flows.

        Reference male: ~390 L/h (~6.5 L/min).
        Lung receives ALL cardiac output, so CO = lung flow.
        The sum of all non-lung, non-plasma organ flows should equal CO.
        """
        return self.lung.blood_flow_L_per_h

    @property
    def total_body_volume_L(self) -> float:
        """Total volume of all compartments."""
        return sum(organ.volume_L for organ in self.all_organs)

    @property
    def all_organs(self) -> list[OrganCompartment]:
        """All organ compartments as a list (ordered for ODE state vector)."""
        return [
            self.plasma,
            self.lung,
            self.liver,
            self.gut,
            self.kidney,
            self.brain,
            self.heart,
            self.muscle,
            self.fat,
            self.skin,
            self.bone,
            self.spleen,
            self.pancreas,
            self.rest,
        ]

    @property
    def organ_names(self) -> list[str]:
        """Names of all compartments (index matches ODE state vector)."""
        return [o.name for o in self.all_organs]

    @property
    def portal_vein_organs(self) -> list[OrganCompartment]:
        """Organs draining into the portal vein (liver input)."""
        return [self.gut, self.spleen, self.pancreas]

    @property
    def portal_vein_flow_L_per_h(self) -> float:
        """Total blood flow entering liver via portal vein."""
        return sum(o.blood_flow_L_per_h for o in self.portal_vein_organs)

    @property
    def hepatic_artery_flow_L_per_h(self) -> float:
        """Hepatic artery flow = total hepatic flow - portal vein flow."""
        return self.liver.blood_flow_L_per_h - self.portal_vein_flow_L_per_h

    def __repr__(self) -> str:
        return (
            f"PhysiologyModel(14 organs, CO={self.cardiac_output_L_per_h:.0f} "
            f"L/h, total_V={self.total_body_volume_L:.1f} L)"
        )
