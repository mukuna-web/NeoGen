"""ICRP Reference Human Physiology Constants.

Standard physiological parameters for a 70 kg, 176 cm, 30-year-old male.
Sources:
  - ICRP Publication 89 (2002): Reference human organ masses/volumes
  - Brown et al. (1997): Organ blood flows
  - Simcyp V21 population library (validation cross-reference)
  - Rodgers & Rowland (2006): Tissue compositions

These are the BASELINE values. PatientProfile scaling adjusts them
for individual patients via allometric equations.
"""

from neogen.core.entities.physiology import (
    OrganCompartment,
    PhysiologyModel,
    TissueComposition,
)

# ═══════════════════════════════════════════════════════════════════
# TISSUE COMPOSITIONS (Rodgers & Rowland 2006, Table 1)
# Used for Kp prediction — partition coefficient computation
# ═══════════════════════════════════════════════════════════════════

COMPOSITION_PLASMA = TissueComposition(
    fraction_water=0.945,
    fraction_neutral_lipid=0.0023,
    fraction_phospholipid=0.0012,
    fraction_acidic_phospholipid=0.0,
    fraction_protein=0.042,
    ph=7.4,
)

COMPOSITION_LUNG = TissueComposition(
    fraction_water=0.626,
    fraction_neutral_lipid=0.003,
    fraction_phospholipid=0.009,
    fraction_acidic_phospholipid=0.0036,
    ph=7.4,
)

COMPOSITION_LIVER = TissueComposition(
    fraction_water=0.751,
    fraction_neutral_lipid=0.014,
    fraction_phospholipid=0.024,
    fraction_acidic_phospholipid=0.0049,
    ph=7.2,
)

COMPOSITION_GUT = TissueComposition(
    fraction_water=0.744,
    fraction_neutral_lipid=0.038,
    fraction_phospholipid=0.011,
    fraction_acidic_phospholipid=0.0033,
    ph=7.0,
)

COMPOSITION_KIDNEY = TissueComposition(
    fraction_water=0.783,
    fraction_neutral_lipid=0.012,
    fraction_phospholipid=0.024,
    fraction_acidic_phospholipid=0.0044,
    ph=7.4,
)

COMPOSITION_BRAIN = TissueComposition(
    fraction_water=0.774,
    fraction_neutral_lipid=0.051,
    fraction_phospholipid=0.056,
    fraction_acidic_phospholipid=0.0044,
    ph=7.0,
)

COMPOSITION_HEART = TissueComposition(
    fraction_water=0.758,
    fraction_neutral_lipid=0.014,
    fraction_phospholipid=0.011,
    fraction_acidic_phospholipid=0.0044,
    ph=7.0,
)

COMPOSITION_MUSCLE = TissueComposition(
    fraction_water=0.756,
    fraction_neutral_lipid=0.010,
    fraction_phospholipid=0.009,
    fraction_acidic_phospholipid=0.0044,
    ph=7.0,
)

COMPOSITION_FAT = TissueComposition(
    fraction_water=0.174,
    fraction_neutral_lipid=0.853,
    fraction_phospholipid=0.002,
    fraction_acidic_phospholipid=0.0004,
    ph=7.0,
)

COMPOSITION_SKIN = TissueComposition(
    fraction_water=0.718,
    fraction_neutral_lipid=0.028,
    fraction_phospholipid=0.011,
    fraction_acidic_phospholipid=0.0033,
    ph=7.0,
)

COMPOSITION_BONE = TissueComposition(
    fraction_water=0.439,
    fraction_neutral_lipid=0.074,
    fraction_phospholipid=0.011,
    fraction_acidic_phospholipid=0.0010,
    ph=7.0,
)

COMPOSITION_SPLEEN = TissueComposition(
    fraction_water=0.788,
    fraction_neutral_lipid=0.008,
    fraction_phospholipid=0.013,
    fraction_acidic_phospholipid=0.0044,
    ph=7.0,
)

COMPOSITION_PANCREAS = TissueComposition(
    fraction_water=0.641,
    fraction_neutral_lipid=0.041,
    fraction_phospholipid=0.011,
    fraction_acidic_phospholipid=0.0033,
    ph=7.0,
)

COMPOSITION_REST = TissueComposition(
    fraction_water=0.700,
    fraction_neutral_lipid=0.020,
    fraction_phospholipid=0.010,
    fraction_acidic_phospholipid=0.0030,
    ph=7.0,
)


# ═══════════════════════════════════════════════════════════════════
# REFERENCE PHYSIOLOGY MODEL — 70 kg Male
# Organ volumes: ICRP Publication 89
# Blood flows: Brown et al. (1997), validated against Simcyp
# ═══════════════════════════════════════════════════════════════════

def build_reference_male() -> PhysiologyModel:
    """Build the ICRP reference male physiology model.

    70 kg, 176 cm, 30 years old.
    Cardiac output: ~390 L/h (6.5 L/min).
    """
    return PhysiologyModel(
        plasma=OrganCompartment(
            name="plasma", volume_L=3.0, blood_flow_L_per_h=0.0,
            composition=COMPOSITION_PLASMA,
        ),
        lung=OrganCompartment(
            name="lung", volume_L=0.50, blood_flow_L_per_h=390.0,
            composition=COMPOSITION_LUNG,
        ),
        liver=OrganCompartment(
            name="liver", volume_L=1.80, blood_flow_L_per_h=90.0,
            has_metabolic_capacity=True, mppgl=40.0, hpgl=120.0,
            density_g_per_ml=1.08,
            composition=COMPOSITION_LIVER,
        ),
        gut=OrganCompartment(
            name="gut", volume_L=1.20, blood_flow_L_per_h=60.0,
            has_metabolic_capacity=True,
            composition=COMPOSITION_GUT,
        ),
        kidney=OrganCompartment(
            name="kidney", volume_L=0.31, blood_flow_L_per_h=76.0,
            composition=COMPOSITION_KIDNEY,
        ),
        brain=OrganCompartment(
            name="brain", volume_L=1.40, blood_flow_L_per_h=47.0,
            composition=COMPOSITION_BRAIN,
        ),
        heart=OrganCompartment(
            name="heart", volume_L=0.33, blood_flow_L_per_h=15.0,
            composition=COMPOSITION_HEART,
        ),
        muscle=OrganCompartment(
            name="muscle", volume_L=28.0, blood_flow_L_per_h=66.0,
            composition=COMPOSITION_MUSCLE,
        ),
        fat=OrganCompartment(
            name="fat", volume_L=14.0, blood_flow_L_per_h=18.0,
            composition=COMPOSITION_FAT,
        ),
        skin=OrganCompartment(
            name="skin", volume_L=3.0, blood_flow_L_per_h=18.0,
            composition=COMPOSITION_SKIN,
        ),
        bone=OrganCompartment(
            name="bone", volume_L=5.5, blood_flow_L_per_h=6.0,
            composition=COMPOSITION_BONE,
        ),
        spleen=OrganCompartment(
            name="spleen", volume_L=0.18, blood_flow_L_per_h=9.0,
            composition=COMPOSITION_SPLEEN,
        ),
        pancreas=OrganCompartment(
            name="pancreas", volume_L=0.10, blood_flow_L_per_h=6.0,
            composition=COMPOSITION_PANCREAS,
        ),
        rest=OrganCompartment(
            name="rest", volume_L=5.0, blood_flow_L_per_h=9.0,
            composition=COMPOSITION_REST,
        ),
    )


def build_reference_female() -> PhysiologyModel:
    """Build the ICRP reference female physiology model.

    60 kg, 163 cm, 30 years old.
    Cardiac output: ~330 L/h (5.5 L/min).
    Smaller organs, more fat, less muscle.
    """
    return PhysiologyModel(
        plasma=OrganCompartment(
            name="plasma", volume_L=2.6, blood_flow_L_per_h=0.0,
            composition=COMPOSITION_PLASMA,
        ),
        lung=OrganCompartment(
            name="lung", volume_L=0.42, blood_flow_L_per_h=330.0,
            composition=COMPOSITION_LUNG,
        ),
        liver=OrganCompartment(
            name="liver", volume_L=1.40, blood_flow_L_per_h=76.5,
            has_metabolic_capacity=True, mppgl=40.0, hpgl=120.0,
            density_g_per_ml=1.08,
            composition=COMPOSITION_LIVER,
        ),
        gut=OrganCompartment(
            name="gut", volume_L=1.00, blood_flow_L_per_h=51.0,
            has_metabolic_capacity=True,
            composition=COMPOSITION_GUT,
        ),
        kidney=OrganCompartment(
            name="kidney", volume_L=0.28, blood_flow_L_per_h=64.6,
            composition=COMPOSITION_KIDNEY,
        ),
        brain=OrganCompartment(
            name="brain", volume_L=1.30, blood_flow_L_per_h=39.9,
            composition=COMPOSITION_BRAIN,
        ),
        heart=OrganCompartment(
            name="heart", volume_L=0.25, blood_flow_L_per_h=12.75,
            composition=COMPOSITION_HEART,
        ),
        muscle=OrganCompartment(
            name="muscle", volume_L=17.5, blood_flow_L_per_h=56.1,
            composition=COMPOSITION_MUSCLE,
        ),
        fat=OrganCompartment(
            name="fat", volume_L=19.0, blood_flow_L_per_h=15.3,
            composition=COMPOSITION_FAT,
        ),
        skin=OrganCompartment(
            name="skin", volume_L=2.3, blood_flow_L_per_h=15.3,
            composition=COMPOSITION_SKIN,
        ),
        bone=OrganCompartment(
            name="bone", volume_L=4.0, blood_flow_L_per_h=5.1,
            composition=COMPOSITION_BONE,
        ),
        spleen=OrganCompartment(
            name="spleen", volume_L=0.15, blood_flow_L_per_h=7.65,
            composition=COMPOSITION_SPLEEN,
        ),
        pancreas=OrganCompartment(
            name="pancreas", volume_L=0.085, blood_flow_L_per_h=5.1,
            composition=COMPOSITION_PANCREAS,
        ),
        rest=OrganCompartment(
            name="rest", volume_L=4.0, blood_flow_L_per_h=7.65,
            composition=COMPOSITION_REST,
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# SCALING CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# Reference GFR (mL/min) — for renal clearance scaling
REFERENCE_GFR_ML_PER_MIN = 120.0

# Liver density (g/mL) — for IVIVE scaling
LIVER_DENSITY_G_PER_ML = 1.08

# Microsomal protein per gram liver (mg/g)
# Source: Barter et al. (2007)
MPPGL_REFERENCE = 40.0

# Hepatocytes per gram liver (million/g)
# Source: Rane et al. (1977)
HPGL_REFERENCE = 120.0

# GI tract parameters
GI_TRANSIT_TIMES = {
    "stomach": 0.25,  # hours (fasted)
    "duodenum": 0.25,
    "jejunum": 0.93,  # ~56 min
    "ileum": 1.4,  # ~84 min
    "colon": 13.5,
}

GI_SEGMENT_PH = {
    "stomach": 1.5,
    "duodenum": 6.0,
    "jejunum": 6.5,
    "ileum": 7.0,
    "colon": 7.5,
}

GI_SEGMENT_SURFACE_AREA_CM2 = {
    "duodenum": 1_880.0,
    "jejunum": 18_400.0,
    "ileum": 8_070.0,
    "colon": 2_500.0,
}

# Blood:plasma ratio constants
HEMATOCRIT_REFERENCE = 0.45

# Albumin and AAG concentrations
ALBUMIN_PLASMA_UM = 600.0  # ~4.0 g/dL
AAG_PLASMA_UM = 20.0  # ~0.08 g/dL
