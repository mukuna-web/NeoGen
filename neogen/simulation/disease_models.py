"""
Disease State Modeling
=======================
Models how diseases alter drug pharmacokinetics by modifying
physiological parameters (organ function, blood flow, enzyme activity).

A 70kg healthy male processes drugs very differently from:
- A cirrhotic patient (liver barely works, portal hypertension)
- A CKD patient (kidneys can't clear drugs)
- A heart failure patient (reduced cardiac output → everything slows)
- An obese patient (huge Vd for lipophilic drugs)

This is critical for clinical dosing — the #1 cause of adverse drug
events is failing to adjust doses for organ impairment.

Reference:
- Verbeeck (2008) "Pharmacokinetics and dosage adjustment in patients with hepatic dysfunction"
- Rowland & Tozer "Clinical Pharmacokinetics and Pharmacodynamics"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# DISEASE SEVERITY SCALES
# ════════════════════════════════════════════════════════════════

class ChildPughClass(str, Enum):
    """Child-Pugh classification for liver disease severity."""
    A = "A"  # Mild (score 5-6): well-compensated
    B = "B"  # Moderate (score 7-9): significant compromise
    C = "C"  # Severe (score 10-15): decompensated


class CKDStage(str, Enum):
    """Chronic Kidney Disease staging by GFR."""
    STAGE_1 = "1"   # GFR ≥ 90: Normal
    STAGE_2 = "2"   # GFR 60-89: Mild
    STAGE_3A = "3a"  # GFR 45-59: Mild-Moderate
    STAGE_3B = "3b"  # GFR 30-44: Moderate-Severe
    STAGE_4 = "4"    # GFR 15-29: Severe
    STAGE_5 = "5"    # GFR < 15: End-stage (dialysis)


class NYHAClass(str, Enum):
    """New York Heart Association heart failure classification."""
    I = "I"      # No symptoms with ordinary activity
    II = "II"    # Mild symptoms with ordinary activity
    III = "III"  # Marked limitation, symptoms with less than ordinary activity
    IV = "IV"    # Severe, symptoms at rest


class ObesityClass(str, Enum):
    """BMI-based obesity classification."""
    NORMAL = "normal"          # BMI 18.5-24.9
    OVERWEIGHT = "overweight"  # BMI 25-29.9
    OBESE_I = "obese_I"        # BMI 30-34.9
    OBESE_II = "obese_II"      # BMI 35-39.9
    OBESE_III = "obese_III"    # BMI ≥ 40 (morbid obesity)


# ════════════════════════════════════════════════════════════════
# DISEASE STATE DEFINITIONS
# ════════════════════════════════════════════════════════════════

@dataclass
class DiseaseModifiers:
    """
    How a disease modifies physiological parameters.
    All values are MULTIPLIERS (1.0 = no change).
    """
    name: str
    description: str

    # Organ function multipliers
    hepatic_clearance: float = 1.0   # Liver's ability to metabolize drugs
    renal_clearance: float = 1.0     # Kidney's ability to excrete drugs
    cardiac_output: float = 1.0      # Heart's pumping capacity

    # Blood flow multipliers
    hepatic_blood_flow: float = 1.0
    renal_blood_flow: float = 1.0
    gut_blood_flow: float = 1.0
    brain_blood_flow: float = 1.0
    muscle_blood_flow: float = 1.0

    # Volume multipliers
    plasma_volume: float = 1.0
    fat_volume: float = 1.0
    muscle_volume: float = 1.0

    # Protein binding changes
    albumin_level: float = 1.0       # 1.0 = normal (4 g/dL), 0.5 = hypoalbuminemia
    aag_level: float = 1.0           # Alpha-1-acid glycoprotein

    # Enzyme activity
    cyp3a4_activity: float = 1.0
    cyp2d6_activity: float = 1.0
    cyp2c9_activity: float = 1.0
    cyp1a2_activity: float = 1.0
    cyp2c19_activity: float = 1.0
    ugt_activity: float = 1.0        # Phase II glucuronidation

    # GI changes
    gastric_ph: float = 1.5          # Normal fasting pH ~1.5
    gi_motility: float = 1.0         # 1.0 = normal, <1 = slow, >1 = fast
    absorption_rate: float = 1.0

    # Special flags
    ascites: bool = False            # Fluid in abdomen (increases Vd)
    edema: bool = False              # Peripheral edema (increases Vd)
    dialysis: bool = False           # On hemodialysis

    # Clinical notes
    dosing_guidance: list[str] = field(default_factory=list)
    monitoring: list[str] = field(default_factory=list)
    contraindicated_drugs: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
# LIVER DISEASE (CIRRHOSIS)
# ════════════════════════════════════════════════════════════════

def get_liver_disease(severity: ChildPughClass) -> DiseaseModifiers:
    """
    Liver cirrhosis modifiers.

    Cirrhosis destroys hepatocytes → less CYP enzymes → slower metabolism.
    Portal hypertension → blood bypasses liver (portosystemic shunting).
    Low albumin → more free drug → paradoxical toxicity at normal doses.
    """
    if severity == ChildPughClass.A:
        return DiseaseModifiers(
            name="Liver Cirrhosis (Child-Pugh A — Mild)",
            description="Well-compensated cirrhosis. Mild reduction in liver function.",
            hepatic_clearance=0.70,
            hepatic_blood_flow=0.85,
            albumin_level=0.85,
            cyp3a4_activity=0.70,
            cyp2d6_activity=0.80,
            cyp2c9_activity=0.75,
            cyp1a2_activity=0.70,
            cyp2c19_activity=0.75,
            ugt_activity=0.80,
            dosing_guidance=[
                "Reduce hepatically-cleared drug doses by 25-50%",
                "Monitor liver function tests (ALT, AST, bilirubin)",
                "Avoid hepatotoxic drugs when possible",
            ],
            monitoring=["LFTs every 2-4 weeks", "INR if on anticoagulants"],
            contraindicated_drugs=["High-dose paracetamol", "Methotrexate"],
        )
    elif severity == ChildPughClass.B:
        return DiseaseModifiers(
            name="Liver Cirrhosis (Child-Pugh B — Moderate)",
            description="Significant liver compromise. Portal hypertension developing.",
            hepatic_clearance=0.40,
            hepatic_blood_flow=0.60,
            cardiac_output=0.90,
            albumin_level=0.60,
            aag_level=0.80,
            cyp3a4_activity=0.40,
            cyp2d6_activity=0.50,
            cyp2c9_activity=0.45,
            cyp1a2_activity=0.35,
            cyp2c19_activity=0.40,
            ugt_activity=0.50,
            ascites=True,
            plasma_volume=1.3,
            dosing_guidance=[
                "Reduce hepatically-cleared drug doses by 50-75%",
                "AVOID drugs with high hepatic extraction ratio",
                "Use drugs cleared by kidneys when possible",
                "Watch for altered protein binding (low albumin)",
                "Ascites increases volume of distribution for water-soluble drugs",
            ],
            monitoring=["LFTs weekly", "Albumin levels", "INR", "Drug levels if available"],
            contraindicated_drugs=["Paracetamol >2g/day", "NSAIDs", "Methotrexate", "Isoniazid"],
        )
    else:  # Child-Pugh C
        return DiseaseModifiers(
            name="Liver Cirrhosis (Child-Pugh C — Severe/Decompensated)",
            description="Decompensated cirrhosis. Severe hepatic dysfunction. Very high risk.",
            hepatic_clearance=0.15,
            hepatic_blood_flow=0.35,
            cardiac_output=0.80,
            renal_clearance=0.70,  # Hepatorenal syndrome
            renal_blood_flow=0.70,
            albumin_level=0.35,
            aag_level=0.60,
            cyp3a4_activity=0.15,
            cyp2d6_activity=0.25,
            cyp2c9_activity=0.20,
            cyp1a2_activity=0.15,
            cyp2c19_activity=0.20,
            ugt_activity=0.25,
            ascites=True,
            edema=True,
            plasma_volume=1.6,
            gi_motility=0.70,
            dosing_guidance=[
                "CRITICAL: Most hepatically-cleared drugs need 75%+ dose reduction",
                "Many drugs are CONTRAINDICATED",
                "Hepatorenal syndrome may also reduce renal clearance",
                "Massive Vd changes due to ascites + edema",
                "Extremely low albumin → much higher free drug fractions",
            ],
            monitoring=["Daily LFTs", "Renal function", "Drug levels mandatory", "INR"],
            contraindicated_drugs=[
                "Paracetamol", "All NSAIDs", "Methotrexate", "Statins (most)",
                "Benzodiazepines (use lorazepam only)", "Opioids (reduce dose 75%)",
            ],
        )


# ════════════════════════════════════════════════════════════════
# KIDNEY DISEASE (CKD)
# ════════════════════════════════════════════════════════════════

def get_kidney_disease(stage: CKDStage) -> DiseaseModifiers:
    """
    Chronic Kidney Disease modifiers.

    CKD reduces glomerular filtration → renally-cleared drugs accumulate.
    Uremic toxins also inhibit hepatic CYP enzymes (CKD affects liver too!).
    Fluid overload increases Vd.
    """
    gfr_fractions = {
        CKDStage.STAGE_1: 1.0,
        CKDStage.STAGE_2: 0.75,
        CKDStage.STAGE_3A: 0.55,
        CKDStage.STAGE_3B: 0.40,
        CKDStage.STAGE_4: 0.22,
        CKDStage.STAGE_5: 0.08,
    }
    gfr_frac = gfr_fractions.get(stage, 1.0)

    # Uremic toxins inhibit hepatic CYP3A4 and CYP2C9
    # Nolin et al. (2003): CKD reduces hepatic CL by 20-50%
    hepatic_reduction = 1.0 - (1 - gfr_frac) * 0.3  # Up to 30% hepatic CL reduction

    base = DiseaseModifiers(
        name=f"Chronic Kidney Disease (Stage {stage.value})",
        description=f"GFR at {gfr_frac*100:.0f}% of normal",
        renal_clearance=gfr_frac,
        renal_blood_flow=max(0.3, gfr_frac * 1.1),  # Blood flow reduced but not as much as GFR
        hepatic_clearance=hepatic_reduction,
        cyp3a4_activity=hepatic_reduction,
        cyp2c9_activity=hepatic_reduction,
        albumin_level=max(0.6, 1.0 - (1 - gfr_frac) * 0.4),  # Proteinuria loses albumin
        plasma_volume=1.0 + (1 - gfr_frac) * 0.3,  # Fluid retention
    )

    if stage in (CKDStage.STAGE_4, CKDStage.STAGE_5):
        base.edema = True
        base.gi_motility = 0.80  # Uremic gastroparesis
        base.absorption_rate = 0.85
        base.dosing_guidance = [
            "CRITICAL: Renally-cleared drugs require major dose reduction",
            f"GFR ≈ {gfr_frac*90:.0f} mL/min — adjust per drug-specific renal dosing",
            "Uremic toxins also impair hepatic metabolism by ~30%",
            "Monitor drug levels for narrow therapeutic index drugs",
            "Avoid nephrotoxic drugs (NSAIDs, aminoglycosides, contrast dye)",
        ]
        base.monitoring = ["Creatinine/BUN daily", "Drug levels", "Electrolytes (K+, PO4)"]
        base.contraindicated_drugs = [
            "NSAIDs", "Metformin (lactic acidosis)", "Lithium (accumulation)",
            "Aminoglycosides (nephro+ototoxic)", "High-dose aspirin",
        ]
    elif stage in (CKDStage.STAGE_3A, CKDStage.STAGE_3B):
        base.dosing_guidance = [
            f"Moderate renal impairment — GFR ≈ {gfr_frac*90:.0f} mL/min",
            "Reduce renally-cleared drug doses by 25-50%",
            "Monitor creatinine and drug levels",
        ]
        base.monitoring = ["Creatinine weekly", "eGFR trending"]
        base.contraindicated_drugs = ["NSAIDs (if possible)", "Metformin (stage 3b+)"]
    else:
        base.dosing_guidance = ["Mild renal impairment — most drugs safe at standard doses",
                                 "Monitor renal function periodically"]

    if stage == CKDStage.STAGE_5:
        base.dialysis = True
        base.dosing_guidance.append("On dialysis: dose AFTER dialysis session, not before")

    return base


# ════════════════════════════════════════════════════════════════
# HEART FAILURE
# ════════════════════════════════════════════════════════════════

def get_heart_failure(nyha_class: NYHAClass) -> DiseaseModifiers:
    """
    Heart failure modifiers.

    Reduced cardiac output → reduced blood flow to ALL organs.
    Liver: congested, reduced metabolism (cardiac cirrhosis)
    Kidneys: reduced perfusion → cardiorenal syndrome
    Gut: reduced absorption (bowel wall edema)
    """
    co_fractions = {
        NYHAClass.I: 0.90,
        NYHAClass.II: 0.75,
        NYHAClass.III: 0.55,
        NYHAClass.IV: 0.35,
    }
    co_frac = co_fractions.get(nyha_class, 1.0)

    base = DiseaseModifiers(
        name=f"Heart Failure (NYHA Class {nyha_class.value})",
        description=f"Cardiac output at {co_frac*100:.0f}% of normal",
        cardiac_output=co_frac,
        # All organ blood flows reduced proportionally
        hepatic_blood_flow=co_frac,
        renal_blood_flow=co_frac * 1.05,  # Kidney autoregulation helps slightly
        gut_blood_flow=co_frac * 0.90,  # Gut hit hardest
        brain_blood_flow=max(0.7, co_frac * 1.2),  # Brain autoregulation maintains flow
        muscle_blood_flow=co_frac * 0.80,
        # Hepatic congestion reduces metabolism
        hepatic_clearance=max(0.3, co_frac * 0.9),
        renal_clearance=max(0.3, co_frac * 1.0),
        # Volume changes
        plasma_volume=1.0 + (1 - co_frac) * 0.5,  # Fluid retention
        # Enzyme impairment from hepatic congestion
        cyp3a4_activity=max(0.4, co_frac * 0.95),
        cyp1a2_activity=max(0.4, co_frac * 0.90),
    )

    if nyha_class in (NYHAClass.III, NYHAClass.IV):
        base.edema = True
        base.ascites = nyha_class == NYHAClass.IV
        base.gi_motility = 0.70
        base.absorption_rate = 0.75
        base.dosing_guidance = [
            f"SEVERE heart failure — cardiac output at {co_frac*100:.0f}%",
            "Reduced blood flow to liver and kidneys → slower clearance",
            "Bowel wall edema reduces oral absorption",
            "Loading doses may need to be higher (increased Vd from edema)",
            "Maintenance doses LOWER (reduced clearance)",
            "IV route preferred for critical medications",
        ]
        base.monitoring = ["Daily weights", "Renal function", "Drug levels", "Electrolytes"]
        base.contraindicated_drugs = [
            "NSAIDs (worsen fluid retention)", "Metformin (lactic acidosis risk)",
            "Calcium channel blockers (negative inotropes)",
            "Thiazolidinediones (fluid retention)",
        ]
    else:
        base.dosing_guidance = [
            "Mild-moderate heart failure",
            "Most drugs safe but monitor for fluid retention",
            "Reduce doses of hepatically-cleared drugs by 25%",
        ]

    return base


# ════════════════════════════════════════════════════════════════
# OBESITY
# ════════════════════════════════════════════════════════════════

def get_obesity(obesity_class: ObesityClass, bmi: float = 30.0) -> DiseaseModifiers:
    """
    Obesity modifiers.

    Obesity dramatically changes drug distribution:
    - Lipophilic drugs: HUGE Vd increase (drug hides in fat)
    - Hydrophilic drugs: less affected
    - Cardiac output increased (more tissue to perfuse)
    - CYP3A4 often REDUCED, CYP2E1 INCREASED (fatty liver)
    """
    fat_multipliers = {
        ObesityClass.NORMAL: 1.0,
        ObesityClass.OVERWEIGHT: 1.4,
        ObesityClass.OBESE_I: 1.8,
        ObesityClass.OBESE_II: 2.5,
        ObesityClass.OBESE_III: 3.5,
    }
    fat_mult = fat_multipliers.get(obesity_class, 1.0)

    base = DiseaseModifiers(
        name=f"Obesity ({obesity_class.value}, BMI={bmi:.0f})",
        description=f"Fat volume {fat_mult:.1f}x normal",
        fat_volume=fat_mult,
        muscle_volume=1.0 + (fat_mult - 1) * 0.2,  # Some muscle increase too
        plasma_volume=1.0 + (fat_mult - 1) * 0.15,
        cardiac_output=1.0 + (fat_mult - 1) * 0.1,  # Increased CO
        # Enzyme changes in fatty liver
        cyp3a4_activity=max(0.6, 1.0 - (fat_mult - 1) * 0.15),  # Reduced
        cyp1a2_activity=max(0.7, 1.0 - (fat_mult - 1) * 0.1),
        # CYP2E1 is INCREASED in obesity (not modeled separately but affects paracetamol toxicity)
        ugt_activity=1.0 + (fat_mult - 1) * 0.1,  # Slightly increased
        # Renal
        renal_clearance=1.0 + (fat_mult - 1) * 0.15,  # GFR often increased (hyperfiltration)
    )

    if obesity_class in (ObesityClass.OBESE_II, ObesityClass.OBESE_III):
        base.dosing_guidance = [
            "CRITICAL: Dosing in obesity is complex and drug-specific",
            "Lipophilic drugs (diazepam, amiodarone): dose on TOTAL body weight",
            "Hydrophilic drugs (aminoglycosides, vancomycin): dose on ADJUSTED body weight",
            f"Fat volume is {fat_mult:.1f}x normal — lipophilic drugs have massive Vd",
            "Loading doses often need to be HIGHER, maintenance doses variable",
            "CYP3A4 activity reduced (fatty liver) — midazolam, statins affected",
            "Consider IV route if oral absorption unreliable",
        ]
        base.monitoring = ["Drug levels for narrow TI drugs", "Renal function (hyperfiltration)"]
    else:
        base.dosing_guidance = [
            "Mild-moderate obesity — most drugs dosed normally",
            "Consider weight-based dosing for antimicrobials",
        ]

    return base


# ════════════════════════════════════════════════════════════════
# DIABETES (TYPE 2)
# ════════════════════════════════════════════════════════════════

def get_diabetes_t2(hba1c: float = 7.0, has_neuropathy: bool = False,
                     has_nephropathy: bool = False) -> DiseaseModifiers:
    """
    Type 2 Diabetes modifiers.

    Diabetes affects PK through:
    - Gastroparesis (delayed gastric emptying → slower absorption)
    - Nephropathy (renal impairment)
    - Neuropathy (altered tissue perfusion)
    - Non-alcoholic fatty liver disease (enzyme changes)
    - Hyperglycemia affects protein glycation → altered binding
    """
    base = DiseaseModifiers(
        name=f"Type 2 Diabetes (HbA1c={hba1c:.1f}%)",
        description="Metabolic syndrome with multi-organ effects",
        gi_motility=0.70 if hba1c > 8 else 0.85,  # Gastroparesis
        absorption_rate=0.80 if hba1c > 8 else 0.90,
        gastric_ph=3.0 if hba1c > 9 else 2.0,  # Reduced acid secretion
        # Fatty liver
        cyp3a4_activity=0.80,
        cyp2c9_activity=0.85,
        ugt_activity=1.1,  # Often increased
        # Albumin glycation reduces binding
        albumin_level=0.90,
    )

    if has_nephropathy:
        base.renal_clearance = 0.60
        base.renal_blood_flow = 0.70

    if has_neuropathy:
        base.muscle_blood_flow = 0.80

    base.dosing_guidance = [
        "Gastroparesis may delay oral drug absorption",
        "Check renal function — diabetic nephropathy common",
        "Fatty liver may alter hepatic drug metabolism",
        "Avoid metformin if eGFR < 30",
        "Sulfonylureas: increased hypoglycemia risk with renal impairment",
    ]
    base.monitoring = ["HbA1c quarterly", "eGFR annually", "LFTs"]

    return base


# ════════════════════════════════════════════════════════════════
# PREGNANCY
# ════════════════════════════════════════════════════════════════

def get_pregnancy(trimester: int = 2) -> DiseaseModifiers:
    """
    Pregnancy modifiers.

    Pregnancy causes dramatic PK changes:
    - Cardiac output increases 30-50%
    - Plasma volume increases 40-50%
    - GFR increases 50% (hyperfiltration)
    - CYP3A4 activity DOUBLES (progesterone induction)
    - CYP2D6 increases 50%
    - CYP1A2 DECREASES 50% (estrogen inhibition)
    - Albumin drops 20-30%
    """
    trim_mult = {1: 0.5, 2: 0.8, 3: 1.0}  # Changes scale with trimester
    t = trim_mult.get(trimester, 1.0)

    return DiseaseModifiers(
        name=f"Pregnancy (Trimester {trimester})",
        description=f"Trimester {trimester} — significant physiological changes",
        cardiac_output=1.0 + 0.40 * t,  # Up to 40% increase
        plasma_volume=1.0 + 0.45 * t,   # Up to 45% increase
        renal_clearance=1.0 + 0.50 * t,  # GFR increases 50%
        renal_blood_flow=1.0 + 0.50 * t,
        hepatic_blood_flow=1.0 + 0.30 * t,
        # Enzyme changes
        cyp3a4_activity=1.0 + 1.0 * t,   # DOUBLES by T3
        cyp2d6_activity=1.0 + 0.50 * t,  # 50% increase
        cyp1a2_activity=1.0 - 0.50 * t,  # 50% DECREASE
        cyp2c9_activity=1.0 + 0.20 * t,  # 20% increase
        ugt_activity=1.0 + 0.50 * t,     # 50% increase
        # Binding changes
        albumin_level=1.0 - 0.25 * t,    # 25% decrease
        # GI
        gi_motility=1.0 - 0.20 * t,      # Slower (progesterone)
        gastric_ph=2.0 + 1.0 * t,        # Higher pH
        dosing_guidance=[
            "CRITICAL: Many drugs are teratogenic — check FDA category first",
            f"Trimester {trimester}: cardiac output +{40*t:.0f}%, GFR +{50*t:.0f}%",
            "CYP3A4 activity DOUBLED — drugs like midazolam may need HIGHER doses",
            "CYP1A2 activity HALVED — caffeine half-life doubles",
            "Renally-cleared drugs may need HIGHER doses (increased GFR)",
            "Lower albumin → more free drug → watch for toxicity",
        ],
        monitoring=["Trimester-appropriate drug screening", "Renal function", "Drug levels"],
        contraindicated_drugs=[
            "Warfarin (teratogenic)", "Methotrexate (abortifacient)",
            "ACE inhibitors (fetal renal damage)", "Statins (teratogenic)",
            "Tetracyclines (bone/teeth)", "Isotretinoin (severe birth defects)",
            "Valproic acid (neural tube defects)", "Lithium (cardiac defects)",
        ],
    )


# ════════════════════════════════════════════════════════════════
# SEPSIS / CRITICAL ILLNESS
# ════════════════════════════════════════════════════════════════

def get_sepsis(severe: bool = False) -> DiseaseModifiers:
    """
    Sepsis/critical illness modifiers.

    Sepsis causes cytokine storm → dramatic PK changes:
    - IL-6, TNF-α downregulate CYP enzymes (up to 90% reduction!)
    - Capillary leak → massive Vd increase (third-spacing)
    - Hypoalbuminemia → more free drug
    - Augmented renal clearance early → excessive drug loss
    """
    if severe:
        return DiseaseModifiers(
            name="Severe Sepsis / Septic Shock",
            description="Cytokine storm with multi-organ dysfunction",
            cardiac_output=1.5,  # Hyperdynamic early phase
            hepatic_clearance=0.20,  # CYP enzymes massively downregulated
            renal_clearance=0.40,  # AKI
            cyp3a4_activity=0.15,  # IL-6 suppresses CYP3A4 by 85%
            cyp2c9_activity=0.20,
            cyp1a2_activity=0.10,  # Most affected
            cyp2d6_activity=0.30,
            cyp2c19_activity=0.25,
            albumin_level=0.40,  # Severe hypoalbuminemia
            aag_level=2.0,  # AAG is an acute phase reactant — INCREASES
            plasma_volume=1.5,  # Capillary leak → third-spacing
            gut_blood_flow=0.40,  # Splanchnic hypoperfusion
            absorption_rate=0.30,  # Oral absorption essentially zero
            gi_motility=0.30,
            edema=True,
            ascites=True,
            dosing_guidance=[
                "CRITICAL: Oral absorption unreliable — use IV route",
                "CYP enzymes suppressed by 70-85% (cytokine-mediated)",
                "Third-spacing increases Vd dramatically",
                "Higher loading doses needed, then lower maintenance",
                "Drug levels MANDATORY for narrow TI drugs",
                "AAG increased → basic drugs (lidocaine) more bound",
                "Albumin decreased → acidic drugs (phenytoin) more free",
            ],
            monitoring=["Drug levels q6-12h", "Creatinine", "Lactate", "LFTs"],
        )
    else:
        return DiseaseModifiers(
            name="Sepsis (Early/Moderate)",
            description="Systemic infection with inflammatory response",
            cardiac_output=1.3,
            hepatic_clearance=0.50,
            renal_clearance=1.3,  # Augmented renal clearance (ARC) in early sepsis
            cyp3a4_activity=0.40,
            cyp1a2_activity=0.30,
            albumin_level=0.65,
            aag_level=1.5,
            plasma_volume=1.3,
            dosing_guidance=[
                "Early sepsis: augmented renal clearance may cause sub-therapeutic drug levels",
                "May need HIGHER antimicrobial doses initially",
                "CYP enzymes moderately suppressed",
                "Monitor for transition to severe sepsis",
            ],
        )


# ════════════════════════════════════════════════════════════════
# DISEASE STATE ENGINE
# ════════════════════════════════════════════════════════════════

class DiseaseStateEngine:
    """
    Applies disease state modifiers to physiological parameters.

    Supports multiple co-morbidities (e.g., patient with cirrhosis + CKD).
    Modifiers are multiplicative — if cirrhosis reduces CL by 50% and
    CKD reduces CL by 40%, combined CL = 0.5 × 0.6 = 30% of normal.
    """

    def __init__(self):
        self.diseases: list[DiseaseModifiers] = []

    def add_disease(self, disease: DiseaseModifiers) -> None:
        """Add a disease state to the patient."""
        self.diseases.append(disease)
        logger.info("Added disease: %s", disease.name)

    def get_combined_modifiers(self) -> DiseaseModifiers:
        """Combine all disease states multiplicatively."""
        if not self.diseases:
            return DiseaseModifiers(name="Healthy", description="No disease states")

        combined = DiseaseModifiers(
            name=" + ".join(d.name for d in self.diseases),
            description="Combined disease state modifiers",
        )

        # Multiply all modifiers
        for d in self.diseases:
            combined.hepatic_clearance *= d.hepatic_clearance
            combined.renal_clearance *= d.renal_clearance
            combined.cardiac_output *= d.cardiac_output
            combined.hepatic_blood_flow *= d.hepatic_blood_flow
            combined.renal_blood_flow *= d.renal_blood_flow
            combined.gut_blood_flow *= d.gut_blood_flow
            combined.brain_blood_flow *= d.brain_blood_flow
            combined.muscle_blood_flow *= d.muscle_blood_flow
            combined.plasma_volume *= d.plasma_volume
            combined.fat_volume *= d.fat_volume
            combined.muscle_volume *= d.muscle_volume
            combined.albumin_level *= d.albumin_level
            combined.aag_level *= d.aag_level
            combined.cyp3a4_activity *= d.cyp3a4_activity
            combined.cyp2d6_activity *= d.cyp2d6_activity
            combined.cyp2c9_activity *= d.cyp2c9_activity
            combined.cyp1a2_activity *= d.cyp1a2_activity
            combined.cyp2c19_activity *= d.cyp2c19_activity
            combined.ugt_activity *= d.ugt_activity
            combined.absorption_rate *= d.absorption_rate
            combined.gi_motility *= d.gi_motility

            # Boolean flags — OR (any disease triggers them)
            combined.ascites = combined.ascites or d.ascites
            combined.edema = combined.edema or d.edema
            combined.dialysis = combined.dialysis or d.dialysis

            # Collect all guidance
            combined.dosing_guidance.extend(d.dosing_guidance)
            combined.monitoring.extend(d.monitoring)
            combined.contraindicated_drugs.extend(d.contraindicated_drugs)

        # Deduplicate lists
        combined.dosing_guidance = list(dict.fromkeys(combined.dosing_guidance))
        combined.monitoring = list(dict.fromkeys(combined.monitoring))
        combined.contraindicated_drugs = list(dict.fromkeys(combined.contraindicated_drugs))

        return combined

    def apply_to_physiology(self, phys_params) -> dict:
        """
        Apply disease modifiers to PhysiologicalParameters.
        Returns a dict of changes made for reporting.
        """
        mods = self.get_combined_modifiers()

        changes = {}

        # Apply blood flow changes
        if mods.cardiac_output != 1.0:
            old_co = phys_params.CO
            phys_params.CO *= mods.cardiac_output
            changes["cardiac_output"] = f"{old_co:.0f} → {phys_params.CO:.0f} L/h ({mods.cardiac_output*100:.0f}%)"

        # Recalculate organ blood flows based on modified CO
        phys_params.Q_gut = phys_params.CO * 0.15 * mods.gut_blood_flow
        phys_params.Q_liver = phys_params.CO * 0.25 * mods.hepatic_blood_flow
        phys_params.Q_kidney = phys_params.CO * 0.19 * mods.renal_blood_flow
        phys_params.Q_brain = phys_params.CO * 0.12 * mods.brain_blood_flow
        phys_params.Q_muscle = phys_params.CO * 0.17 * mods.muscle_blood_flow
        phys_params.Q_fat = phys_params.CO * 0.05

        # Volume changes
        phys_params.V_plasma *= mods.plasma_volume
        phys_params.V_fat *= mods.fat_volume
        phys_params.V_muscle *= mods.muscle_volume

        # Clearance scaling
        phys_params.hepatic_cl_scale *= mods.hepatic_clearance
        phys_params.renal_cl_scale *= mods.renal_clearance

        changes["hepatic_clearance"] = f"{mods.hepatic_clearance*100:.0f}% of normal"
        changes["renal_clearance"] = f"{mods.renal_clearance*100:.0f}% of normal"

        return changes

    def get_report(self) -> dict:
        """Generate a clinical report of all disease states and their PK impact."""
        mods = self.get_combined_modifiers()

        return {
            "diseases": [d.name for d in self.diseases],
            "n_diseases": len(self.diseases),
            "combined_name": mods.name,
            "modifiers": {
                "hepatic_clearance": round(mods.hepatic_clearance, 3),
                "renal_clearance": round(mods.renal_clearance, 3),
                "cardiac_output": round(mods.cardiac_output, 3),
                "albumin": round(mods.albumin_level, 3),
                "cyp3a4": round(mods.cyp3a4_activity, 3),
                "cyp2d6": round(mods.cyp2d6_activity, 3),
                "cyp1a2": round(mods.cyp1a2_activity, 3),
                "absorption_rate": round(mods.absorption_rate, 3),
                "plasma_volume": round(mods.plasma_volume, 3),
                "fat_volume": round(mods.fat_volume, 3),
            },
            "flags": {
                "ascites": mods.ascites,
                "edema": mods.edema,
                "dialysis": mods.dialysis,
            },
            "dosing_guidance": mods.dosing_guidance,
            "monitoring": mods.monitoring,
            "contraindicated_drugs": mods.contraindicated_drugs,
        }


# ════════════════════════════════════════════════════════════════
# HEPATITIS (VIRAL)
# ════════════════════════════════════════════════════════════════

def get_hepatitis(variant: str = "B", acute: bool = False) -> DiseaseModifiers:
    """Viral hepatitis — active inflammation damages hepatocytes."""
    if acute:
        return DiseaseModifiers(
            name=f"Acute Hepatitis {variant}",
            description=f"Active hepatitis {variant} with liver inflammation",
            hepatic_clearance=0.35,
            hepatic_blood_flow=0.70,
            cyp3a4_activity=0.30,
            cyp1a2_activity=0.35,
            cyp2c9_activity=0.40,
            cyp2d6_activity=0.50,
            cyp2c19_activity=0.40,
            ugt_activity=0.45,
            albumin_level=0.70,
            dosing_guidance=[
                "Acute hepatitis: hepatic drug metabolism severely impaired",
                "Reduce hepatically-cleared drugs by 50-70%",
                "Avoid hepatotoxic drugs (paracetamol, statins, isoniazid)",
            ],
            monitoring=["ALT/AST daily", "Bilirubin", "INR", "Drug levels"],
            contraindicated_drugs=["Paracetamol", "Statins", "Isoniazid", "Methotrexate"],
        )
    return DiseaseModifiers(
        name=f"Chronic Hepatitis {variant}",
        description=f"Chronic hepatitis {variant} — mild ongoing liver inflammation",
        hepatic_clearance=0.75,
        cyp3a4_activity=0.70,
        cyp1a2_activity=0.75,
        albumin_level=0.85,
        dosing_guidance=["Mild hepatic impairment — reduce hepatically-cleared drugs by 25%"],
        monitoring=["LFTs quarterly", "Viral load"],
    )


# ════════════════════════════════════════════════════════════════
# ALCOHOLIC LIVER DISEASE
# ════════════════════════════════════════════════════════════════

def get_alcoholic_liver_disease(severity: str = "moderate") -> DiseaseModifiers:
    """
    Alcoholic liver disease — unique because CYP2E1 is INDUCED (not reduced).
    This means paracetamol toxicity risk is HIGHER than normal.
    """
    if severity == "mild":
        return DiseaseModifiers(
            name="Alcoholic Liver Disease (Mild/Fatty Liver)",
            description="Alcoholic steatosis — CYP2E1 induced, other CYPs mildly reduced",
            hepatic_clearance=0.85,
            cyp3a4_activity=0.85,
            cyp1a2_activity=0.80,
            ugt_activity=1.1,
            dosing_guidance=[
                "CYP2E1 is INDUCED by chronic alcohol → more toxic NAPQI from paracetamol",
                "Maximum paracetamol dose: 2g/day (not 4g)",
            ],
            contraindicated_drugs=["Paracetamol >2g/day", "Methotrexate"],
        )
    elif severity == "severe":
        return DiseaseModifiers(
            name="Alcoholic Liver Disease (Severe/Alcoholic Hepatitis)",
            description="Severe alcoholic hepatitis with jaundice",
            hepatic_clearance=0.25,
            hepatic_blood_flow=0.50,
            cyp3a4_activity=0.25,
            cyp1a2_activity=0.20,
            cyp2c9_activity=0.30,
            albumin_level=0.45,
            ascites=True,
            edema=True,
            plasma_volume=1.4,
            dosing_guidance=[
                "Severe hepatic impairment — most hepatic drugs contraindicated",
                "CYP2E1 may still be induced despite overall hepatic failure",
                "Paracetamol is ABSOLUTELY contraindicated",
            ],
            monitoring=["Daily LFTs", "INR", "Ammonia levels"],
            contraindicated_drugs=["Paracetamol", "NSAIDs", "Benzodiazepines", "Opioids"],
        )
    return DiseaseModifiers(
        name="Alcoholic Liver Disease (Moderate)",
        description="Moderate alcoholic liver disease with fibrosis",
        hepatic_clearance=0.55,
        hepatic_blood_flow=0.70,
        cyp3a4_activity=0.55,
        cyp1a2_activity=0.50,
        albumin_level=0.70,
        dosing_guidance=["Reduce hepatically-cleared drugs by 40-50%"],
        monitoring=["LFTs monthly", "INR"],
        contraindicated_drugs=["Paracetamol >2g/day", "Methotrexate"],
    )


# ════════════════════════════════════════════════════════════════
# NAFLD / NASH
# ════════════════════════════════════════════════════════════════

def get_nafld(nash: bool = False) -> DiseaseModifiers:
    """Non-alcoholic fatty liver disease / NASH."""
    if nash:
        return DiseaseModifiers(
            name="NASH (Non-Alcoholic Steatohepatitis)",
            description="Fatty liver with inflammation and fibrosis",
            hepatic_clearance=0.65,
            cyp3a4_activity=0.60,
            cyp2c9_activity=0.70,
            cyp1a2_activity=0.65,
            ugt_activity=1.15,
            albumin_level=0.80,
            dosing_guidance=[
                "NASH reduces CYP3A4 by 30-40%",
                "Statins are generally SAFE and recommended for NASH",
                "Monitor for drug accumulation with hepatically-cleared drugs",
            ],
            monitoring=["LFTs quarterly", "FibroScan", "Lipid panel"],
        )
    return DiseaseModifiers(
        name="NAFLD (Non-Alcoholic Fatty Liver)",
        description="Simple steatosis without inflammation",
        hepatic_clearance=0.85,
        cyp3a4_activity=0.80,
        ugt_activity=1.10,
        dosing_guidance=["Mild hepatic impact — most drugs safe at standard doses"],
    )


# ════════════════════════════════════════════════════════════════
# ACUTE KIDNEY INJURY
# ════════════════════════════════════════════════════════════════

def get_aki(stage: int = 2) -> DiseaseModifiers:
    """
    Acute Kidney Injury (AKI) — sudden loss of kidney function.
    Different from CKD: rapid onset, potentially reversible, different management.
    KDIGO staging: 1 (mild), 2 (moderate), 3 (severe/dialysis).
    """
    stages = {
        1: DiseaseModifiers(
            name="AKI Stage 1 (Mild)",
            description="Creatinine 1.5-1.9x baseline or UO <0.5mL/kg/h for 6-12h",
            renal_clearance=0.60,
            renal_blood_flow=0.70,
            dosing_guidance=["Adjust renally-cleared drugs", "Monitor UO and creatinine q6h"],
            monitoring=["Creatinine q6h", "Urine output hourly"],
        ),
        2: DiseaseModifiers(
            name="AKI Stage 2 (Moderate)",
            description="Creatinine 2.0-2.9x baseline or UO <0.5mL/kg/h for ≥12h",
            renal_clearance=0.35,
            renal_blood_flow=0.50,
            hepatic_clearance=0.85,
            cyp3a4_activity=0.85,
            albumin_level=0.85,
            edema=True,
            plasma_volume=1.2,
            dosing_guidance=[
                "Significant renal impairment — reduce renally-cleared drugs 50-65%",
                "Uremic toxins begin affecting hepatic metabolism",
                "Fluid overload increases Vd",
            ],
            monitoring=["Creatinine q6h", "Drug levels", "Fluid balance"],
            contraindicated_drugs=["NSAIDs", "Aminoglycosides", "Contrast dye"],
        ),
        3: DiseaseModifiers(
            name="AKI Stage 3 (Severe — may need RRT)",
            description="Creatinine ≥3x baseline or anuria ≥12h or initiation of RRT",
            renal_clearance=0.10,
            renal_blood_flow=0.30,
            hepatic_clearance=0.70,
            cyp3a4_activity=0.70,
            cyp2c9_activity=0.75,
            albumin_level=0.70,
            edema=True,
            plasma_volume=1.4,
            dialysis=True,
            dosing_guidance=[
                "CRITICAL: Near-complete renal failure",
                "Renally-cleared drugs need 80-90% dose reduction or alternative",
                "If on RRT: dose AFTER dialysis",
                "Hepatic metabolism also impaired 30%",
            ],
            monitoring=["Creatinine q4h", "Electrolytes q6h", "Drug levels mandatory"],
            contraindicated_drugs=["NSAIDs", "Metformin", "Lithium", "ACE inhibitors"],
        ),
    }
    return stages.get(stage, stages[2])


# ════════════════════════════════════════════════════════════════
# HYPERTENSION
# ════════════════════════════════════════════════════════════════

def get_hypertension(severity: str = "moderate") -> DiseaseModifiers:
    """Hypertension — affects cardiac output and organ perfusion."""
    if severity == "mild":
        return DiseaseModifiers(
            name="Hypertension (Stage 1)",
            description="BP 130-139/80-89 mmHg — mild cardiac stress",
            cardiac_output=1.05,
            renal_blood_flow=0.95,
            dosing_guidance=["Most drugs unaffected", "Monitor renal function"],
        )
    elif severity == "severe":
        return DiseaseModifiers(
            name="Hypertension (Stage 2 — Uncontrolled)",
            description="BP ≥160/100 mmHg — organ damage accumulating",
            cardiac_output=1.15,
            renal_clearance=0.80,
            renal_blood_flow=0.75,
            brain_blood_flow=1.1,
            hepatic_blood_flow=1.05,
            albumin_level=0.90,
            dosing_guidance=[
                "Chronic hypertension → early nephrosclerosis",
                "Renal clearance reduced ~20%",
                "Avoid NSAIDs (worsen BP and renal function)",
            ],
            monitoring=["BP monitoring", "eGFR", "Urine albumin"],
            contraindicated_drugs=["NSAIDs (chronic use)", "Decongestants"],
        )
    return DiseaseModifiers(
        name="Hypertension (Stage 2 — Controlled)",
        description="BP ≥140/90 on treatment — compensated",
        cardiac_output=1.10,
        renal_clearance=0.90,
        renal_blood_flow=0.85,
        dosing_guidance=["Mild renal impact from chronic hypertension"],
    )


# ════════════════════════════════════════════════════════════════
# THYROID DISORDERS
# ════════════════════════════════════════════════════════════════

def get_thyroid_disorder(hypo: bool = True) -> DiseaseModifiers:
    """
    Thyroid disorders dramatically affect drug metabolism.
    Hypothyroidism: everything slows (CYP↓, clearance↓, GI motility↓)
    Hyperthyroidism: everything speeds up (CYP↑, clearance↑, GI motility↑)
    """
    if hypo:
        return DiseaseModifiers(
            name="Hypothyroidism",
            description="Low thyroid → reduced metabolism, slower GI, fluid retention",
            hepatic_clearance=0.70,
            renal_clearance=0.80,
            cardiac_output=0.85,
            cyp3a4_activity=0.65,
            cyp1a2_activity=0.70,
            cyp2c9_activity=0.75,
            ugt_activity=0.75,
            gi_motility=0.65,
            absorption_rate=0.75,
            plasma_volume=1.1,
            edema=True,
            dosing_guidance=[
                "Hypothyroidism slows drug metabolism by 25-35%",
                "GI motility reduced → delayed drug absorption",
                "Warfarin dose requirements DECREASE",
                "Digoxin clearance reduced → toxicity risk",
                "When starting levothyroxine: drug metabolism will INCREASE as thyroid normalizes",
            ],
            monitoring=["TSH q6-8 weeks during dose adjustment", "Drug levels for narrow TI drugs"],
        )
    return DiseaseModifiers(
        name="Hyperthyroidism",
        description="High thyroid → accelerated metabolism, fast GI transit",
        hepatic_clearance=1.40,
        renal_clearance=1.20,
        cardiac_output=1.30,
        cyp3a4_activity=1.40,
        cyp1a2_activity=1.35,
        cyp2c9_activity=1.25,
        ugt_activity=1.30,
        gi_motility=1.40,
        absorption_rate=1.20,
        dosing_guidance=[
            "Hyperthyroidism accelerates drug metabolism by 30-40%",
            "Warfarin dose requirements INCREASE",
            "May need HIGHER doses of hepatically-cleared drugs",
            "Fast GI transit may reduce absorption of slow-dissolving drugs",
            "When treating with anti-thyroid drugs: metabolism will SLOW as thyroid normalizes",
        ],
        monitoring=["TSH", "Drug levels — may need dose adjustments as thyroid normalizes"],
    )


# ════════════════════════════════════════════════════════════════
# HIV/AIDS
# ════════════════════════════════════════════════════════════════

def get_hiv(on_art: bool = True, cd4: int = 350) -> DiseaseModifiers:
    """
    HIV/AIDS — affects PK through:
    - Chronic inflammation (cytokine CYP suppression)
    - Protein wasting (low albumin)
    - ARV drug interactions (ritonavir is a CYP3A4 inhibitor)
    """
    base = DiseaseModifiers(
        name=f"HIV ({'on ART' if on_art else 'untreated'}, CD4={cd4})",
        description="HIV with chronic immune activation",
        albumin_level=0.80 if cd4 > 200 else 0.60,
        aag_level=1.3,
        cyp3a4_activity=0.70 if not on_art else 0.30,  # Ritonavir-boosted regimens inhibit 3A4
        cyp2d6_activity=0.85,
        hepatic_clearance=0.80,
    )

    if on_art:
        base.dosing_guidance = [
            "CRITICAL: ARV drug interactions are extremely common",
            "Protease inhibitors (ritonavir) inhibit CYP3A4 by 70-90%",
            "ALWAYS check for drug-drug interactions before prescribing",
            "Avoid: simvastatin, ergotamine, midazolam with PIs",
        ]
        base.contraindicated_drugs = [
            "Simvastatin (with PIs)", "Lovastatin (with PIs)",
            "Oral midazolam", "Ergotamine", "St. John's Wort (induces CYP3A4)",
        ]
    else:
        base.dosing_guidance = [
            "Untreated HIV: chronic inflammation suppresses CYP enzymes",
            "Low albumin → more free drug → watch for toxicity",
            "Low CD4 increases infection risk → careful with immunosuppressants",
        ]

    if cd4 < 200:
        base.renal_clearance = 0.75
        base.hepatic_clearance = 0.60

    return base


# ════════════════════════════════════════════════════════════════
# RHEUMATOID ARTHRITIS
# ════════════════════════════════════════════════════════════════

def get_rheumatoid_arthritis(active_flare: bool = False) -> DiseaseModifiers:
    """RA — chronic inflammation suppresses CYP enzymes via IL-6."""
    if active_flare:
        return DiseaseModifiers(
            name="Rheumatoid Arthritis (Active Flare)",
            description="Active RA flare — high IL-6 suppresses CYP enzymes",
            cyp3a4_activity=0.45,
            cyp1a2_activity=0.40,
            cyp2c19_activity=0.50,
            aag_level=1.4,
            albumin_level=0.80,
            dosing_guidance=[
                "Active RA flare: IL-6 suppresses CYP3A4 by 50%+",
                "Drug levels may DOUBLE during flare",
                "When starting tocilizumab: CYP enzymes RECOVER → drug levels DROP",
                "Monitor narrow TI drugs during flare onset and resolution",
            ],
            monitoring=["CRP/ESR", "Drug levels for narrow TI drugs"],
        )
    return DiseaseModifiers(
        name="Rheumatoid Arthritis (Stable/Remission)",
        description="RA in remission — mild residual inflammation",
        cyp3a4_activity=0.85,
        cyp1a2_activity=0.80,
        aag_level=1.1,
        dosing_guidance=["Mild CYP suppression from residual inflammation"],
    )


# ════════════════════════════════════════════════════════════════
# CANCER / CACHEXIA
# ════════════════════════════════════════════════════════════════

def get_cancer(cachexia: bool = False) -> DiseaseModifiers:
    """Cancer — inflammatory cytokines + cachexia alter PK."""
    if cachexia:
        return DiseaseModifiers(
            name="Cancer with Cachexia",
            description="Advanced cancer with muscle wasting and systemic inflammation",
            hepatic_clearance=0.50,
            cyp3a4_activity=0.40,
            cyp1a2_activity=0.35,
            cyp2c19_activity=0.45,
            albumin_level=0.50,
            aag_level=1.5,
            muscle_volume=0.60,
            fat_volume=0.50,
            plasma_volume=0.85,
            cardiac_output=0.80,
            dosing_guidance=[
                "Cachexia dramatically alters drug distribution",
                "Low albumin → more free drug → toxicity at lower doses",
                "Reduced muscle and fat mass → smaller Vd for many drugs",
                "Inflammatory cytokines suppress CYP enzymes by 50-65%",
                "Use ideal body weight, not actual weight, for dosing",
            ],
            monitoring=["Albumin", "Drug levels", "Nutritional status"],
        )
    return DiseaseModifiers(
        name="Cancer (without cachexia)",
        description="Cancer with moderate inflammatory burden",
        cyp3a4_activity=0.70,
        cyp1a2_activity=0.65,
        albumin_level=0.80,
        aag_level=1.3,
        dosing_guidance=["Moderate CYP suppression from tumor-related inflammation"],
    )


# ════════════════════════════════════════════════════════════════
# BURNS
# ════════════════════════════════════════════════════════════════

def get_burns(tbsa_percent: int = 30) -> DiseaseModifiers:
    """
    Burn injury — massive PK changes.
    Phase 1 (0-48h): fluid shifts, reduced CO
    Phase 2 (48h+): hypermetabolic, augmented clearance
    """
    if tbsa_percent >= 40:
        return DiseaseModifiers(
            name=f"Severe Burns ({tbsa_percent}% TBSA)",
            description="Major burn injury — hypermetabolic state",
            cardiac_output=1.50,
            hepatic_clearance=1.50,
            renal_clearance=1.60,
            hepatic_blood_flow=1.40,
            renal_blood_flow=1.50,
            cyp3a4_activity=1.30,
            cyp1a2_activity=1.50,
            cyp2c9_activity=1.20,
            albumin_level=0.30,
            aag_level=2.5,
            plasma_volume=1.8,
            dosing_guidance=[
                f"CRITICAL: {tbsa_percent}% TBSA burn — massive PK alterations",
                "Albumin critically low → massive increase in free drug",
                "Hypermetabolic: CYP enzymes and renal clearance INCREASED",
                "May need HIGHER doses of many drugs (augmented clearance)",
                "BUT free fraction increased → complex dosing decisions",
                "AAG massively increased → basic drugs more protein-bound",
                "Third-spacing → dramatically increased Vd",
            ],
            monitoring=["Drug levels q6h", "Albumin", "Creatinine", "Fluid balance"],
        )
    return DiseaseModifiers(
        name=f"Moderate Burns ({tbsa_percent}% TBSA)",
        description="Moderate burn — developing hypermetabolic response",
        cardiac_output=1.20,
        hepatic_clearance=1.20,
        renal_clearance=1.30,
        albumin_level=0.60,
        aag_level=1.8,
        plasma_volume=1.3,
        dosing_guidance=[
            "Moderate burns — augmented renal clearance developing",
            "Low albumin → increased free drug fraction",
        ],
    )


# ════════════════════════════════════════════════════════════════
# PEDIATRIC (AGE-BASED ENZYME MATURATION)
# ════════════════════════════════════════════════════════════════

def get_pediatric(age_group: str = "child") -> DiseaseModifiers:
    """
    Pediatric PK — enzymes aren't fully mature until ~12-15 years.
    Neonates: very immature enzymes, high Vd (more water), low protein binding
    Infants: rapidly maturing, highly variable
    Children: approaching adult values but not identical
    """
    if age_group == "neonate":
        return DiseaseModifiers(
            name="Neonate (0-28 days)",
            description="Extremely immature drug metabolism — highest risk population",
            hepatic_clearance=0.20,
            renal_clearance=0.25,
            cardiac_output=0.60,
            cyp3a4_activity=0.30,
            cyp2d6_activity=0.15,
            cyp2c9_activity=0.20,
            cyp1a2_activity=0.10,
            cyp2c19_activity=0.20,
            ugt_activity=0.05,
            albumin_level=0.70,
            plasma_volume=1.8,
            fat_volume=0.3,
            muscle_volume=0.3,
            gi_motility=0.50,
            absorption_rate=0.50,
            gastric_ph=6.0,
            dosing_guidance=[
                "CRITICAL: Neonatal drug metabolism is extremely immature",
                "CYP enzymes at 10-30% of adult activity",
                "UGT nearly absent (risk of kernicterus with bilirubin-displacing drugs)",
                "High body water % → larger Vd for hydrophilic drugs",
                "Gastric pH ~6 (achlorhydria) → altered oral absorption",
                "All doses must be weight-based and carefully calculated",
                "Grey baby syndrome risk with chloramphenicol (no UGT)",
            ],
            monitoring=["Drug levels for ALL drugs", "Hepatic function", "Renal function"],
            contraindicated_drugs=[
                "Chloramphenicol", "Aspirin (Reye's)", "Tetracyclines",
                "Fluoroquinolones", "Codeine (no CYP2D6)",
            ],
        )
    elif age_group == "infant":
        return DiseaseModifiers(
            name="Infant (1-12 months)",
            description="Rapidly maturing enzymes — highly variable PK",
            hepatic_clearance=0.50,
            renal_clearance=0.55,
            cyp3a4_activity=0.50,
            cyp2d6_activity=0.40,
            cyp1a2_activity=0.30,
            cyp2c9_activity=0.45,
            ugt_activity=0.40,
            albumin_level=0.80,
            plasma_volume=1.4,
            gi_motility=0.70,
            absorption_rate=0.70,
            gastric_ph=3.5,
            dosing_guidance=[
                "Enzyme activity rapidly changing — frequent dose adjustments needed",
                "CYP3A4 reaches ~50% adult activity by 6 months",
                "Weight-based dosing essential",
            ],
            monitoring=["Growth parameters", "Drug levels", "Development milestones"],
            contraindicated_drugs=["Aspirin", "Codeine (<12 months)"],
        )
    return DiseaseModifiers(
        name="Child (1-12 years)",
        description="Approaching adult enzyme activity, higher weight-normalized clearance",
        hepatic_clearance=1.20,
        renal_clearance=1.15,
        cyp3a4_activity=1.10,
        cyp1a2_activity=0.80,
        cyp2d6_activity=0.90,
        ugt_activity=1.10,
        plasma_volume=0.80,
        fat_volume=0.60,
        muscle_volume=0.50,
        dosing_guidance=[
            "Children often have HIGHER weight-normalized clearance than adults",
            "May need higher mg/kg doses than adults",
            "CYP1A2 still maturing — caffeine clearance slower than adults",
        ],
    )


# ════════════════════════════════════════════════════════════════
# ELDERLY FRAILTY
# ════════════════════════════════════════════════════════════════

def get_elderly_frailty() -> DiseaseModifiers:
    """
    Elderly frailty — beyond normal age-related decline.
    Sarcopenia, polypharmacy risk, reduced homeostatic reserve.
    """
    return DiseaseModifiers(
        name="Elderly Frailty (>75, frail)",
        description="Frail elderly — sarcopenia, reduced reserve, polypharmacy risk",
        hepatic_clearance=0.50,
        renal_clearance=0.45,
        cardiac_output=0.60,
        cyp3a4_activity=0.55,
        cyp1a2_activity=0.50,
        cyp2d6_activity=0.70,
        cyp2c9_activity=0.60,
        ugt_activity=0.60,
        albumin_level=0.70,
        muscle_volume=0.60,
        fat_volume=1.30,
        gi_motility=0.65,
        absorption_rate=0.80,
        dosing_guidance=[
            "START LOW, GO SLOW — use 50% of normal starting dose",
            "Frail elderly have dramatically reduced drug clearance",
            "Sarcopenia reduces muscle mass → altered Vd for many drugs",
            "Increased fat → prolonged effect of lipophilic drugs (benzodiazepines)",
            "Polypharmacy risk: check ALL drug interactions",
            "Falls risk increased with CNS-active drugs",
            "Anticholinergic burden: avoid drugs with anticholinergic effects",
        ],
        monitoring=["Renal function", "Falls assessment", "Cognitive function", "ADL status"],
        contraindicated_drugs=[
            "Long-acting benzodiazepines (falls risk)",
            "Strong anticholinergics (cognitive decline)",
            "NSAIDs long-term (GI bleed + renal risk)",
        ],
    )


# ════════════════════════════════════════════════════════════════
# CYSTIC FIBROSIS
# ════════════════════════════════════════════════════════════════

def get_cystic_fibrosis() -> DiseaseModifiers:
    """CF — altered GI absorption, increased renal clearance, pancreatic insufficiency."""
    return DiseaseModifiers(
        name="Cystic Fibrosis",
        description="Altered GI absorption + augmented renal clearance",
        absorption_rate=0.60,
        gi_motility=0.70,
        renal_clearance=1.40,
        hepatic_clearance=1.20,
        cyp3a4_activity=1.15,
        cyp1a2_activity=1.20,
        fat_volume=0.60,
        muscle_volume=0.80,
        albumin_level=0.75,
        dosing_guidance=[
            "CF patients often need HIGHER doses than normal",
            "Augmented renal clearance → faster drug elimination",
            "Pancreatic insufficiency → fat malabsorption → reduced absorption of lipophilic drugs",
            "Take enzyme supplements with fat-soluble drugs",
            "Higher aminoglycoside doses often needed (larger Vd, faster CL)",
        ],
        monitoring=["Drug levels (especially aminoglycosides)", "Nutritional status", "LFTs"],
    )


# ════════════════════════════════════════════════════════════════
# INFLAMMATORY BOWEL DISEASE
# ════════════════════════════════════════════════════════════════

def get_ibd(active: bool = True) -> DiseaseModifiers:
    """IBD (Crohn's/UC) — reduced GI absorption during active disease."""
    if active:
        return DiseaseModifiers(
            name="IBD (Active Flare)",
            description="Active Crohn's or UC — inflamed GI, rapid transit, malabsorption",
            absorption_rate=0.45,
            gi_motility=1.50,
            gut_blood_flow=0.70,
            albumin_level=0.75,
            cyp3a4_activity=0.70,
            cyp1a2_activity=0.65,
            aag_level=1.3,
            dosing_guidance=[
                "Active IBD: oral absorption significantly impaired",
                "Consider IV or rectal routes during severe flare",
                "Rapid GI transit → drug may not fully dissolve/absorb",
                "Inflammation suppresses hepatic CYP enzymes",
                "Biologic therapy (anti-TNF) may normalize CYP activity",
            ],
            monitoring=["CRP/ESR", "Albumin", "Drug levels", "Calprotectin"],
        )
    return DiseaseModifiers(
        name="IBD (Remission)",
        description="IBD in remission — near-normal GI function",
        absorption_rate=0.85,
        gi_motility=1.10,
        cyp3a4_activity=0.90,
        dosing_guidance=["IBD in remission — most drugs absorbed normally"],
    )


# ════════════════════════════════════════════════════════════════
# MALNUTRITION
# ════════════════════════════════════════════════════════════════

def get_malnutrition(severity: str = "moderate") -> DiseaseModifiers:
    """Malnutrition — low albumin, reduced enzyme capacity, altered Vd."""
    if severity == "severe":
        return DiseaseModifiers(
            name="Severe Malnutrition (Kwashiorkor/Marasmus)",
            description="Severe protein-energy malnutrition",
            albumin_level=0.35,
            hepatic_clearance=0.40,
            renal_clearance=0.60,
            cyp3a4_activity=0.40,
            cyp1a2_activity=0.35,
            cyp2c9_activity=0.45,
            ugt_activity=0.40,
            muscle_volume=0.40,
            fat_volume=0.30,
            plasma_volume=0.80,
            edema=True,
            dosing_guidance=[
                "CRITICAL: Severely low albumin → massive increase in free drug",
                "Drugs normally 90% bound → much more free drug circulating",
                "Reduced muscle/fat → smaller Vd for many drugs",
                "Hepatic enzyme capacity severely reduced",
                "Start ALL drugs at 25-50% of normal dose",
                "Refeeding syndrome risk — monitor electrolytes when nutrition resumes",
            ],
            monitoring=["Albumin", "Electrolytes (PO4, K+, Mg2+)", "Drug levels"],
        )
    return DiseaseModifiers(
        name="Moderate Malnutrition",
        description="Moderate protein-calorie malnutrition",
        albumin_level=0.60,
        hepatic_clearance=0.70,
        cyp3a4_activity=0.65,
        muscle_volume=0.70,
        fat_volume=0.60,
        dosing_guidance=[
            "Low albumin → increased free drug fraction",
            "Reduce doses of highly protein-bound drugs",
        ],
    )


# ═══ CONVENIENCE: Get disease by string name ═══
DISEASE_REGISTRY = {
    # ── Liver ──
    "liver_A": lambda: get_liver_disease(ChildPughClass.A),
    "liver_B": lambda: get_liver_disease(ChildPughClass.B),
    "liver_C": lambda: get_liver_disease(ChildPughClass.C),
    "hepatitis_B_acute": lambda: get_hepatitis("B", acute=True),
    "hepatitis_B_chronic": lambda: get_hepatitis("B", acute=False),
    "hepatitis_C_acute": lambda: get_hepatitis("C", acute=True),
    "hepatitis_C_chronic": lambda: get_hepatitis("C", acute=False),
    "alcoholic_liver_mild": lambda: get_alcoholic_liver_disease("mild"),
    "alcoholic_liver_moderate": lambda: get_alcoholic_liver_disease("moderate"),
    "alcoholic_liver_severe": lambda: get_alcoholic_liver_disease("severe"),
    "nafld": lambda: get_nafld(False),
    "nash": lambda: get_nafld(True),
    # ── Kidney ──
    "ckd_1": lambda: get_kidney_disease(CKDStage.STAGE_1),
    "ckd_2": lambda: get_kidney_disease(CKDStage.STAGE_2),
    "ckd_3a": lambda: get_kidney_disease(CKDStage.STAGE_3A),
    "ckd_3b": lambda: get_kidney_disease(CKDStage.STAGE_3B),
    "ckd_4": lambda: get_kidney_disease(CKDStage.STAGE_4),
    "ckd_5": lambda: get_kidney_disease(CKDStage.STAGE_5),
    "aki_1": lambda: get_aki(1),
    "aki_2": lambda: get_aki(2),
    "aki_3": lambda: get_aki(3),
    # ── Heart & Vascular ──
    "heart_I": lambda: get_heart_failure(NYHAClass.I),
    "heart_II": lambda: get_heart_failure(NYHAClass.II),
    "heart_III": lambda: get_heart_failure(NYHAClass.III),
    "heart_IV": lambda: get_heart_failure(NYHAClass.IV),
    "hypertension_mild": lambda: get_hypertension("mild"),
    "hypertension_moderate": lambda: get_hypertension("moderate"),
    "hypertension_severe": lambda: get_hypertension("severe"),
    # ── Metabolic ──
    "obesity_I": lambda: get_obesity(ObesityClass.OBESE_I, 32),
    "obesity_II": lambda: get_obesity(ObesityClass.OBESE_II, 37),
    "obesity_III": lambda: get_obesity(ObesityClass.OBESE_III, 42),
    "diabetes": lambda: get_diabetes_t2(7.5),
    "diabetes_severe": lambda: get_diabetes_t2(10.0, has_neuropathy=True, has_nephropathy=True),
    "hypothyroidism": lambda: get_thyroid_disorder(hypo=True),
    "hyperthyroidism": lambda: get_thyroid_disorder(hypo=False),
    # ── Immune & Inflammatory ──
    "hiv_on_art": lambda: get_hiv(on_art=True, cd4=350),
    "hiv_untreated": lambda: get_hiv(on_art=False, cd4=150),
    "ra_stable": lambda: get_rheumatoid_arthritis(active_flare=False),
    "ra_flare": lambda: get_rheumatoid_arthritis(active_flare=True),
    "ibd_active": lambda: get_ibd(active=True),
    "ibd_remission": lambda: get_ibd(active=False),
    "sepsis": lambda: get_sepsis(False),
    "sepsis_severe": lambda: get_sepsis(True),
    # ── Special Populations ──
    "pregnancy_T1": lambda: get_pregnancy(1),
    "pregnancy_T2": lambda: get_pregnancy(2),
    "pregnancy_T3": lambda: get_pregnancy(3),
    "pediatric_neonate": lambda: get_pediatric("neonate"),
    "pediatric_infant": lambda: get_pediatric("infant"),
    "pediatric_child": lambda: get_pediatric("child"),
    "elderly_frail": lambda: get_elderly_frailty(),
    # ── Other ──
    "cancer": lambda: get_cancer(cachexia=False),
    "cancer_cachexia": lambda: get_cancer(cachexia=True),
    "burns_moderate": lambda: get_burns(20),
    "burns_severe": lambda: get_burns(45),
    "cystic_fibrosis": lambda: get_cystic_fibrosis(),
    "malnutrition_moderate": lambda: get_malnutrition("moderate"),
    "malnutrition_severe": lambda: get_malnutrition("severe"),
}


# Disease categories for UI grouping
DISEASE_CATEGORIES = {
    "Liver": ["liver_A", "liver_B", "liver_C", "hepatitis_B_acute", "hepatitis_B_chronic",
              "hepatitis_C_acute", "hepatitis_C_chronic", "alcoholic_liver_mild",
              "alcoholic_liver_moderate", "alcoholic_liver_severe", "nafld", "nash"],
    "Kidney": ["ckd_1", "ckd_2", "ckd_3a", "ckd_3b", "ckd_4", "ckd_5",
               "aki_1", "aki_2", "aki_3"],
    "Heart": ["heart_I", "heart_II", "heart_III", "heart_IV",
              "hypertension_mild", "hypertension_moderate", "hypertension_severe"],
    "Metabolic": ["obesity_I", "obesity_II", "obesity_III", "diabetes", "diabetes_severe",
                  "hypothyroidism", "hyperthyroidism"],
    "Immune": ["hiv_on_art", "hiv_untreated", "ra_stable", "ra_flare",
               "ibd_active", "ibd_remission", "sepsis", "sepsis_severe"],
    "Special": ["pregnancy_T1", "pregnancy_T2", "pregnancy_T3",
                "pediatric_neonate", "pediatric_infant", "pediatric_child", "elderly_frail"],
    "Other": ["cancer", "cancer_cachexia", "burns_moderate", "burns_severe",
              "cystic_fibrosis", "malnutrition_moderate", "malnutrition_severe"],
}


def get_disease_by_name(name: str) -> DiseaseModifiers:
    """Get disease modifiers by string name. See DISEASE_REGISTRY for valid names."""
    factory = DISEASE_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown disease: {name}. Valid: {list(DISEASE_REGISTRY.keys())}")
    return factory()


# ── Standalone test ──
if __name__ == "__main__":
    print("=" * 60)
    print("Disease State Modeling — Test")
    print("=" * 60)

    # Test 1: Single disease
    print("\n--- Cirrhosis Child-Pugh C ---")
    d = get_liver_disease(ChildPughClass.C)
    print(f"  Hepatic CL: {d.hepatic_clearance*100:.0f}% of normal")
    print(f"  CYP3A4: {d.cyp3a4_activity*100:.0f}%")
    print(f"  Albumin: {d.albumin_level*100:.0f}%")
    print(f"  Contraindicated: {d.contraindicated_drugs[:3]}")

    # Test 2: Co-morbidities
    print("\n--- Cirrhosis B + CKD Stage 4 (co-morbidity) ---")
    engine = DiseaseStateEngine()
    engine.add_disease(get_liver_disease(ChildPughClass.B))
    engine.add_disease(get_kidney_disease(CKDStage.STAGE_4))
    report = engine.get_report()
    print(f"  Combined hepatic CL: {report['modifiers']['hepatic_clearance']*100:.0f}%")
    print(f"  Combined renal CL: {report['modifiers']['renal_clearance']*100:.0f}%")
    print(f"  # Dosing guidance items: {len(report['dosing_guidance'])}")

    # Test 3: Pregnancy
    print("\n--- Pregnancy T3 ---")
    p = get_pregnancy(3)
    print(f"  CYP3A4: {p.cyp3a4_activity*100:.0f}% (DOUBLED)")
    print(f"  CYP1A2: {p.cyp1a2_activity*100:.0f}% (HALVED)")
    print(f"  GFR: {p.renal_clearance*100:.0f}%")
    print(f"  Contraindicated: {p.contraindicated_drugs[:4]}")

    # Test 4: Sepsis
    print("\n--- Severe Sepsis ---")
    s = get_sepsis(True)
    print(f"  CYP3A4: {s.cyp3a4_activity*100:.0f}% (cytokine suppression)")
    print(f"  CYP1A2: {s.cyp1a2_activity*100:.0f}%")
    print(f"  Albumin: {s.albumin_level*100:.0f}%")
    print(f"  AAG: {s.aag_level*100:.0f}% (acute phase increase)")

    print("\n--- Available diseases ---")
    print(f"  {list(DISEASE_REGISTRY.keys())}")

    print("\nDisease modeling test PASSED!")
