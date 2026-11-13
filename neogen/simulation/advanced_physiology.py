"""
Advanced Physiology Module
===========================
Adds 5 realistic physiological systems to the PBPK engine:

1. Enterohepatic Recirculation (EHR)
   Drug excreted into bile → reabsorbed from gut → creates secondary PK peaks

2. pH-Dependent Absorption
   Drug ionization changes with GI pH → affects absorption rate
   PPIs (omeprazole) raise stomach pH → alters other drugs' absorption

3. Food Effects
   High-fat meals increase lipophilic drug absorption
   Grapefruit blocks gut CYP3A4 → increases bioavailability

4. Transporter Proteins
   P-gp, OATP1B1, OCT2, BCRP, MRP2 — active pumps that move drugs
   between compartments regardless of concentration gradient

5. Active Metabolite Tracking
   Parent drug → metabolites (active, inactive, toxic)
   Codeine → morphine, paracetamol → NAPQI, clopidogrel → active thiol
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. ENTEROHEPATIC RECIRCULATION
# ════════════════════════════════════════════════════════════════

@dataclass
class EHRProfile:
    """Enterohepatic recirculation parameters for a drug."""
    fraction_biliary: float = 0.0    # Fraction of hepatic clearance excreted into bile (0-1)
    reabsorption_rate: float = 0.0   # Rate of reabsorption from gut (1/h)
    cycle_delay_h: float = 4.0       # Delay for bile to reach gut (hours)
    n_cycles: int = 0                # Typical number of recirculation cycles


# Known drugs with significant EHR
EHR_DATABASE = {
    # drug_name: (fraction_biliary, reabsorption_rate, cycle_delay_h)
    "morphine":           EHRProfile(0.40, 0.8, 4.0, 3),
    "ethinylestradiol":   EHRProfile(0.45, 0.9, 5.0, 3),
    "mycophenolate":      EHRProfile(0.50, 1.0, 6.0, 4),
    "ezetimibe":          EHRProfile(0.70, 0.6, 4.0, 5),
    "indomethacin":       EHRProfile(0.30, 0.7, 3.5, 2),
    "diclofenac":         EHRProfile(0.35, 0.8, 4.0, 2),
    "piroxicam":          EHRProfile(0.25, 0.5, 5.0, 2),
    "warfarin":           EHRProfile(0.10, 0.4, 4.0, 1),
    "rifampicin":         EHRProfile(0.30, 0.6, 3.0, 2),
    "chloramphenicol":    EHRProfile(0.20, 0.5, 4.0, 2),
    "methotrexate":       EHRProfile(0.35, 0.7, 5.0, 3),
    "digitoxin":          EHRProfile(0.30, 0.8, 6.0, 4),
    "sorafenib":          EHRProfile(0.50, 0.6, 5.0, 3),
    "raloxifene":         EHRProfile(0.60, 0.9, 4.0, 4),
}


def get_ehr_profile(smiles: str = "", drug_name: str = "") -> EHRProfile:
    """Get EHR profile for a drug. Returns default (no EHR) if unknown."""
    name_lower = drug_name.lower().strip()
    for known_name, profile in EHR_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            logger.info("EHR profile found for %s: biliary=%.0f%%", drug_name, profile.fraction_biliary * 100)
            return profile

    # Structural prediction: glucuronide conjugates often undergo EHR
    # High MW (>400) compounds tend to have more biliary excretion
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.ExactMolWt(mol)
            # Drugs with MW > 500 tend to have significant biliary excretion
            if mw > 500:
                f_bil = min(0.30, (mw - 500) / 1000)
                return EHRProfile(f_bil, 0.5, 4.0, 2)
            elif mw > 400:
                f_bil = min(0.15, (mw - 400) / 1000)
                return EHRProfile(f_bil, 0.3, 4.0, 1)
    except Exception:
        pass

    return EHRProfile()  # No EHR


# ════════════════════════════════════════════════════════════════
# 2. pH-DEPENDENT ABSORPTION
# ════════════════════════════════════════════════════════════════

@dataclass
class pHAbsorptionProfile:
    """pH-dependent absorption parameters."""
    pka: float = 7.0          # Drug pKa
    is_acid: bool = True       # True = weak acid, False = weak base
    stomach_pH: float = 1.5    # Normal fasting stomach pH
    duodenum_pH: float = 6.0   # Duodenal pH
    # Modifiers
    ppi_effect: bool = False   # Is patient on PPI? (raises stomach pH to 4-5)
    antacid_effect: bool = False  # Is patient on antacids?

    def get_ionization_fraction(self, pH: float) -> float:
        """Henderson-Hasselbalch: fraction unionized (absorbable)."""
        if self.is_acid:
            # Weak acid: unionized at pH < pKa
            ratio = 10 ** (pH - self.pka)
            return 1.0 / (1.0 + ratio)
        else:
            # Weak base: unionized at pH > pKa
            ratio = 10 ** (self.pka - pH)
            return 1.0 / (1.0 + ratio)

    def get_effective_stomach_pH(self) -> float:
        """Stomach pH adjusted for PPI/antacid use."""
        pH = self.stomach_pH
        if self.ppi_effect:
            pH = max(pH, 4.5)  # PPI raises to ~4.5-5
        if self.antacid_effect:
            pH = max(pH, 3.5)  # Antacids raise to ~3.5
        return pH

    def get_absorption_modifier(self) -> float:
        """
        Overall absorption rate modifier based on pH.
        Returns multiplier on ka (absorption rate constant).
        """
        stomach_pH = self.get_effective_stomach_pH()

        # Fraction unionized in stomach vs duodenum
        f_stomach = self.get_ionization_fraction(stomach_pH)
        f_duodenum = self.get_ionization_fraction(self.duodenum_pH)

        # Most absorption occurs in duodenum (large surface area)
        # but stomach absorption matters for weak acids
        weighted = 0.2 * f_stomach + 0.8 * f_duodenum

        # Clamp to reasonable range
        return np.clip(weighted / 0.5, 0.3, 2.0)  # Normalized to default


# Known drug pKa values
PKA_DATABASE = {
    "aspirin": (3.5, True),       # Weak acid
    "ibuprofen": (4.4, True),
    "naproxen": (4.2, True),
    "diclofenac": (4.0, True),
    "warfarin": (5.0, True),
    "phenytoin": (8.3, True),
    "indomethacin": (4.5, True),
    "furosemide": (3.9, True),
    "metformin": (12.4, False),   # Weak base
    "caffeine": (14.0, False),    # Very weak base (essentially neutral)
    "morphine": (8.0, False),
    "codeine": (8.2, False),
    "diphenhydramine": (8.3, False),
    "propranolol": (9.5, False),
    "metoprolol": (9.7, False),
    "amitriptyline": (9.4, False),
    "diazepam": (3.4, False),
    "chloroquine": (8.1, False),
    "ketoconazole": (6.5, False),
    "omeprazole": (4.0, False),   # Weak base (prodrug)
    "ranitidine": (8.2, False),
    "ciprofloxacin": (6.1, True), # Zwitterion
    "ampicillin": (2.5, True),
    "paracetamol": (9.5, True),   # Very weak acid (neutral-ish)
}


def get_ph_absorption_profile(smiles: str = "", drug_name: str = "",
                               ppi: bool = False, antacid: bool = False) -> pHAbsorptionProfile:
    """Get pH absorption profile for a drug."""
    name_lower = drug_name.lower().strip()

    for known_name, (pka, is_acid) in PKA_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            return pHAbsorptionProfile(pka=pka, is_acid=is_acid, ppi_effect=ppi, antacid_effect=antacid)

    # Predict from structure
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Simple heuristic: count acidic/basic groups
            has_cooh = smiles.count("C(=O)O") > 0 or smiles.count("C(O)=O") > 0
            has_amine = smiles.count("N") > 0 and smiles.count("N(=O)") == 0

            if has_cooh:
                return pHAbsorptionProfile(pka=4.0, is_acid=True, ppi_effect=ppi, antacid_effect=antacid)
            elif has_amine:
                return pHAbsorptionProfile(pka=8.5, is_acid=False, ppi_effect=ppi, antacid_effect=antacid)
    except Exception:
        pass

    return pHAbsorptionProfile(pka=7.0, is_acid=True, ppi_effect=ppi, antacid_effect=antacid)


# ════════════════════════════════════════════════════════════════
# 3. FOOD EFFECTS
# ════════════════════════════════════════════════════════════════

@dataclass
class FoodEffectProfile:
    """How food affects drug absorption."""
    fed_state: str = "fasted"         # "fasted", "light_meal", "high_fat"
    absorption_multiplier: float = 1.0  # Effect on ka
    bioavail_multiplier: float = 1.0    # Effect on F
    tmax_delay_h: float = 0.0           # Delay in Tmax
    mechanism: str = ""                  # Explanation


# BCS-class based food effects (FDA guidance)
BCS_FOOD_EFFECTS = {
    # BCS I: high solubility, high permeability → minimal food effect
    1: FoodEffectProfile(absorption_multiplier=0.9, bioavail_multiplier=1.0, tmax_delay_h=0.5,
                         mechanism="BCS I: Minimal food effect. Slight Tmax delay."),
    # BCS II: low solubility, high permeability → food INCREASES absorption
    2: FoodEffectProfile(absorption_multiplier=1.3, bioavail_multiplier=1.5, tmax_delay_h=1.0,
                         mechanism="BCS II: Food increases bile → dissolves lipophilic drug → higher F."),
    # BCS III: high solubility, low permeability → food DECREASES absorption
    3: FoodEffectProfile(absorption_multiplier=0.7, bioavail_multiplier=0.8, tmax_delay_h=1.5,
                         mechanism="BCS III: Food competes for intestinal transporters → lower absorption."),
    # BCS IV: low solubility, low permeability → variable
    4: FoodEffectProfile(absorption_multiplier=1.0, bioavail_multiplier=1.0, tmax_delay_h=1.0,
                         mechanism="BCS IV: Unpredictable food effect. Case-by-case."),
}


# Drug-specific food effects (known clinical data)
FOOD_EFFECT_DATABASE = {
    "griseofulvin":    FoodEffectProfile(bioavail_multiplier=2.0, tmax_delay_h=1.5,
                       mechanism="High-fat meal doubles absorption. Take with fatty food."),
    "itraconazole":    FoodEffectProfile(bioavail_multiplier=1.8, tmax_delay_h=1.0,
                       mechanism="Acidic cola/food increases dissolution of capsule form."),
    "albendazole":     FoodEffectProfile(bioavail_multiplier=5.0, tmax_delay_h=2.0,
                       mechanism="Fatty food increases F from 5% to 25%."),
    "saquinavir":      FoodEffectProfile(bioavail_multiplier=6.0, tmax_delay_h=2.0,
                       mechanism="Must take with high-calorie, high-fat meal."),
    "abiraterone":     FoodEffectProfile(bioavail_multiplier=5.0, tmax_delay_h=1.0,
                       mechanism="10x increase with high-fat meal. Take on empty stomach."),
    "erlotinib":       FoodEffectProfile(bioavail_multiplier=1.6, tmax_delay_h=1.0,
                       mechanism="Food increases AUC by 60%. Take 1h before or 2h after meals."),
    "tetracycline":    FoodEffectProfile(bioavail_multiplier=0.5, absorption_multiplier=0.5, tmax_delay_h=2.0,
                       mechanism="Dairy chelates tetracycline → 50% reduction. Avoid milk/calcium."),
    "ciprofloxacin":   FoodEffectProfile(bioavail_multiplier=0.7, tmax_delay_h=1.0,
                       mechanism="Metal ions (Ca²⁺, Fe²⁺, Al³⁺) chelate → reduced absorption."),
    "levothyroxine":   FoodEffectProfile(bioavail_multiplier=0.6, tmax_delay_h=2.0,
                       mechanism="Food/coffee/calcium reduce absorption. Take 30-60min before breakfast."),
    "paracetamol":     FoodEffectProfile(absorption_multiplier=0.6, bioavail_multiplier=1.0, tmax_delay_h=1.5,
                       mechanism="Food delays absorption (slower gastric emptying) but F unchanged."),
    "aspirin":         FoodEffectProfile(absorption_multiplier=0.7, bioavail_multiplier=0.9, tmax_delay_h=1.0,
                       mechanism="Food delays absorption, slightly reduces total exposure."),
}


GRAPEFRUIT_AFFECTED = {
    # Drugs where grapefruit juice inhibits gut CYP3A4 → increased F
    "simvastatin": 3.0,      # 3x increase in AUC
    "lovastatin": 3.5,
    "atorvastatin": 1.8,
    "felodipine": 2.8,
    "nifedipine": 1.5,
    "cyclosporine": 1.6,
    "tacrolimus": 1.5,
    "buspirone": 9.0,        # 9x increase!
    "carbamazepine": 1.4,
    "midazolam": 1.5,
    "triazolam": 1.5,
    "sildenafil": 1.3,
    "sertraline": 1.5,
    "amiodarone": 1.5,
}


def get_food_effect(smiles: str = "", drug_name: str = "", bcs_class: int = 1,
                     fed_state: str = "fasted", grapefruit: bool = False) -> FoodEffectProfile:
    """Get food effect profile for a drug."""
    name_lower = drug_name.lower().strip()

    # Check drug-specific database first
    for known_name, profile in FOOD_EFFECT_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            if fed_state == "fasted":
                return FoodEffectProfile(mechanism=profile.mechanism + " (Currently fasted)")
            profile.fed_state = fed_state
            # Scale effect by meal size
            if fed_state == "light_meal":
                profile.bioavail_multiplier = 1.0 + (profile.bioavail_multiplier - 1.0) * 0.5
                profile.tmax_delay_h *= 0.5
            return profile

    # Fall back to BCS-class prediction
    if fed_state != "fasted" and bcs_class in BCS_FOOD_EFFECTS:
        profile = BCS_FOOD_EFFECTS[bcs_class]
        profile.fed_state = fed_state
        if fed_state == "light_meal":
            profile.bioavail_multiplier = 1.0 + (profile.bioavail_multiplier - 1.0) * 0.5
            profile.tmax_delay_h *= 0.5
        return profile

    result = FoodEffectProfile(fed_state=fed_state)

    # Grapefruit effect
    if grapefruit:
        for known_name, multiplier in GRAPEFRUIT_AFFECTED.items():
            if name_lower and len(name_lower) >= 3 and known_name in name_lower:
                result.bioavail_multiplier *= multiplier
                result.mechanism = f"Grapefruit inhibits gut CYP3A4 → {multiplier}x AUC increase"
                break

    return result


# ════════════════════════════════════════════════════════════════
# 4. TRANSPORTER PROTEINS
# ════════════════════════════════════════════════════════════════

@dataclass
class TransporterProfile:
    """Transporter-mediated drug movement."""
    # Efflux transporters (pump drug OUT of cells)
    pgp_substrate: bool = False       # P-glycoprotein (MDR1/ABCB1)
    pgp_km: float = 10.0             # Michaelis constant (µM)
    pgp_vmax: float = 100.0          # Max velocity (µmol/h)

    bcrp_substrate: bool = False      # Breast cancer resistance protein (ABCG2)
    mrp2_substrate: bool = False      # Multidrug resistance protein 2 (bile canaliculus)

    # Uptake transporters (pull drug INTO cells)
    oatp1b1_substrate: bool = False   # Liver uptake (statins, sartans)
    oatp1b3_substrate: bool = False   # Liver uptake
    oct2_substrate: bool = False       # Kidney organic cation transporter
    oat1_substrate: bool = False       # Kidney organic anion transporter
    oat3_substrate: bool = False       # Kidney organic anion transporter

    # Net effects on PBPK parameters
    gut_efflux_factor: float = 1.0    # <1 = P-gp reduces gut absorption
    liver_uptake_factor: float = 1.0  # >1 = OATP increases liver conc
    kidney_secretion_factor: float = 1.0  # >1 = OCT/OAT increases renal CL
    brain_efflux_factor: float = 1.0  # <1 = P-gp/BCRP blocks brain entry
    biliary_efflux_factor: float = 1.0  # >1 = MRP2 increases biliary excretion


# Known transporter substrates
TRANSPORTER_DATABASE = {
    "digoxin":       TransporterProfile(pgp_substrate=True, gut_efflux_factor=0.3, brain_efflux_factor=0.1,
                     biliary_efflux_factor=2.0),
    "dabigatran":    TransporterProfile(pgp_substrate=True, gut_efflux_factor=0.15),
    "loperamide":    TransporterProfile(pgp_substrate=True, brain_efflux_factor=0.05),  # P-gp keeps it out of brain
    "cyclosporine":  TransporterProfile(pgp_substrate=True, oatp1b1_substrate=True, gut_efflux_factor=0.4,
                     liver_uptake_factor=2.5),
    "atorvastatin":  TransporterProfile(oatp1b1_substrate=True, pgp_substrate=True, bcrp_substrate=True,
                     liver_uptake_factor=3.0, gut_efflux_factor=0.7),
    "rosuvastatin":  TransporterProfile(oatp1b1_substrate=True, oatp1b3_substrate=True, bcrp_substrate=True,
                     liver_uptake_factor=4.0, brain_efflux_factor=0.02),
    "simvastatin":   TransporterProfile(oatp1b1_substrate=True, liver_uptake_factor=2.0),
    "pravastatin":   TransporterProfile(oatp1b1_substrate=True, oatp1b3_substrate=True,
                     liver_uptake_factor=3.5, gut_efflux_factor=0.6),
    "metformin":     TransporterProfile(oct2_substrate=True, kidney_secretion_factor=3.0),
    "cisplatin":     TransporterProfile(oct2_substrate=True, kidney_secretion_factor=2.5),
    "penicillin":    TransporterProfile(oat1_substrate=True, oat3_substrate=True, kidney_secretion_factor=4.0),
    "furosemide":    TransporterProfile(oat1_substrate=True, kidney_secretion_factor=3.0, mrp2_substrate=True,
                     biliary_efflux_factor=1.5),
    "methotrexate":  TransporterProfile(oatp1b1_substrate=True, oat1_substrate=True, mrp2_substrate=True,
                     liver_uptake_factor=2.0, kidney_secretion_factor=3.0, biliary_efflux_factor=2.0),
    "topotecan":     TransporterProfile(bcrp_substrate=True, pgp_substrate=True, gut_efflux_factor=0.4,
                     brain_efflux_factor=0.1),
    "irinotecan":    TransporterProfile(pgp_substrate=True, mrp2_substrate=True, biliary_efflux_factor=3.0),
    "morphine":      TransporterProfile(pgp_substrate=True, brain_efflux_factor=0.5),
    "fexofenadine":  TransporterProfile(pgp_substrate=True, oatp1b1_substrate=True, gut_efflux_factor=0.5,
                     liver_uptake_factor=2.0, brain_efflux_factor=0.05),
    "valsartan":     TransporterProfile(oatp1b1_substrate=True, oatp1b3_substrate=True, mrp2_substrate=True,
                     liver_uptake_factor=3.0, biliary_efflux_factor=2.5),
}


def get_transporter_profile(smiles: str = "", drug_name: str = "") -> TransporterProfile:
    """Get transporter profile for a drug."""
    name_lower = drug_name.lower().strip()

    for known_name, profile in TRANSPORTER_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            logger.info("Transporter profile: %s (P-gp=%s, OATP=%s, OCT2=%s)",
                       drug_name, profile.pgp_substrate, profile.oatp1b1_substrate, profile.oct2_substrate)
            return profile

    # Structural prediction
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            profile = TransporterProfile()

            # P-gp: tends to efflux large, lipophilic molecules
            if mw > 400 and logp > 2.5:
                profile.pgp_substrate = True
                profile.gut_efflux_factor = max(0.3, 1.0 - (mw - 400) / 800)
                profile.brain_efflux_factor = max(0.1, 1.0 - (mw - 400) / 600)

            # OATP1B1: anionic drugs with moderate MW
            if tpsa > 75 and 300 < mw < 900:
                profile.oatp1b1_substrate = True
                profile.liver_uptake_factor = 1.5 + tpsa / 200

            # OCT2: small cationic molecules
            has_amine = smiles.count("N") > 0 and smiles.count("[NH+]") + smiles.count("[nH]") > 0
            if has_amine and mw < 500:
                profile.oct2_substrate = True
                profile.kidney_secretion_factor = 2.0

            return profile
    except Exception:
        pass

    return TransporterProfile()


# ════════════════════════════════════════════════════════════════
# 5. ACTIVE METABOLITE TRACKING
# ════════════════════════════════════════════════════════════════

@dataclass
class MetaboliteInfo:
    """Information about a drug metabolite."""
    name: str                         # Metabolite name
    type: str = "inactive"            # "active", "inactive", "toxic", "prodrug"
    fraction: float = 0.0             # Fraction of parent converted to this metabolite
    activity_ratio: float = 0.0       # Relative potency vs parent (1.0 = equipotent)
    half_life_h: float = 4.0          # Metabolite half-life
    clearance_pathway: str = "renal"  # "renal", "hepatic", "biliary"
    cyp_enzyme: str = ""              # CYP responsible for formation
    toxicity_concern: str = ""        # Any toxicity concern


@dataclass
class MetaboliteProfile:
    """Complete metabolite profile for a drug."""
    parent_name: str = ""
    is_prodrug: bool = False          # Parent is inactive, metabolite is active
    metabolites: list = field(default_factory=list)

    def get_active_metabolites(self) -> list[MetaboliteInfo]:
        return [m for m in self.metabolites if m.type in ("active", "prodrug")]

    def get_toxic_metabolites(self) -> list[MetaboliteInfo]:
        return [m for m in self.metabolites if m.type == "toxic"]

    def total_active_fraction(self) -> float:
        return sum(m.fraction * m.activity_ratio for m in self.get_active_metabolites())


# Known metabolite profiles
METABOLITE_DATABASE = {
    "codeine": MetaboliteProfile(
        parent_name="codeine",
        is_prodrug=True,
        metabolites=[
            MetaboliteInfo("morphine", "prodrug", 0.10, 10.0, 3.0, "hepatic", "CYP2D6",
                          "Active analgesic. CYP2D6 poor metabolizers get NO pain relief."),
            MetaboliteInfo("norcodeine", "inactive", 0.10, 0.0, 4.0, "hepatic", "CYP3A4"),
            MetaboliteInfo("codeine-6-glucuronide", "active", 0.70, 0.5, 3.0, "renal", "UGT2B7"),
        ]
    ),
    "paracetamol": MetaboliteProfile(
        parent_name="paracetamol",
        metabolites=[
            MetaboliteInfo("NAPQI", "toxic", 0.05, 0.0, 0.5, "hepatic", "CYP2E1",
                          "Highly reactive. Detoxified by GSH. At overdose, GSH depleted → liver necrosis."),
            MetaboliteInfo("paracetamol-glucuronide", "inactive", 0.55, 0.0, 2.0, "renal", "UGT1A1"),
            MetaboliteInfo("paracetamol-sulfate", "inactive", 0.30, 0.0, 2.0, "renal", "SULT1A1"),
        ]
    ),
    "clopidogrel": MetaboliteProfile(
        parent_name="clopidogrel",
        is_prodrug=True,
        metabolites=[
            MetaboliteInfo("active thiol metabolite", "prodrug", 0.15, 50.0, 0.5, "hepatic", "CYP2C19",
                          "The actual antiplatelet agent. CYP2C19 poor metabolizers → no drug effect."),
            MetaboliteInfo("inactive carboxylic acid", "inactive", 0.85, 0.0, 6.0, "renal", "esterases"),
        ]
    ),
    "tamoxifen": MetaboliteProfile(
        parent_name="tamoxifen",
        metabolites=[
            MetaboliteInfo("endoxifen", "active", 0.05, 100.0, 70.0, "hepatic", "CYP2D6",
                          "Primary active metabolite. CYP2D6 poor metabolizers have reduced efficacy."),
            MetaboliteInfo("4-OH-tamoxifen", "active", 0.07, 30.0, 14.0, "hepatic", "CYP2D6"),
            MetaboliteInfo("N-desmethyl-tamoxifen", "active", 0.60, 0.3, 200.0, "hepatic", "CYP3A4"),
        ]
    ),
    "diazepam": MetaboliteProfile(
        parent_name="diazepam",
        metabolites=[
            MetaboliteInfo("nordiazepam", "active", 0.50, 0.5, 60.0, "hepatic", "CYP3A4",
                          "Long-acting active metabolite. Contributes to prolonged sedation."),
            MetaboliteInfo("temazepam", "active", 0.10, 0.3, 10.0, "hepatic", "CYP3A4"),
            MetaboliteInfo("oxazepam", "active", 0.05, 0.2, 8.0, "renal", "UGT"),
        ]
    ),
    "morphine": MetaboliteProfile(
        parent_name="morphine",
        metabolites=[
            MetaboliteInfo("M6G", "active", 0.10, 6.0, 3.0, "renal", "UGT2B7",
                          "Morphine-6-glucuronide. More potent than morphine. Accumulates in renal failure."),
            MetaboliteInfo("M3G", "inactive", 0.60, 0.0, 3.0, "renal", "UGT2B7",
                          "No analgesic activity. May cause hyperalgesia at high doses."),
        ]
    ),
    "losartan": MetaboliteProfile(
        parent_name="losartan",
        is_prodrug=True,
        metabolites=[
            MetaboliteInfo("EXP3174", "prodrug", 0.14, 10.0, 6.0, "hepatic", "CYP2C9",
                          "Active metabolite 10-40x more potent than losartan."),
        ]
    ),
    "irinotecan": MetaboliteProfile(
        parent_name="irinotecan",
        is_prodrug=True,
        metabolites=[
            MetaboliteInfo("SN-38", "prodrug", 0.04, 1000.0, 10.0, "hepatic", "carboxylesterase",
                          "Active topoisomerase I inhibitor. UGT1A1 poor metabolizers → severe diarrhea."),
            MetaboliteInfo("SN-38G", "inactive", 0.30, 0.0, 12.0, "biliary", "UGT1A1"),
        ]
    ),
    "tramadol": MetaboliteProfile(
        parent_name="tramadol",
        metabolites=[
            MetaboliteInfo("O-desmethyltramadol (M1)", "active", 0.20, 6.0, 7.0, "hepatic", "CYP2D6",
                          "200x more potent µ-opioid agonist. CYP2D6 ultra-rapid → overdose risk."),
            MetaboliteInfo("N-desmethyltramadol (M2)", "inactive", 0.15, 0.1, 6.0, "hepatic", "CYP3A4"),
        ]
    ),
}


def get_metabolite_profile(smiles: str = "", drug_name: str = "") -> MetaboliteProfile:
    """Get metabolite profile for a drug."""
    name_lower = drug_name.lower().strip()

    for known_name, profile in METABOLITE_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            logger.info("Metabolite profile: %s (prodrug=%s, %d metabolites)",
                       drug_name, profile.is_prodrug, len(profile.metabolites))
            return profile

    return MetaboliteProfile(parent_name=drug_name or "Unknown")


# ════════════════════════════════════════════════════════════════
# UNIFIED ADVANCED PHYSIOLOGY ASSESSMENT
# ════════════════════════════════════════════════════════════════

@dataclass
class AdvancedPhysiologyResult:
    """Combined result from all advanced physiology systems."""
    # EHR
    ehr: EHRProfile = field(default_factory=EHRProfile)
    ehr_active: bool = False

    # pH Absorption
    ph_profile: pHAbsorptionProfile = field(default_factory=pHAbsorptionProfile)
    ka_modifier: float = 1.0

    # Food
    food_effect: FoodEffectProfile = field(default_factory=FoodEffectProfile)

    # Transporters
    transporters: TransporterProfile = field(default_factory=TransporterProfile)

    # Metabolites
    metabolites: MetaboliteProfile = field(default_factory=MetaboliteProfile)

    def to_dict(self) -> dict:
        """Convert to API-friendly dict."""
        result = {
            "enterohepatic_recirculation": {
                "active": self.ehr_active,
                "fraction_biliary": self.ehr.fraction_biliary,
                "reabsorption_rate": self.ehr.reabsorption_rate,
                "cycle_delay_h": self.ehr.cycle_delay_h,
                "n_cycles": self.ehr.n_cycles,
            },
            "ph_absorption": {
                "pka": self.ph_profile.pka,
                "is_acid": self.ph_profile.is_acid,
                "stomach_pH": self.ph_profile.get_effective_stomach_pH(),
                "absorption_modifier": self.ka_modifier,
                "ppi_effect": self.ph_profile.ppi_effect,
            },
            "food_effect": {
                "fed_state": self.food_effect.fed_state,
                "absorption_multiplier": self.food_effect.absorption_multiplier,
                "bioavailability_multiplier": self.food_effect.bioavail_multiplier,
                "tmax_delay_h": self.food_effect.tmax_delay_h,
                "mechanism": self.food_effect.mechanism,
            },
            "transporters": {
                "pgp_substrate": self.transporters.pgp_substrate,
                "oatp1b1_substrate": self.transporters.oatp1b1_substrate,
                "oct2_substrate": self.transporters.oct2_substrate,
                "bcrp_substrate": self.transporters.bcrp_substrate,
                "mrp2_substrate": self.transporters.mrp2_substrate,
                "gut_efflux": self.transporters.gut_efflux_factor,
                "liver_uptake": self.transporters.liver_uptake_factor,
                "kidney_secretion": self.transporters.kidney_secretion_factor,
                "brain_efflux": self.transporters.brain_efflux_factor,
                "biliary_efflux": self.transporters.biliary_efflux_factor,
            },
            "metabolites": {
                "parent": self.metabolites.parent_name,
                "is_prodrug": self.metabolites.is_prodrug,
                "metabolites": [
                    {
                        "name": m.name,
                        "type": m.type,
                        "fraction": m.fraction,
                        "activity_ratio": m.activity_ratio,
                        "half_life_h": m.half_life_h,
                        "cyp_enzyme": m.cyp_enzyme,
                        "concern": m.toxicity_concern,
                    }
                    for m in self.metabolites.metabolites
                ],
                "total_active_fraction": self.metabolites.total_active_fraction(),
            },
        }
        return result


def assess_advanced_physiology(
    smiles: str = "",
    drug_name: str = "",
    fed_state: str = "fasted",
    ppi: bool = False,
    antacid: bool = False,
    grapefruit: bool = False,
    bcs_class: int = 1,
) -> AdvancedPhysiologyResult:
    """
    Run all 5 advanced physiology assessments for a drug.

    Returns an AdvancedPhysiologyResult with all modifiers that should
    be applied to the PBPK model.
    """
    result = AdvancedPhysiologyResult()

    # 1. Enterohepatic recirculation
    result.ehr = get_ehr_profile(smiles, drug_name)
    result.ehr_active = result.ehr.fraction_biliary > 0.05

    # 2. pH-dependent absorption
    result.ph_profile = get_ph_absorption_profile(smiles, drug_name, ppi, antacid)
    result.ka_modifier = result.ph_profile.get_absorption_modifier()

    # 3. Food effects
    result.food_effect = get_food_effect(smiles, drug_name, bcs_class, fed_state, grapefruit)

    # 4. Transporters
    result.transporters = get_transporter_profile(smiles, drug_name)

    # 5. Metabolites
    result.metabolites = get_metabolite_profile(smiles, drug_name)

    logger.info(
        "Advanced physiology: EHR=%s, pH_mod=%.2f, food_F=%.1f, "
        "P-gp=%s, OATP=%s, metabolites=%d",
        result.ehr_active, result.ka_modifier,
        result.food_effect.bioavail_multiplier,
        result.transporters.pgp_substrate,
        result.transporters.oatp1b1_substrate,
        len(result.metabolites.metabolites),
    )

    return result
