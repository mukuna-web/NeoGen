"""
Level 5: Clinical Decision Support System
============================================
Real-time drug safety checking for specific patients.

Doctor enters:
  - Patient profile (age, weight, sex, diseases, labs, genetics)
  - Current medications (with doses)
  - New prescription to check

System returns:
  - Drug-drug interactions with ALL current meds
  - Dose adjustments needed for this patient's renal/hepatic function
  - Contraindications based on patient's conditions
  - Monitoring plan
  - Overall safety verdict (SAFE / CAUTION / DANGER / CONTRAINDICATED)

This is what hospitals need for bedside prescribing decisions.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PatientProfile:
    """Complete patient profile for clinical decision support."""
    # Demographics
    age: int = 45
    sex: str = "male"
    weight_kg: float = 70
    height_cm: float = 170

    # Current medications (list of dicts with name, smiles, dose_mg, frequency)
    medications: list = field(default_factory=list)

    # Lab values
    egfr: float = 90          # mL/min/1.73m2 (normal >90)
    creatinine: float = 1.0   # mg/dL (normal 0.7-1.3)
    alt: float = 25           # U/L (normal <40)
    ast: float = 22           # U/L (normal <40)
    albumin: float = 4.0      # g/dL (normal 3.5-5.0)
    bilirubin: float = 0.8    # mg/dL (normal 0.1-1.2)
    inr: float = 1.0          # normal 1.0, warfarin target 2-3
    potassium: float = 4.0    # mEq/L (normal 3.5-5.0)
    sodium: float = 140       # mEq/L (normal 136-145)
    platelets: float = 250    # x10^3/uL (normal 150-400)

    # Diseases
    diseases: list = field(default_factory=list)

    # Genetics
    cyp2d6_status: str = "normal"    # poor/intermediate/normal/ultra-rapid
    cyp2c19_status: str = "normal"
    cyp2c9_status: str = "normal"

    # Allergies
    allergies: list = field(default_factory=list)

    @property
    def ckd_stage(self):
        if self.egfr >= 90: return 0
        if self.egfr >= 60: return 2
        if self.egfr >= 45: return 3
        if self.egfr >= 30: return 3
        if self.egfr >= 15: return 4
        return 5

    @property
    def liver_impairment(self):
        if self.alt > 200 or self.bilirubin > 3: return "severe"
        if self.alt > 80 or self.bilirubin > 2: return "moderate"
        if self.alt > 40 or self.bilirubin > 1.2: return "mild"
        return "none"

    @property
    def bleeding_risk(self):
        risk = 0
        if self.inr > 3: risk += 3
        elif self.inr > 2: risk += 1
        if self.platelets < 50: risk += 3
        elif self.platelets < 100: risk += 1
        if self.age > 75: risk += 1
        if any(d in str(self.diseases) for d in ['ckd', 'cirrhosis']): risk += 1
        return "high" if risk >= 3 else "moderate" if risk >= 2 else "low"


# ════════════════════════════════════════════════════════════
# COMMON MEDICATION DATABASE
# ════════════════════════════════════════════════════════════

MEDICATION_DB = {
    "aspirin": {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "class": "NSAID/Antiplatelet",
                "cyp": ["CYP2C9"], "effects": ["COX-1 inhibitor", "antiplatelet"],
                "renal_adjust": False, "hepatic_adjust": False, "bleeding_risk": True},
    "warfarin": {"smiles": "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O", "class": "Anticoagulant",
                 "cyp": ["CYP2C9", "CYP3A4"], "effects": ["Vitamin K antagonist"],
                 "renal_adjust": False, "hepatic_adjust": True, "bleeding_risk": True,
                 "monitoring": ["INR every 1-4 weeks", "Target INR 2.0-3.0"]},
    "metformin": {"smiles": "CN(C)C(=N)NC(=N)N", "class": "Antidiabetic",
                  "cyp": [], "effects": ["AMPK activator", "reduces hepatic glucose"],
                  "renal_adjust": True, "renal_cutoff_egfr": 30,
                  "renal_warning_egfr": 45, "hepatic_adjust": False,
                  "contraindicated_in": ["ckd_5", "lactic_acidosis"]},
    "metoprolol": {"smiles": "CC(C)NCC(O)COC1=CC=C(C=C1)CCOC", "class": "Beta-blocker",
                   "cyp": ["CYP2D6"], "effects": ["Beta-1 selective blocker"],
                   "renal_adjust": False, "hepatic_adjust": True},
    "amlodipine": {"smiles": "CCOC(=O)C1=C(COCCN)NC(C)=C(C1C1=CC=CC(Cl)=C1)C(=O)OC", "class": "CCB",
                   "cyp": ["CYP3A4"], "effects": ["L-type Ca channel blocker"],
                   "renal_adjust": False, "hepatic_adjust": True},
    "atorvastatin": {"smiles": "CC(C)C1=NC2=CC(F)=CC=C2C(C1/C=C/C(O)CC(O)CC(O)=O)C1=CC=C(F)C=C1", "class": "Statin",
                     "cyp": ["CYP3A4"], "effects": ["HMG-CoA reductase inhibitor"],
                     "renal_adjust": False, "hepatic_adjust": True,
                     "contraindicated_in": ["severe_liver_disease", "pregnancy"]},
    "lisinopril": {"smiles": "NCCCC(NC(CCC1=CC=CC=C1)C(O)=O)C(=O)N1CCCC1C(O)=O", "class": "ACE inhibitor",
                   "cyp": [], "effects": ["ACE inhibitor"],
                   "renal_adjust": True, "renal_warning_egfr": 30,
                   "monitoring": ["Potassium", "Creatinine", "BP"]},
    "omeprazole": {"smiles": "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C", "class": "PPI",
                   "cyp": ["CYP2C19", "CYP3A4"], "effects": ["Proton pump inhibitor"],
                   "renal_adjust": False, "hepatic_adjust": True},
    "ciprofloxacin": {"smiles": "O=C(O)C1=CN(C2CC2)C2=CC(N3CCNCC3)=C(F)C=C2C1=O", "class": "Fluoroquinolone",
                      "cyp": ["CYP1A2"], "effects": ["DNA gyrase inhibitor", "CYP1A2 inhibitor"],
                      "renal_adjust": True, "renal_cutoff_egfr": 15, "renal_warning_egfr": 50,
                      "monitoring": ["Tendon pain", "QTc if on other QT drugs"]},
    "gabapentin": {"smiles": "NCC1(CC(O)=O)CCCCC1", "class": "Anticonvulsant",
                   "cyp": [], "effects": ["Calcium channel alpha-2-delta ligand"],
                   "renal_adjust": True, "renal_warning_egfr": 60},
    "furosemide": {"smiles": "NS(=O)(=O)C1=CC(C(=O)O)=C(NCC2=CC=CO2)C=C1Cl", "class": "Loop diuretic",
                   "cyp": [], "effects": ["Na-K-2Cl cotransporter inhibitor"],
                   "renal_adjust": False, "hepatic_adjust": False,
                   "monitoring": ["Potassium", "Sodium", "Creatinine", "Weight"]},
    "clopidogrel": {"smiles": "COC(=O)C(C1=CC=CC=C1Cl)N1CCC2=CC(=CS2)C1", "class": "Antiplatelet",
                    "cyp": ["CYP2C19"], "effects": ["P2Y12 inhibitor (prodrug)"],
                    "renal_adjust": False, "hepatic_adjust": True, "bleeding_risk": True,
                    "genetic_note": "CYP2C19 poor metabolizers: clopidogrel will NOT work. Use prasugrel or ticagrelor."},
    "fluoxetine": {"smiles": "CNCCC(OC1=CC=C(C(F)(F)F)C=C1)C1=CC=CC=C1", "class": "SSRI",
                   "cyp": ["CYP2D6", "CYP2C19"], "effects": ["Serotonin reuptake inhibitor", "Strong CYP2D6 inhibitor"],
                   "renal_adjust": False, "hepatic_adjust": True},
    "prednisone": {"smiles": "CC12CCC3C(CCC4=CC(=O)C=CC34C)C1CCC2(O)C(=O)CO", "class": "Corticosteroid",
                   "cyp": ["CYP3A4"], "effects": ["Glucocorticoid receptor agonist", "Immunosuppressant"],
                   "renal_adjust": False, "hepatic_adjust": False,
                   "monitoring": ["Blood glucose", "Blood pressure", "Bone density if chronic"]},
}

# ════════════════════════════════════════════════════════════
# DRUG-DRUG INTERACTION RULES
# ════════════════════════════════════════════════════════════

DDI_RULES = [
    {"drug_a": "warfarin", "drug_b": "aspirin", "severity": "HIGH",
     "mechanism": "Dual anticoagulation + antiplatelet. Warfarin (vitamin K antagonist) + aspirin (COX-1 inhibitor) dramatically increases bleeding risk.",
     "recommendation": "Avoid combination unless compelling indication (e.g., mechanical heart valve + recent stent). If used, target INR 2.0-2.5 and add PPI for GI protection."},

    {"drug_a": "warfarin", "drug_b": "ciprofloxacin", "severity": "HIGH",
     "mechanism": "Ciprofloxacin inhibits CYP1A2 and kills vitamin K-producing gut bacteria. Both effects increase warfarin's anticoagulant effect. INR can rise 2-3x within 3-5 days.",
     "recommendation": "If cipro is essential, reduce warfarin dose by 25-33% and check INR at day 3 and 7. Consider levofloxacin as alternative (less interaction)."},

    {"drug_a": "warfarin", "drug_b": "omeprazole", "severity": "LOW",
     "mechanism": "Omeprazole weakly inhibits CYP2C19 which handles R-warfarin (less potent enantiomer). Clinically minor but can increase INR slightly.",
     "recommendation": "Monitor INR. Consider pantoprazole as alternative (no CYP interaction)."},

    {"drug_a": "metformin", "drug_b": "ciprofloxacin", "severity": "MODERATE",
     "mechanism": "Ciprofloxacin can cause dysglycemia (both hypo and hyperglycemia). Combined with metformin, unpredictable glucose changes.",
     "recommendation": "Monitor blood glucose more frequently during cipro course. Educate patient about hypoglycemia symptoms."},

    {"drug_a": "atorvastatin", "drug_b": "ciprofloxacin", "severity": "LOW",
     "mechanism": "Ciprofloxacin is a weak CYP3A4 inhibitor. May slightly increase atorvastatin levels. Rhabdomyolysis risk is very low.",
     "recommendation": "No dose adjustment needed for short cipro course. Watch for muscle pain."},

    {"drug_a": "metoprolol", "drug_b": "fluoxetine", "severity": "HIGH",
     "mechanism": "Fluoxetine is a strong CYP2D6 inhibitor. Metoprolol is a CYP2D6 substrate. Fluoxetine will increase metoprolol levels 4-6x, causing severe bradycardia and hypotension.",
     "recommendation": "Avoid combination. Switch metoprolol to atenolol (not CYP2D6 metabolized) or switch fluoxetine to sertraline (weaker CYP2D6 inhibition)."},

    {"drug_a": "clopidogrel", "drug_b": "omeprazole", "severity": "HIGH",
     "mechanism": "Omeprazole inhibits CYP2C19 which activates clopidogrel (prodrug). Omeprazole reduces clopidogrel's antiplatelet effect by 40-50%. Increased risk of stent thrombosis.",
     "recommendation": "Use pantoprazole instead of omeprazole (no CYP2C19 interaction). FDA black box warning against this combination."},

    {"drug_a": "clopidogrel", "drug_b": "fluoxetine", "severity": "MODERATE",
     "mechanism": "Fluoxetine inhibits CYP2C19, reducing clopidogrel activation. Less severe than omeprazole but still clinically relevant.",
     "recommendation": "Consider alternative SSRI (sertraline has minimal CYP2C19 effect) or alternative antiplatelet (prasugrel doesn't need CYP2C19)."},

    {"drug_a": "lisinopril", "drug_b": "furosemide", "severity": "MODERATE",
     "mechanism": "Both lower blood pressure. First-dose ACE inhibitor + diuretic can cause severe hypotension, especially if dehydrated.",
     "recommendation": "If starting lisinopril in patient on furosemide, hold diuretic for 24-48h or start lisinopril at half dose. Monitor BP closely."},

    {"drug_a": "aspirin", "drug_b": "clopidogrel", "severity": "MODERATE",
     "mechanism": "Dual antiplatelet therapy (DAPT). Standard after coronary stents but significantly increases bleeding risk (2-3x vs aspirin alone).",
     "recommendation": "DAPT is appropriate for 6-12 months post-stent. Add PPI for GI protection. Monitor for signs of bleeding."},

    {"drug_a": "gabapentin", "drug_b": "furosemide", "severity": "LOW",
     "mechanism": "Furosemide can reduce gabapentin absorption by 20% (unknown mechanism). May need higher gabapentin dose.",
     "recommendation": "Take gabapentin at least 2 hours after furosemide. Monitor pain/seizure control."},

    {"drug_a": "prednisone", "drug_b": "metformin", "severity": "MODERATE",
     "mechanism": "Prednisone increases blood glucose by promoting gluconeogenesis and insulin resistance. Opposes metformin's effect.",
     "recommendation": "Monitor blood glucose 4x daily during prednisone course. May need to temporarily increase metformin dose or add insulin."},

    {"drug_a": "prednisone", "drug_b": "warfarin", "severity": "MODERATE",
     "mechanism": "Corticosteroids can both increase and decrease warfarin effect (variable). High-dose prednisone increases bleeding risk independently.",
     "recommendation": "Check INR within 3-5 days of starting/stopping prednisone. Adjust warfarin accordingly."},

    {"drug_a": "prednisone", "drug_b": "aspirin", "severity": "HIGH",
     "mechanism": "Both damage GI mucosa independently. Combined GI bleeding risk is 4-15x higher than either alone.",
     "recommendation": "Avoid combination if possible. If essential, MUST add PPI (omeprazole/pantoprazole) for GI protection."},
]


def check_prescription(patient: PatientProfile, new_drug_name: str, new_drug_dose: float) -> dict:
    """
    Check if a new prescription is safe for this specific patient.

    Returns comprehensive safety assessment.
    """
    new_drug_name = new_drug_name.lower().strip()
    new_drug = MEDICATION_DB.get(new_drug_name, {})

    result = {
        "patient_summary": {
            "age": patient.age,
            "sex": patient.sex,
            "weight": patient.weight_kg,
            "egfr": patient.egfr,
            "ckd_stage": patient.ckd_stage,
            "liver": patient.liver_impairment,
            "bleeding_risk": patient.bleeding_risk,
            "n_current_meds": len(patient.medications),
            "current_meds": [m.get("name", "") for m in patient.medications],
            "diseases": patient.diseases,
            "cyp2d6": patient.cyp2d6_status,
            "cyp2c19": patient.cyp2c19_status,
        },
        "new_drug": {
            "name": new_drug_name,
            "dose_mg": new_drug_dose,
            "class": new_drug.get("class", "Unknown"),
            "smiles": new_drug.get("smiles", ""),
        },
        "alerts": [],
        "dose_adjustments": [],
        "contraindications": [],
        "interactions": [],
        "monitoring": [],
        "genetic_warnings": [],
        "overall_verdict": "SAFE",
    }

    # ── Check drug-drug interactions ──
    for med in patient.medications:
        med_name = med.get("name", "").lower().strip()
        for rule in DDI_RULES:
            if (rule["drug_a"] == new_drug_name and rule["drug_b"] == med_name) or \
               (rule["drug_b"] == new_drug_name and rule["drug_a"] == med_name):
                interaction = {
                    "between": f"{new_drug_name} + {med_name}",
                    "severity": rule["severity"],
                    "mechanism": rule["mechanism"],
                    "recommendation": rule["recommendation"],
                }
                result["interactions"].append(interaction)
                if rule["severity"] == "HIGH":
                    result["overall_verdict"] = "DANGER"
                elif rule["severity"] == "MODERATE" and result["overall_verdict"] != "DANGER":
                    result["overall_verdict"] = "CAUTION"

    # ── Check renal dose adjustment ──
    if new_drug.get("renal_adjust"):
        cutoff = new_drug.get("renal_cutoff_egfr", 30)
        warning = new_drug.get("renal_warning_egfr", 45)

        if patient.egfr < cutoff:
            result["contraindications"].append({
                "reason": f"{new_drug_name} is contraindicated with eGFR <{cutoff} mL/min (patient eGFR: {patient.egfr})",
                "action": f"Do NOT prescribe {new_drug_name}. Consider alternative.",
            })
            result["overall_verdict"] = "CONTRAINDICATED"
        elif patient.egfr < warning:
            result["dose_adjustments"].append({
                "reason": f"Reduced renal function (eGFR {patient.egfr})",
                "adjustment": f"Reduce {new_drug_name} dose by 50% or extend dosing interval",
                "monitor": "Recheck creatinine and eGFR at day 5",
            })
            if result["overall_verdict"] == "SAFE":
                result["overall_verdict"] = "CAUTION"

    # ── Check hepatic dose adjustment ──
    if new_drug.get("hepatic_adjust") and patient.liver_impairment != "none":
        if patient.liver_impairment == "severe":
            result["dose_adjustments"].append({
                "reason": f"Severe hepatic impairment (ALT {patient.alt}, bilirubin {patient.bilirubin})",
                "adjustment": f"Reduce {new_drug_name} dose by 50-75% or avoid",
                "monitor": "LFTs weekly",
            })
            if result["overall_verdict"] in ["SAFE", "CAUTION"]:
                result["overall_verdict"] = "DANGER"
        else:
            result["dose_adjustments"].append({
                "reason": f"{patient.liver_impairment.title()} hepatic impairment (ALT {patient.alt})",
                "adjustment": f"Consider reducing {new_drug_name} dose by 25-50%",
                "monitor": "LFTs at 2 weeks",
            })
            if result["overall_verdict"] == "SAFE":
                result["overall_verdict"] = "CAUTION"

    # ── Check contraindications ──
    if new_drug.get("contraindicated_in"):
        for condition in new_drug["contraindicated_in"]:
            if condition in [d.lower() for d in patient.diseases]:
                result["contraindications"].append({
                    "reason": f"{new_drug_name} is contraindicated in {condition}",
                    "action": "Do NOT prescribe. Choose alternative.",
                })
                result["overall_verdict"] = "CONTRAINDICATED"

    # ── Check bleeding risk ──
    if new_drug.get("bleeding_risk") and patient.bleeding_risk != "low":
        existing_bleeders = [m for m in patient.medications
                            if MEDICATION_DB.get(m.get("name","").lower(), {}).get("bleeding_risk")]
        if existing_bleeders:
            result["alerts"].append({
                "type": "BLEEDING",
                "message": f"Adding {new_drug_name} (bleeding risk) to patient already on {', '.join(m['name'] for m in existing_bleeders)}. Patient baseline bleeding risk: {patient.bleeding_risk}.",
                "action": "Consider GI protection (PPI). Monitor for signs of bleeding. Check CBC and INR.",
            })
            if result["overall_verdict"] == "SAFE":
                result["overall_verdict"] = "CAUTION"

    # ── Check genetic warnings ──
    if new_drug_name == "clopidogrel" and patient.cyp2c19_status == "poor":
        result["genetic_warnings"].append({
            "gene": "CYP2C19",
            "status": "Poor Metabolizer",
            "impact": "Clopidogrel is a PRODRUG that requires CYP2C19 to activate. This patient CANNOT activate clopidogrel. The drug will have ZERO antiplatelet effect.",
            "recommendation": "Use prasugrel or ticagrelor instead. DO NOT use clopidogrel.",
        })
        result["overall_verdict"] = "CONTRAINDICATED"

    if new_drug_name == "codeine" and patient.cyp2d6_status == "poor":
        result["genetic_warnings"].append({
            "gene": "CYP2D6",
            "status": "Poor Metabolizer",
            "impact": "Codeine requires CYP2D6 to convert to morphine (active). This patient cannot convert codeine. No analgesic effect.",
            "recommendation": "Use morphine, oxycodone, or hydromorphone directly.",
        })
        result["overall_verdict"] = "CONTRAINDICATED"

    if new_drug_name == "codeine" and patient.cyp2d6_status == "ultra-rapid":
        result["genetic_warnings"].append({
            "gene": "CYP2D6",
            "status": "Ultra-Rapid Metabolizer",
            "impact": "This patient converts codeine to morphine 4-5x faster than normal. RISK OF RESPIRATORY DEPRESSION AND DEATH, especially in children.",
            "recommendation": "DO NOT use codeine. Use non-CYP2D6 opioids (morphine, oxycodone) at reduced doses.",
        })
        result["overall_verdict"] = "CONTRAINDICATED"

    if new_drug_name == "metoprolol" and patient.cyp2d6_status in ["poor", "intermediate"]:
        result["genetic_warnings"].append({
            "gene": "CYP2D6",
            "status": patient.cyp2d6_status.title() + " Metabolizer",
            "impact": f"Metoprolol levels will be {3 if patient.cyp2d6_status == 'poor' else 1.5}x higher than normal. Risk of excessive bradycardia and hypotension.",
            "recommendation": f"Reduce metoprolol dose by {50 if patient.cyp2d6_status == 'poor' else 25}% or use atenolol (not CYP2D6 metabolized).",
        })
        if result["overall_verdict"] == "SAFE":
            result["overall_verdict"] = "CAUTION"

    # ── Standard monitoring ──
    if new_drug.get("monitoring"):
        result["monitoring"].extend(new_drug["monitoring"])

    # ── Check genetics for EXISTING medications too ──
    for med in patient.medications:
        med_name = med.get("name", "").lower().strip()
        if med_name == "metoprolol" and patient.cyp2d6_status in ["poor", "intermediate"]:
            result["genetic_warnings"].append({
                "gene": "CYP2D6",
                "status": patient.cyp2d6_status.title() + " Metabolizer",
                "impact": f"Patient is on metoprolol (CYP2D6 substrate) and is a CYP2D6 {patient.cyp2d6_status} metabolizer. Metoprolol levels may be {3 if patient.cyp2d6_status == 'poor' else 1.5}x higher than expected. Risk of bradycardia.",
                "recommendation": f"Consider switching metoprolol to atenolol (not CYP2D6 metabolized) or reduce dose by {50 if patient.cyp2d6_status == 'poor' else 25}%.",
            })
        if med_name == "clopidogrel" and patient.cyp2c19_status == "poor":
            result["genetic_warnings"].append({
                "gene": "CYP2C19",
                "status": "Poor Metabolizer",
                "impact": "Patient is on clopidogrel but is a CYP2C19 poor metabolizer. Clopidogrel has NO antiplatelet effect in this patient.",
                "recommendation": "URGENT: Switch clopidogrel to prasugrel or ticagrelor immediately.",
            })

    # ── Age-specific alerts ──
    if patient.age > 75:
        result["alerts"].append({
            "type": "ELDERLY",
            "message": f"Patient is {patient.age} years old. Start {new_drug_name} at lowest recommended dose. Increased sensitivity to side effects.",
            "action": "Consider 50% starting dose. Monitor closely for 2 weeks.",
        })

    # ── Allergy check ──
    drug_class = new_drug.get("class", "").lower()
    for allergy in patient.allergies:
        if allergy.lower() in new_drug_name or allergy.lower() in drug_class:
            result["contraindications"].append({
                "reason": f"Patient has documented allergy to {allergy}",
                "action": "DO NOT prescribe. Use alternative class.",
            })
            result["overall_verdict"] = "CONTRAINDICATED"

    return result
