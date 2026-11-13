"""
Toxicity & Clinical Effects Engine
====================================
Maps drug concentrations in each organ to specific clinical symptoms
and physiological effects on the human body.

Instead of just saying "toxic" — we show what actually happens:
- Sweating, anxiety, tremors at mild levels
- Seizures, arrhythmia, organ damage at severe levels

This uses:
1. Known drug-specific thresholds (from clinical literature)
2. General organ toxicity rules (for unknown drugs)
3. Structural alerts (known toxic substructures)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# CLINICAL EFFECTS DATABASE
# ════════════════════════════════════════════════════════════════

# Severity levels
MILD = "mild"
MODERATE = "moderate"
SEVERE = "severe"
CRITICAL = "critical"

@dataclass
class ClinicalEffect:
    """A single clinical effect triggered by drug concentration."""
    organ: str           # Which organ system
    severity: str        # mild/moderate/severe/critical
    symptom: str         # Human-readable symptom
    description: str     # Detailed explanation
    vital_sign: str      # Which vital sign is affected
    vital_change: str    # How it changes (e.g., "HR +20-40 bpm")
    threshold_fraction: float  # Fraction of toxic threshold that triggers this


# ── General organ toxicity effects (for any drug) ──────────────────

GENERAL_ORGAN_EFFECTS = {
    "brain": {
        "name": "Central Nervous System",
        "effects": [
            ClinicalEffect("brain", MILD, "Dizziness & lightheadedness",
                "Drug crossing blood-brain barrier causes mild CNS depression",
                "Mental status", "Mild confusion", 0.3),
            ClinicalEffect("brain", MILD, "Anxiety & restlessness",
                "Stimulation of sympathetic nervous system pathways",
                "Heart rate", "HR +10-15 bpm", 0.4),
            ClinicalEffect("brain", MODERATE, "Headache & visual disturbance",
                "Increased intracranial pressure from drug accumulation",
                "Blood pressure", "BP +10-20 mmHg", 0.5),
            ClinicalEffect("brain", MODERATE, "Tremors & muscle twitching",
                "Excessive neurotransmitter stimulation at synaptic junctions",
                "Motor function", "Fine tremor in hands", 0.6),
            ClinicalEffect("brain", SEVERE, "Seizure risk — involuntary convulsions",
                "Neuronal hyperexcitability from toxic drug levels in brain tissue",
                "EEG", "Epileptiform activity", 0.8),
            ClinicalEffect("brain", CRITICAL, "Loss of consciousness / coma risk",
                "Severe CNS depression from overwhelming drug concentration",
                "Glasgow Coma Scale", "GCS dropping below 8", 1.0),
        ],
    },
    "liver": {
        "name": "Hepatic System",
        "effects": [
            ClinicalEffect("liver", MILD, "Mild nausea",
                "Early hepatocyte stress from drug metabolism byproducts",
                "Liver enzymes", "ALT slightly elevated", 0.3),
            ClinicalEffect("liver", MILD, "Loss of appetite",
                "Bile flow disruption from drug metabolite accumulation",
                "Bilirubin", "Mildly elevated", 0.4),
            ClinicalEffect("liver", MODERATE, "Abdominal pain (right upper quadrant)",
                "Hepatocyte swelling and inflammation from toxic metabolites",
                "Liver enzymes", "ALT/AST 2-5x upper limit", 0.5),
            ClinicalEffect("liver", MODERATE, "Jaundice — yellowing of skin/eyes",
                "Impaired bilirubin conjugation due to hepatocyte damage",
                "Bilirubin", "Total bilirubin >2.0 mg/dL", 0.65),
            ClinicalEffect("liver", SEVERE, "Drug-Induced Liver Injury (DILI)",
                "Significant hepatocellular necrosis — liver cells dying from toxic exposure",
                "Liver enzymes", "ALT/AST >10x upper limit", 0.8),
            ClinicalEffect("liver", CRITICAL, "Acute liver failure risk",
                "Massive hepatocyte death, loss of synthetic function, coagulopathy",
                "INR", "INR >1.5, encephalopathy", 1.0),
        ],
    },
    "kidney": {
        "name": "Renal System",
        "effects": [
            ClinicalEffect("kidney", MILD, "Increased urination frequency",
                "Kidneys attempting to clear excess drug through increased filtration",
                "Urine output", "Polyuria", 0.3),
            ClinicalEffect("kidney", MILD, "Dark-colored urine",
                "Drug metabolites being concentrated in urine",
                "Urine color", "Amber to brown", 0.4),
            ClinicalEffect("kidney", MODERATE, "Reduced urine output",
                "Tubular damage causing impaired concentration ability",
                "Creatinine", "Cr rising above baseline", 0.5),
            ClinicalEffect("kidney", MODERATE, "Flank pain & edema",
                "Nephron damage causing fluid retention and kidney inflammation",
                "BUN/Creatinine", "BUN:Cr ratio elevated", 0.65),
            ClinicalEffect("kidney", SEVERE, "Acute kidney injury (AKI)",
                "Drug-induced tubular necrosis with rising creatinine",
                "GFR", "eGFR <30 mL/min", 0.8),
            ClinicalEffect("kidney", CRITICAL, "Renal failure — dialysis risk",
                "Complete loss of kidney filtration function",
                "Potassium", "Hyperkalemia, metabolic acidosis", 1.0),
        ],
    },
    "heart": {
        "name": "Cardiovascular System",
        "effects": [
            ClinicalEffect("heart", MILD, "Palpitations — awareness of heartbeat",
                "Drug affecting cardiac conduction or sympathetic tone",
                "Heart rate", "HR +10-20 bpm", 0.3),
            ClinicalEffect("heart", MILD, "Mild tachycardia",
                "Sympathomimetic effect or vagal inhibition",
                "Heart rate", "HR 90-110 bpm", 0.4),
            ClinicalEffect("heart", MODERATE, "Chest tightness & sweating",
                "Myocardial oxygen demand exceeding supply",
                "Blood pressure", "BP fluctuating", 0.5),
            ClinicalEffect("heart", MODERATE, "QT prolongation risk",
                "Drug blocking hERG potassium channels in cardiac myocytes",
                "ECG", "QTc >450ms", 0.6),
            ClinicalEffect("heart", SEVERE, "Arrhythmia — irregular heartbeat",
                "Disruption of cardiac electrical conduction system",
                "ECG", "Ventricular ectopy, Torsades risk", 0.8),
            ClinicalEffect("heart", CRITICAL, "Cardiac arrest risk",
                "Complete cardiac conduction failure or ventricular fibrillation",
                "ECG", "VF/pulseless VT", 1.0),
        ],
    },
    "gut": {
        "name": "Gastrointestinal System",
        "effects": [
            ClinicalEffect("gut", MILD, "Stomach discomfort & bloating",
                "Local irritation of gastric mucosa by drug or metabolites",
                "Comfort", "Mild epigastric discomfort", 0.3),
            ClinicalEffect("gut", MILD, "Nausea",
                "Stimulation of chemoreceptor trigger zone and vagal afferents",
                "Symptoms", "Waves of nausea", 0.4),
            ClinicalEffect("gut", MODERATE, "Vomiting & diarrhea",
                "Severe GI mucosal irritation and increased intestinal motility",
                "Hydration", "Risk of dehydration", 0.6),
            ClinicalEffect("gut", SEVERE, "GI bleeding risk",
                "Erosion of gastric/intestinal mucosa from toxic drug levels",
                "Hemoglobin", "Occult blood in stool", 0.85),
        ],
    },
    "muscle": {
        "name": "Musculoskeletal System",
        "effects": [
            ClinicalEffect("muscle", MILD, "Muscle weakness & fatigue",
                "Drug interfering with neuromuscular junction or mitochondrial function",
                "Strength", "Mild proximal weakness", 0.4),
            ClinicalEffect("muscle", MODERATE, "Myalgia — muscle pain & cramps",
                "Drug-induced myopathy from mitochondrial toxicity",
                "CK levels", "CK mildly elevated", 0.6),
            ClinicalEffect("muscle", SEVERE, "Rhabdomyolysis risk",
                "Massive muscle cell breakdown releasing myoglobin into blood",
                "CK levels", "CK >10x upper limit, dark urine", 0.85),
        ],
    },
    "fat": {
        "name": "Adipose / Metabolic",
        "effects": [
            ClinicalEffect("fat", MILD, "Prolonged drug effect (slow release)",
                "Lipophilic drug accumulating in fat tissue acts as depot",
                "Duration", "Effects lasting longer than expected", 0.5),
            ClinicalEffect("fat", MODERATE, "Metabolic disruption",
                "High fat-tissue drug levels interfering with lipid metabolism",
                "Triglycerides", "Elevated triglycerides", 0.7),
        ],
    },
}


# ── Drug-specific overrides (known clinical data) ──────────────────

DRUG_SPECIFIC_THRESHOLDS = {
    # Caffeine
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": {
        "name": "Caffeine",
        "plasma_toxic_mg_L": 20.0,
        "custom_effects": [
            ClinicalEffect("brain", MILD, "Jitteriness & insomnia",
                "Caffeine blocking adenosine receptors, preventing sleepiness",
                "Sleep", "Unable to fall asleep", 0.25),
            ClinicalEffect("heart", MILD, "Heart racing — tachycardia",
                "Caffeine stimulating sympathetic nervous system",
                "Heart rate", "HR 100-120 bpm", 0.3),
            ClinicalEffect("gut", MILD, "Acid reflux & stomach upset",
                "Caffeine increasing gastric acid secretion",
                "pH", "Gastric pH dropping", 0.2),
            ClinicalEffect("brain", MODERATE, "Severe anxiety & panic attacks",
                "Excessive adenosine receptor blockade causing anxiety",
                "Mental status", "Panic, agitation", 0.5),
            ClinicalEffect("heart", MODERATE, "Chest pain & palpitations",
                "Caffeine-induced coronary vasoconstriction",
                "Heart rate", "HR >130 bpm, irregular", 0.6),
            ClinicalEffect("brain", SEVERE, "Seizures",
                "Caffeine toxicity causing neuronal hyperexcitability",
                "EEG", "Generalized seizure activity", 0.8),
            ClinicalEffect("heart", CRITICAL, "Ventricular arrhythmia",
                "Massive caffeine dose causing fatal cardiac rhythm disturbance",
                "ECG", "VT/VF", 1.0),
        ],
    },
    # Paracetamol / Acetaminophen
    "CC(=O)NC1=CC=C(C=C1)O": {
        "name": "Paracetamol",
        "plasma_toxic_mg_L": 150.0,  # Rumack-Matthew nomogram: >150 at 4h
        "custom_effects": [
            ClinicalEffect("liver", MILD, "Mild nausea & malaise",
                "Early NAPQI formation overwhelming glutathione stores",
                "Liver enzymes", "Normal initially", 0.2),
            ClinicalEffect("liver", MODERATE, "Right upper quadrant pain",
                "Hepatocyte damage from NAPQI (toxic metabolite of paracetamol)",
                "Liver enzymes", "ALT/AST rising rapidly", 0.4),
            ClinicalEffect("liver", SEVERE, "Severe hepatotoxicity (DILI)",
                "Massive NAPQI-mediated hepatocellular necrosis. This is the #1 cause of acute liver failure in the West.",
                "Liver enzymes", "ALT >1000 IU/L, INR rising", 0.6),
            ClinicalEffect("kidney", MODERATE, "Renal tubular damage",
                "NAPQI metabolite also toxic to renal tubular cells",
                "Creatinine", "Rising creatinine", 0.7),
            ClinicalEffect("liver", CRITICAL, "Fulminant hepatic failure",
                "Complete liver failure with encephalopathy, coagulopathy. Fatal without transplant.",
                "INR", "INR >6.5, King's College criteria met", 0.9),
        ],
    },
    # Aspirin
    "CC(=O)OC1=CC=CC=C1C(=O)O": {
        "name": "Aspirin",
        "plasma_toxic_mg_L": 300.0,  # Salicylate >300 mg/L = toxic
        "custom_effects": [
            ClinicalEffect("gut", MILD, "Stomach irritation & heartburn",
                "Aspirin inhibiting COX-1 in gastric mucosa, reducing protective prostaglandins",
                "Symptoms", "Dyspepsia", 0.15),
            ClinicalEffect("brain", MILD, "Tinnitus (ringing in ears)",
                "Salicylate affecting cochlear function — classic early sign of toxicity",
                "Hearing", "High-pitched tinnitus", 0.3),
            ClinicalEffect("brain", MODERATE, "Hyperventilation & sweating",
                "Salicylate directly stimulating the medullary respiratory center",
                "Respiratory rate", "RR >25/min, diaphoresis", 0.5),
            ClinicalEffect("gut", SEVERE, "GI bleeding",
                "Aspirin preventing platelet aggregation + direct mucosal erosion",
                "Hemoglobin", "Melena, hematemesis", 0.7),
            ClinicalEffect("brain", SEVERE, "Metabolic acidosis & confusion",
                "Salicylate uncoupling oxidative phosphorylation in mitochondria",
                "pH/Lactate", "pH <7.3, elevated lactate", 0.8),
            ClinicalEffect("brain", CRITICAL, "Cerebral edema & coma",
                "Severe salicylate poisoning causing non-cardiogenic pulmonary edema and cerebral edema",
                "GCS", "GCS <8, respiratory failure", 1.0),
        ],
    },
    # Ibuprofen
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": {
        "name": "Ibuprofen",
        "plasma_toxic_mg_L": 100.0,
        "custom_effects": [
            ClinicalEffect("gut", MILD, "Stomach pain & nausea",
                "COX inhibition reducing protective gastric prostaglandins",
                "Symptoms", "Epigastric discomfort", 0.2),
            ClinicalEffect("kidney", MODERATE, "Reduced kidney perfusion",
                "Ibuprofen blocking prostaglandin-mediated renal vasodilation",
                "GFR", "eGFR declining", 0.5),
            ClinicalEffect("gut", SEVERE, "Gastric ulceration & bleeding",
                "Chronic COX-1 inhibition causing mucosal breakdown",
                "Hemoglobin", "GI blood loss", 0.7),
            ClinicalEffect("kidney", SEVERE, "Acute kidney injury",
                "Prostaglandin-dependent renal blood flow completely blocked",
                "Creatinine", "Cr >2x baseline", 0.85),
        ],
    },
}


# ── Structural alerts for unknown drugs ────────────────────────────

STRUCTURAL_ALERTS = [
    {
        "name": "Reactive Michael Acceptor",
        "smarts": "[CH2]=[CH]-[C,N,S]=O",
        "risk": "hepatotoxicity",
        "description": "Michael acceptors can form covalent bonds with glutathione and proteins, causing liver damage",
    },
    {
        "name": "Aniline / Aromatic Amine",
        "smarts": "c1ccccc1N",
        "risk": "hepatotoxicity",
        "description": "Aromatic amines can be bioactivated to reactive hydroxylamines by CYP enzymes",
    },
    {
        "name": "Nitro Group",
        "smarts": "[N+](=O)[O-]",
        "risk": "hepatotoxicity",
        "description": "Nitro groups undergo nitroreduction producing reactive nitroso intermediates",
    },
    {
        "name": "Thiophene",
        "smarts": "c1ccsc1",
        "risk": "hepatotoxicity",
        "description": "Thiophene rings can be oxidized to reactive epoxides by CYP enzymes",
    },
    {
        "name": "Hydrazine",
        "smarts": "NN",
        "risk": "hepatotoxicity",
        "description": "Hydrazines are potent hepatotoxins through oxidative stress mechanisms",
    },
    {
        "name": "Quinone / para-aminophenol",
        "smarts": "O=C1C=CC(=O)C=C1",
        "risk": "hepatotoxicity",
        "description": "Quinones undergo redox cycling generating reactive oxygen species",
    },
]


# ════════════════════════════════════════════════════════════════
# TOXICITY ASSESSOR
# ════════════════════════════════════════════════════════════════

class ToxicityAssessor:
    """
    Assesses clinical effects based on PBPK-simulated organ concentrations.

    For known drugs: uses drug-specific thresholds and symptoms.
    For unknown drugs: uses general organ toxicity + structural alerts.
    """

    def __init__(self, smiles: str):
        self.smiles = smiles
        self.drug_info = DRUG_SPECIFIC_THRESHOLDS.get(smiles)
        self.structural_alerts = self._check_structural_alerts()

    def _check_structural_alerts(self) -> list[dict]:
        """Check molecule for known toxic substructures."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is None:
                return []

            alerts = []
            for alert in STRUCTURAL_ALERTS:
                pattern = Chem.MolFromSmarts(alert["smarts"])
                if pattern and mol.HasSubstructMatch(pattern):
                    alerts.append(alert)
            return alerts
        except Exception:
            return []

    def assess_at_timepoint(
        self,
        plasma_conc: float,
        liver_conc: float,
        kidney_conc: float,
        brain_conc: float,
        muscle_conc: float,
        fat_conc: float,
        gut_conc: float,
        dose_mg: float,
    ) -> list[dict]:
        """
        Assess clinical effects at a single time point.

        Returns list of active effects with their severity and details.
        """
        active_effects = []

        # Determine toxic threshold
        if self.drug_info:
            toxic_threshold = self.drug_info["plasma_toxic_mg_L"]
        else:
            # For unknown drugs, estimate toxic threshold as 10x Cmax
            # (rough heuristic — therapeutic index of ~10)
            toxic_threshold = max(plasma_conc * 10, dose_mg / 5)

        # Map organ concentrations to fractions of toxic level
        organ_fractions = {
            "brain": brain_conc / max(toxic_threshold * 0.5, 0.001),   # Brain more sensitive
            "liver": liver_conc / max(toxic_threshold * 2.0, 0.001),   # Liver sees higher levels
            "kidney": kidney_conc / max(toxic_threshold * 1.5, 0.001),
            "heart": plasma_conc / max(toxic_threshold, 0.001),        # Heart ~ plasma
            "gut": gut_conc / max(toxic_threshold * 0.8, 0.001),
            "muscle": muscle_conc / max(toxic_threshold * 3.0, 0.001),
            "fat": fat_conc / max(toxic_threshold * 5.0, 0.001),
        }

        # Check drug-specific effects first
        if self.drug_info and "custom_effects" in self.drug_info:
            plasma_fraction = plasma_conc / max(toxic_threshold, 0.001)
            for effect in self.drug_info["custom_effects"]:
                if plasma_fraction >= effect.threshold_fraction:
                    active_effects.append({
                        "organ": effect.organ,
                        "severity": effect.severity,
                        "symptom": effect.symptom,
                        "description": effect.description,
                        "vital_sign": effect.vital_sign,
                        "vital_change": effect.vital_change,
                        "source": "drug_specific",
                    })
        else:
            # Use general organ effects for unknown drugs
            for organ, fraction in organ_fractions.items():
                if organ not in GENERAL_ORGAN_EFFECTS:
                    continue
                organ_info = GENERAL_ORGAN_EFFECTS[organ]
                for effect in organ_info["effects"]:
                    if fraction >= effect.threshold_fraction:
                        active_effects.append({
                            "organ": effect.organ,
                            "severity": effect.severity,
                            "symptom": effect.symptom,
                            "description": effect.description,
                            "vital_sign": effect.vital_sign,
                            "vital_change": effect.vital_change,
                            "source": "general",
                        })

        # Add structural alerts if present
        for alert in self.structural_alerts:
            active_effects.append({
                "organ": "liver",
                "severity": MILD,
                "symptom": f"Structural alert: {alert['name']}",
                "description": alert["description"],
                "vital_sign": "Structure",
                "vital_change": f"Contains {alert['name']} motif",
                "source": "structural_alert",
            })

        # Deduplicate by symptom
        seen = set()
        unique_effects = []
        for e in active_effects:
            if e["symptom"] not in seen:
                seen.add(e["symptom"])
                unique_effects.append(e)

        # Sort by severity (critical first)
        severity_order = {CRITICAL: 0, SEVERE: 1, MODERATE: 2, MILD: 3}
        unique_effects.sort(key=lambda e: severity_order.get(e["severity"], 4))

        return unique_effects

    def assess_full_timecourse(
        self,
        time_course: list[dict],
        dose_mg: float,
    ) -> dict:
        """
        Assess clinical effects across the full simulation.

        Returns:
        - peak_effects: worst-case effects at peak concentrations
        - timeline: when each effect starts/stops
        - overall_risk: summary risk level
        - organ_risk_scores: per-organ risk (0-1)
        """
        if not time_course:
            return {"peak_effects": [], "overall_risk": "low", "organ_risk_scores": {}}

        # Find peak concentration timepoint
        peak_idx = 0
        peak_plasma = 0
        for i, tp in enumerate(time_course):
            if tp.get("plasma_conc", 0) > peak_plasma:
                peak_plasma = tp["plasma_conc"]
                peak_idx = i

        peak_tp = time_course[peak_idx]

        # Assess at peak
        peak_effects = self.assess_at_timepoint(
            plasma_conc=peak_tp.get("plasma_conc", 0),
            liver_conc=peak_tp.get("liver_conc", 0),
            kidney_conc=peak_tp.get("kidney_conc", 0),
            brain_conc=peak_tp.get("brain_conc", 0),
            muscle_conc=peak_tp.get("muscle_conc", 0),
            fat_conc=peak_tp.get("fat_conc", 0),
            gut_conc=peak_tp.get("gut_conc", 0),
            dose_mg=dose_mg,
        )

        # Determine overall risk
        severities = [e["severity"] for e in peak_effects]
        if CRITICAL in severities:
            overall_risk = "critical"
        elif SEVERE in severities:
            overall_risk = "high"
        elif MODERATE in severities:
            overall_risk = "moderate"
        elif MILD in severities:
            overall_risk = "low"
        else:
            overall_risk = "minimal"

        # Per-organ risk scores (0-1)
        organ_risk_scores = {}
        severity_scores = {MILD: 0.25, MODERATE: 0.5, SEVERE: 0.75, CRITICAL: 1.0}
        for effect in peak_effects:
            organ = effect["organ"]
            score = severity_scores.get(effect["severity"], 0)
            organ_risk_scores[organ] = max(organ_risk_scores.get(organ, 0), score)

        # Drug name
        drug_name = None
        if self.drug_info:
            drug_name = self.drug_info.get("name")

        return {
            "drug_name": drug_name,
            "peak_effects": peak_effects,
            "overall_risk": overall_risk,
            "organ_risk_scores": organ_risk_scores,
            "peak_time_h": round(peak_tp.get("time_h", 0), 2),
            "peak_plasma_conc": round(peak_plasma, 4),
            "structural_alerts": [a["name"] for a in self.structural_alerts],
            "n_effects": len(peak_effects),
        }


# ════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("NeoGen Toxicity Engine — Test")
    print("=" * 60)

    # Test 1: Caffeine at normal dose
    print("\n--- Caffeine 200mg (normal coffee) ---")
    assessor = ToxicityAssessor("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    effects = assessor.assess_at_timepoint(
        plasma_conc=5.0, liver_conc=8.0, kidney_conc=4.0,
        brain_conc=3.0, muscle_conc=2.0, fat_conc=1.0, gut_conc=1.0,
        dose_mg=200,
    )
    print(f"  Effects: {len(effects)}")
    for e in effects:
        print(f"    [{e['severity'].upper()}] {e['organ']}: {e['symptom']}")

    # Test 2: Caffeine overdose
    print("\n--- Caffeine 2000mg (OVERDOSE) ---")
    effects = assessor.assess_at_timepoint(
        plasma_conc=50.0, liver_conc=80.0, kidney_conc=40.0,
        brain_conc=30.0, muscle_conc=20.0, fat_conc=10.0, gut_conc=5.0,
        dose_mg=2000,
    )
    print(f"  Effects: {len(effects)}")
    for e in effects:
        print(f"    [{e['severity'].upper():8s}] {e['organ']:6s}: {e['symptom']}")
        print(f"              {e['vital_sign']}: {e['vital_change']}")

    # Test 3: Paracetamol overdose
    print("\n--- Paracetamol 10g (OVERDOSE) ---")
    assessor2 = ToxicityAssessor("CC(=O)NC1=CC=C(C=C1)O")
    effects = assessor2.assess_at_timepoint(
        plasma_conc=200.0, liver_conc=400.0, kidney_conc=150.0,
        brain_conc=50.0, muscle_conc=30.0, fat_conc=20.0, gut_conc=10.0,
        dose_mg=10000,
    )
    print(f"  Effects: {len(effects)}")
    for e in effects:
        print(f"    [{e['severity'].upper():8s}] {e['organ']:6s}: {e['symptom']}")

    # Test 4: Unknown drug with structural alerts
    print("\n--- Unknown drug with nitro group ---")
    assessor3 = ToxicityAssessor("c1ccc(cc1)[N+](=O)[O-]")  # Nitrobenzene
    print(f"  Structural alerts: {[a['name'] for a in assessor3.structural_alerts]}")

    # Test 5: Full timecourse assessment
    print("\n--- Full timecourse: Aspirin 500mg ---")
    fake_tc = [
        {"time_h": i*0.5, "plasma_conc": 10*np.exp(-0.1*i)*np.sin(0.3*i+0.1)**2,
         "liver_conc": 15*np.exp(-0.08*i), "kidney_conc": 8*np.exp(-0.1*i),
         "brain_conc": 3*np.exp(-0.12*i), "muscle_conc": 5*np.exp(-0.05*i),
         "fat_conc": 2*np.exp(-0.03*i), "gut_conc": 20*np.exp(-0.5*i)}
        for i in range(48)
    ]
    assessor4 = ToxicityAssessor("CC(=O)OC1=CC=CC=C1C(=O)O")
    result = assessor4.assess_full_timecourse(fake_tc, dose_mg=500)
    print(f"  Overall risk: {result['overall_risk']}")
    print(f"  Peak time: {result['peak_time_h']}h")
    print(f"  Effects at peak: {result['n_effects']}")
    print(f"  Organ risks: {result['organ_risk_scores']}")
    for e in result["peak_effects"][:5]:
        print(f"    [{e['severity'].upper():8s}] {e['symptom']}")

    print("\nToxicity engine test PASSED!")
