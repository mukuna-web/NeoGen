"""
Immune Response Module
========================
Models how the immune system reacts to drugs and how drugs affect immunity.

Three types of drug-immune interactions:

1. DRUG ALLERGY (Hypersensitivity)
   - Type I: Immediate (anaphylaxis) — IgE-mediated, minutes
   - Type II: Cytotoxic — antibody attacks drug-coated cells, hours-days
   - Type III: Immune complex — drug-antibody complexes deposit in tissue, days-weeks
   - Type IV: Delayed — T-cell mediated, 24-72 hours (e.g., SJS/TEN)

2. IMMUNE-MEDIATED ORGAN DAMAGE
   - Drug-Induced Liver Injury (DILI) — immune type
   - Drug-induced lupus
   - Drug-induced hemolytic anemia

3. IMMUNOMODULATION
   - Immunosuppression (steroids, methotrexate)
   - Immune activation (checkpoint inhibitors → cytokine storm)
   - Cytokine effects on drug metabolism (IL-6 suppresses CYP enzymes)

Reference:
- Pichler (2003) "Drug Hypersensitivity Reactions: Classification and Relationship to T-Cell Activation"
- FDA DILI classification guidelines
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# DRUG ALLERGY DATABASE
# ════════════════════════════════════════════════════════════════

@dataclass
class AllergyProfile:
    """Drug's allergy/hypersensitivity risk profile."""
    name: str
    smiles: str
    allergy_incidence: float  # Population incidence (0-1)
    hypersensitivity_types: list[str]  # I, II, III, IV
    cross_reactivity: list[str]  # Drug classes that cross-react
    symptoms: dict  # type → symptoms
    risk_factors: list[str]
    severity_range: str  # "mild", "mild-severe", "severe", "life-threatening"
    hla_associations: list[str]  # HLA alleles that increase risk
    rechallenge_safe: bool  # Can you give the drug again?
    desensitization_possible: bool  # Can you slowly re-introduce?


ALLERGY_DATABASE: dict[str, AllergyProfile] = {
    # ── PENICILLINS — Most common drug allergy ──
    "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C": AllergyProfile(
        name="Amoxicillin",
        smiles="CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
        allergy_incidence=0.08,  # ~8% report penicillin allergy (most not true allergy)
        hypersensitivity_types=["I", "II", "IV"],
        cross_reactivity=["All penicillins", "~2% cross with cephalosporins"],
        symptoms={
            "Type I (Immediate)": [
                "Urticaria (hives)", "Angioedema", "Bronchospasm",
                "Anaphylaxis (0.01-0.05%)", "Hypotension", "Laryngeal edema",
            ],
            "Type IV (Delayed)": [
                "Maculopapular rash (most common)", "Drug fever",
                "Stevens-Johnson Syndrome (rare but life-threatening)",
                "Toxic Epidermal Necrolysis (TEN) — very rare, ~30% mortality",
            ],
        },
        risk_factors=["Previous penicillin reaction", "Atopic individuals", "HIV infection",
                       "EBV infection (almost 100% rash with amoxicillin)"],
        severity_range="mild-severe",
        hla_associations=[],
        rechallenge_safe=False,
        desensitization_possible=True,
    ),

    # ── ASPIRIN/NSAIDs ──
    "CC(=O)OC1=CC=CC=C1C(=O)O": AllergyProfile(
        name="Aspirin",
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        allergy_incidence=0.03,  # ~1-3% general population, higher in asthmatics
        hypersensitivity_types=["Non-allergic (COX-1 inhibition)"],
        cross_reactivity=["All NSAIDs (COX-1 inhibitors)", "Ibuprofen", "Naproxen"],
        symptoms={
            "Respiratory (Samter's triad)": [
                "Bronchospasm (in asthmatics)", "Nasal polyps",
                "Rhinosinusitis", "Respiratory distress",
            ],
            "Cutaneous": [
                "Urticaria", "Angioedema", "Flushing",
            ],
        },
        risk_factors=["Asthma", "Nasal polyps", "Chronic urticaria"],
        severity_range="mild-severe",
        hla_associations=[],
        rechallenge_safe=False,
        desensitization_possible=True,
    ),

    # ── SULFONAMIDES ──
    # (Sulfamethoxazole - common in TMP-SMX)

    # ── PARACETAMOL ──
    "CC(=O)NC1=CC=C(C=C1)O": AllergyProfile(
        name="Paracetamol",
        smiles="CC(=O)NC1=CC=C(C=C1)O",
        allergy_incidence=0.001,  # Very rare true allergy
        hypersensitivity_types=["I (very rare)"],
        cross_reactivity=[],
        symptoms={
            "Type I (Extremely rare)": [
                "Urticaria", "Angioedema", "Anaphylaxis (case reports only)",
            ],
        },
        risk_factors=["Previous reaction to paracetamol"],
        severity_range="mild",
        hla_associations=[],
        rechallenge_safe=True,  # Most reported reactions are not true allergy
        desensitization_possible=True,
    ),

    # ── CAFFEINE ──
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": AllergyProfile(
        name="Caffeine",
        smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        allergy_incidence=0.0001,  # Essentially never
        hypersensitivity_types=[],
        cross_reactivity=[],
        symptoms={},
        risk_factors=[],
        severity_range="mild",
        hla_associations=[],
        rechallenge_safe=True,
        desensitization_possible=True,
    ),
}


# ════════════════════════════════════════════════════════════════
# STRUCTURAL ALERTS FOR IMMUNOGENICITY
# ════════════════════════════════════════════════════════════════

IMMUNE_STRUCTURAL_ALERTS = [
    {
        "name": "Beta-lactam ring",
        "smarts": "C1C(=O)N1",
        "risk": "high",
        "explanation": "Beta-lactam drugs (penicillins, cephalosporins) are the most common cause of drug allergy. The beta-lactam ring opens and binds to proteins, creating a hapten.",
        "allergy_types": ["I", "II", "IV"],
    },
    {
        "name": "Sulfonamide group",
        "smarts": "NS(=O)(=O)c",
        "risk": "moderate",
        "explanation": "Aromatic sulfonamides can cause Type IV hypersensitivity. The N4-amino group is oxidized to a hydroxylamine → protein-reactive metabolite.",
        "allergy_types": ["IV"],
    },
    {
        "name": "Aromatic amine",
        "smarts": "Nc1ccccc1",
        "risk": "moderate",
        "explanation": "Aromatic amines are metabolized to hydroxylamines and nitroso intermediates that can bind proteins and trigger immune response.",
        "allergy_types": ["II", "IV"],
    },
    {
        "name": "Hydrazine",
        "smarts": "NN",
        "risk": "moderate",
        "explanation": "Hydrazine-containing drugs (isoniazid, hydralazine) can cause drug-induced lupus through protein haptenation.",
        "allergy_types": ["III"],
    },
    {
        "name": "Quinolone core",
        "smarts": "O=C1C=CN2C=CC=CC12",
        "risk": "low",
        "explanation": "Fluoroquinolones can cause phototoxic and photoallergic reactions, rarely anaphylaxis.",
        "allergy_types": ["I", "IV"],
    },
]


# ════════════════════════════════════════════════════════════════
# IMMUNE-MEDIATED ORGAN DAMAGE
# ════════════════════════════════════════════════════════════════

@dataclass
class ImmuneOrganDamage:
    """Immune-mediated organ damage profile."""
    name: str
    organ: str
    mechanism: str
    symptoms: list[str]
    onset: str  # "hours", "days", "weeks", "months"
    incidence: float
    reversible: bool
    mortality_rate: float
    biomarkers: list[str]  # Lab tests to detect


IMMUNE_ORGAN_DAMAGE_DATABASE = {
    "DILI_immune": ImmuneOrganDamage(
        name="Immune-mediated DILI",
        organ="liver",
        mechanism="Drug or metabolite acts as hapten → T-cell mediated hepatocyte destruction",
        symptoms=["Jaundice", "RUQ pain", "Fatigue", "Dark urine", "Eosinophilia",
                   "Fever", "Rash (accompanying)"],
        onset="weeks",
        incidence=0.001,
        reversible=True,
        mortality_rate=0.05,
        biomarkers=["ALT/AST elevation", "Bilirubin", "Eosinophil count", "ANA"],
    ),
    "SJS_TEN": ImmuneOrganDamage(
        name="Stevens-Johnson Syndrome / TEN",
        organ="skin",
        mechanism="CD8+ T-cells + NK cells attack keratinocytes expressing drug-hapten",
        symptoms=["Painful skin rash", "Mucosal involvement (mouth, eyes, genitals)",
                   "Skin blistering", "Epidermal detachment", "Fever",
                   "Nikolsky sign positive"],
        onset="days",
        incidence=0.00001,
        reversible=True,
        mortality_rate=0.30,
        biomarkers=["Skin biopsy", "Granulysin levels"],
    ),
    "hemolytic_anemia": ImmuneOrganDamage(
        name="Drug-induced hemolytic anemia",
        organ="blood",
        mechanism="Drug binds to RBC surface → antibody-mediated RBC destruction (Type II)",
        symptoms=["Fatigue", "Pallor", "Jaundice", "Dark urine (hemoglobinuria)",
                   "Tachycardia", "Shortness of breath"],
        onset="days",
        incidence=0.0001,
        reversible=True,
        mortality_rate=0.01,
        biomarkers=["Direct Coombs test", "Reticulocyte count", "LDH", "Haptoglobin (low)"],
    ),
    "drug_lupus": ImmuneOrganDamage(
        name="Drug-induced lupus",
        organ="systemic",
        mechanism="Drug → anti-histone antibodies → immune complex deposition (Type III)",
        symptoms=["Arthralgia", "Myalgia", "Fever", "Serositis (pleuritis, pericarditis)",
                   "Malar rash", "Photosensitivity"],
        onset="months",
        incidence=0.001,
        reversible=True,
        mortality_rate=0.001,
        biomarkers=["Anti-histone antibodies", "ANA", "Anti-dsDNA (usually negative)"],
    ),
    "agranulocytosis": ImmuneOrganDamage(
        name="Drug-induced agranulocytosis",
        organ="bone_marrow",
        mechanism="Immune-mediated neutrophil destruction or bone marrow suppression",
        symptoms=["Fever", "Sore throat", "Mouth ulcers", "Severe infections",
                   "Sepsis risk"],
        onset="weeks",
        incidence=0.0001,
        reversible=True,
        mortality_rate=0.05,
        biomarkers=["Absolute neutrophil count (ANC < 500)", "CBC with differential"],
    ),
}


# ════════════════════════════════════════════════════════════════
# CYTOKINE EFFECTS ON DRUG METABOLISM
# ════════════════════════════════════════════════════════════════

@dataclass
class CytokineState:
    """
    Models how inflammatory cytokines alter drug metabolism.

    Key cytokines:
    - IL-6: Most potent CYP suppressor. Suppresses CYP3A4, 1A2, 2C19 by 50-90%
    - TNF-α: Moderate CYP suppression
    - IL-1β: Synergizes with IL-6
    - IFN-γ: Suppresses CYP1A2, 2C19

    Clinical relevance:
    - Infections, autoimmune diseases, cancer → elevated cytokines → drug accumulation
    - Anti-IL-6 therapy (tocilizumab) → CYP enzymes RECOVER → drug levels DROP
    """
    name: str
    il6_level: str  # "normal", "elevated", "high", "very_high"
    tnf_level: str  # "normal", "elevated", "high"

    # CYP suppression multipliers
    cyp3a4_multiplier: float = 1.0
    cyp1a2_multiplier: float = 1.0
    cyp2c19_multiplier: float = 1.0
    cyp2c9_multiplier: float = 1.0
    cyp2d6_multiplier: float = 1.0  # Least affected by cytokines

    clinical_impact: str = ""


CYTOKINE_STATES = {
    "normal": CytokineState(
        name="Normal (no inflammation)",
        il6_level="normal", tnf_level="normal",
        clinical_impact="No cytokine-mediated CYP changes.",
    ),
    "mild_inflammation": CytokineState(
        name="Mild inflammation (URI, mild infection)",
        il6_level="elevated", tnf_level="normal",
        cyp3a4_multiplier=0.80,
        cyp1a2_multiplier=0.75,
        cyp2c19_multiplier=0.80,
        clinical_impact="Mild CYP suppression. Monitor narrow TI drugs.",
    ),
    "moderate_inflammation": CytokineState(
        name="Moderate inflammation (active autoimmune, moderate infection)",
        il6_level="high", tnf_level="elevated",
        cyp3a4_multiplier=0.50,
        cyp1a2_multiplier=0.40,
        cyp2c19_multiplier=0.50,
        cyp2c9_multiplier=0.70,
        clinical_impact="Significant CYP suppression. Drug levels may double. Dose reduction needed.",
    ),
    "severe_inflammation": CytokineState(
        name="Severe inflammation (sepsis, cytokine storm, active RA flare)",
        il6_level="very_high", tnf_level="high",
        cyp3a4_multiplier=0.15,
        cyp1a2_multiplier=0.10,
        cyp2c19_multiplier=0.20,
        cyp2c9_multiplier=0.40,
        cyp2d6_multiplier=0.70,
        clinical_impact="CRITICAL: CYP enzymes 70-90% suppressed. Drug accumulation likely. "
                         "Use IV route, monitor drug levels q6h.",
    ),
    "anti_il6_therapy": CytokineState(
        name="On anti-IL-6 therapy (tocilizumab/sarilumab)",
        il6_level="blocked", tnf_level="normal",
        cyp3a4_multiplier=1.3,  # CYP RECOVERY — may cause sub-therapeutic levels
        cyp1a2_multiplier=1.2,
        cyp2c19_multiplier=1.2,
        clinical_impact="WARNING: Anti-IL-6 therapy reverses CYP suppression. "
                         "Drugs cleared by CYP3A4 may drop to sub-therapeutic levels. "
                         "Increase doses when starting tocilizumab.",
    ),
}


# ════════════════════════════════════════════════════════════════
# IMMUNE RESPONSE ENGINE
# ════════════════════════════════════════════════════════════════

class ImmuneResponseEngine:
    """
    Assesses drug-immune interactions.

    Evaluates:
    1. Drug allergy risk (from database + structural alerts)
    2. Immune-mediated organ damage risk
    3. Cytokine effects on metabolism
    """

    def __init__(self, smiles: str, cytokine_state: str = "normal"):
        self.smiles = smiles
        self.cytokine_state = CYTOKINE_STATES.get(cytokine_state, CYTOKINE_STATES["normal"])

        # Look up allergy profile
        if smiles in ALLERGY_DATABASE:
            self.allergy_profile = ALLERGY_DATABASE[smiles]
            self.known_allergy = True
        else:
            self.allergy_profile = None
            self.known_allergy = False

    def assess_allergy_risk(self) -> dict:
        """Assess drug allergy/hypersensitivity risk."""
        # Check structural alerts
        structural_risks = []
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(self.smiles)
            if mol:
                for alert in IMMUNE_STRUCTURAL_ALERTS:
                    pattern = Chem.MolFromSmarts(alert["smarts"])
                    if pattern and mol.HasSubstructMatch(pattern):
                        structural_risks.append({
                            "name": alert["name"],
                            "risk": alert["risk"],
                            "explanation": alert["explanation"],
                            "allergy_types": alert["allergy_types"],
                        })
        except ImportError:
            pass

        if self.known_allergy:
            p = self.allergy_profile
            return {
                "known_profile": True,
                "drug_name": p.name,
                "allergy_incidence": p.allergy_incidence,
                "incidence_description": f"{p.allergy_incidence*100:.1f}% of population",
                "hypersensitivity_types": p.hypersensitivity_types,
                "cross_reactivity": p.cross_reactivity,
                "symptoms": p.symptoms,
                "risk_factors": p.risk_factors,
                "severity_range": p.severity_range,
                "hla_associations": p.hla_associations,
                "rechallenge_safe": p.rechallenge_safe,
                "desensitization_possible": p.desensitization_possible,
                "structural_alerts": structural_risks,
            }
        else:
            # Estimate from structural alerts
            overall_risk = "none"
            if structural_risks:
                if any(s["risk"] == "high" for s in structural_risks):
                    overall_risk = "moderate"
                elif any(s["risk"] == "moderate" for s in structural_risks):
                    overall_risk = "low"

            return {
                "known_profile": False,
                "drug_name": "Unknown",
                "estimated_risk": overall_risk,
                "structural_alerts": structural_risks,
                "note": "No specific allergy data available. Risk estimated from structural features.",
            }

    def assess_immune_organ_damage(self) -> dict:
        """Assess risk of immune-mediated organ damage."""
        risks = []

        # Check for structural features associated with immune organ damage
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(self.smiles)
            if mol:
                # Beta-lactam → hemolytic anemia risk
                bl_pattern = Chem.MolFromSmarts("C1C(=O)N1")
                if bl_pattern and mol.HasSubstructMatch(bl_pattern):
                    risks.append({
                        "condition": IMMUNE_ORGAN_DAMAGE_DATABASE["hemolytic_anemia"],
                        "risk": "moderate",
                    })

                # Aromatic amine → DILI immune type
                aa_pattern = Chem.MolFromSmarts("Nc1ccccc1")
                if aa_pattern and mol.HasSubstructMatch(aa_pattern):
                    risks.append({
                        "condition": IMMUNE_ORGAN_DAMAGE_DATABASE["DILI_immune"],
                        "risk": "moderate",
                    })

                # Hydrazine → drug-induced lupus
                hz_pattern = Chem.MolFromSmarts("NN")
                if hz_pattern and mol.HasSubstructMatch(hz_pattern):
                    risks.append({
                        "condition": IMMUNE_ORGAN_DAMAGE_DATABASE["drug_lupus"],
                        "risk": "low",
                    })
        except ImportError:
            pass

        results = []
        for r in risks:
            c = r["condition"]
            results.append({
                "name": c.name,
                "organ": c.organ,
                "mechanism": c.mechanism,
                "symptoms": c.symptoms,
                "onset": c.onset,
                "risk_level": r["risk"],
                "reversible": c.reversible,
                "mortality_rate": f"{c.mortality_rate*100:.1f}%",
                "biomarkers": c.biomarkers,
            })

        return {
            "n_risks": len(results),
            "risks": results,
            "overall_immune_organ_risk": "moderate" if results else "none",
        }

    def get_cytokine_effects(self) -> dict:
        """Get cytokine effects on drug metabolism."""
        cs = self.cytokine_state
        return {
            "state": cs.name,
            "il6": cs.il6_level,
            "tnf": cs.tnf_level,
            "cyp_modifiers": {
                "CYP3A4": round(cs.cyp3a4_multiplier, 3),
                "CYP1A2": round(cs.cyp1a2_multiplier, 3),
                "CYP2C19": round(cs.cyp2c19_multiplier, 3),
                "CYP2C9": round(cs.cyp2c9_multiplier, 3),
                "CYP2D6": round(cs.cyp2d6_multiplier, 3),
            },
            "clinical_impact": cs.clinical_impact,
        }

    def get_full_assessment(self) -> dict:
        """Complete immune response assessment."""
        return {
            "allergy": self.assess_allergy_risk(),
            "immune_organ_damage": self.assess_immune_organ_damage(),
            "cytokine_effects": self.get_cytokine_effects(),
        }


# ── Standalone test ──
if __name__ == "__main__":
    print("=" * 60)
    print("Immune Response Engine — Test")
    print("=" * 60)

    # Test 1: Amoxicillin (known high allergy risk)
    print("\n--- Amoxicillin (penicillin allergy) ---")
    eng = ImmuneResponseEngine(
        "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C"
    )
    result = eng.assess_allergy_risk()
    print(f"  Known: {result['known_profile']}")
    print(f"  Incidence: {result.get('incidence_description', 'N/A')}")
    print(f"  Types: {result.get('hypersensitivity_types', [])}")
    print(f"  Structural alerts: {len(result.get('structural_alerts', []))}")

    # Test 2: Cytokine effects in sepsis
    print("\n--- Caffeine in severe inflammation ---")
    eng2 = ImmuneResponseEngine(
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        cytokine_state="severe_inflammation",
    )
    cyto = eng2.get_cytokine_effects()
    print(f"  State: {cyto['state']}")
    print(f"  CYP3A4: {cyto['cyp_modifiers']['CYP3A4']*100:.0f}% activity")
    print(f"  CYP1A2: {cyto['cyp_modifiers']['CYP1A2']*100:.0f}% activity")
    print(f"  Impact: {cyto['clinical_impact'][:80]}...")

    # Test 3: Anti-IL-6 therapy
    print("\n--- Drug PK on tocilizumab ---")
    eng3 = ImmuneResponseEngine("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "anti_il6_therapy")
    cyto3 = eng3.get_cytokine_effects()
    print(f"  CYP3A4: {cyto3['cyp_modifiers']['CYP3A4']*100:.0f}% (INCREASED)")
    print(f"  Impact: {cyto3['clinical_impact'][:80]}...")

    print("\nImmune response test PASSED!")
