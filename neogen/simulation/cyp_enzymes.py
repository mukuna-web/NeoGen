"""
CYP Enzyme Interaction Database
=================================
Models cytochrome P450 enzyme interactions for drug-drug interaction prediction.

The liver uses CYP enzymes to metabolize drugs. When two drugs use the same
enzyme, they compete — one drug can inhibit or induce the enzyme, changing
how fast the other drug is cleared.

This is how most clinically significant drug interactions work:
- Inhibitor blocks enzyme → victim drug accumulates → toxicity
- Inducer speeds up enzyme → victim drug cleared too fast → treatment failure

Key CYP enzymes (responsible for ~75% of drug metabolism):
- CYP3A4: 50% of drugs (statins, calcium channel blockers, HIV meds)
- CYP2D6: 25% of drugs (antidepressants, beta-blockers, opioids)
- CYP1A2: caffeine, theophylline, some antipsychotics
- CYP2C9: warfarin, NSAIDs, some antidiabetics
- CYP2C19: PPIs (omeprazole), clopidogrel, some antidepressants
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# CYP ENZYME DATABASE
# ════════════════════════════════════════════════════════════════

@dataclass
class CYPProfile:
    """CYP enzyme profile for a drug."""
    name: str
    smiles: str
    # Which enzymes metabolize this drug (substrate)
    substrates: list[str] = field(default_factory=list)
    # Which enzymes this drug inhibits
    inhibits: list[str] = field(default_factory=list)
    # Which enzymes this drug induces (speeds up)
    induces: list[str] = field(default_factory=list)
    # Inhibition potency (0-1, where 1 = complete inhibition)
    inhibition_potency: dict[str, float] = field(default_factory=dict)
    # Primary metabolism fraction per enzyme
    metabolism_fraction: dict[str, float] = field(default_factory=dict)


# ── Known Drug CYP Profiles ───────────────────────────────────────

CYP_DATABASE: dict[str, CYPProfile] = {}

def _add(smiles, name, substrates=None, inhibits=None, induces=None,
         inhibition_potency=None, metabolism_fraction=None):
    CYP_DATABASE[smiles] = CYPProfile(
        name=name, smiles=smiles,
        substrates=substrates or [],
        inhibits=inhibits or [],
        induces=induces or [],
        inhibition_potency=inhibition_potency or {},
        metabolism_fraction=metabolism_fraction or {},
    )

# ── Analgesics / NSAIDs ──
_add("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin",
     substrates=["CYP2C9"], metabolism_fraction={"CYP2C9": 0.6, "other": 0.4})

_add("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen",
     substrates=["CYP2C9", "CYP2C8"],
     inhibits=["CYP2C9"],
     inhibition_potency={"CYP2C9": 0.3},
     metabolism_fraction={"CYP2C9": 0.7, "CYP2C8": 0.2})

_add("CC(=O)NC1=CC=C(C=C1)O", "Paracetamol",
     substrates=["CYP2E1", "CYP1A2", "CYP3A4"],
     metabolism_fraction={"CYP2E1": 0.5, "CYP1A2": 0.2, "CYP3A4": 0.15})

# ── Stimulants ──
_add("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine",
     substrates=["CYP1A2"],
     metabolism_fraction={"CYP1A2": 0.95})

# ── Cardiovascular ──
_add("CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "Atorvastatin",
     substrates=["CYP3A4"],
     metabolism_fraction={"CYP3A4": 0.85})

# Simvastatin (another statin, heavily CYP3A4)
_add("CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C", "Simvastatin",
     substrates=["CYP3A4"],
     metabolism_fraction={"CYP3A4": 0.90})

# ── Antacids / PPIs ──
_add("CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2", "Omeprazole",
     substrates=["CYP2C19", "CYP3A4"],
     inhibits=["CYP2C19"],
     inhibition_potency={"CYP2C19": 0.6},
     metabolism_fraction={"CYP2C19": 0.7, "CYP3A4": 0.25})

# ── Anticoagulants ──
_add("CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O", "Warfarin",
     substrates=["CYP2C9", "CYP3A4", "CYP1A2"],
     metabolism_fraction={"CYP2C9": 0.6, "CYP3A4": 0.2, "CYP1A2": 0.15})

# ── Antibiotics ──
_add("CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C", "Amoxicillin",
     substrates=[], metabolism_fraction={})  # Renally cleared, minimal CYP

# ── Antidepressants ──
# Fluoxetine (Prozac) - strong CYP2D6 inhibitor
_add("CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F", "Fluoxetine",
     substrates=["CYP2D6", "CYP2C9"],
     inhibits=["CYP2D6", "CYP2C19", "CYP3A4"],
     inhibition_potency={"CYP2D6": 0.9, "CYP2C19": 0.5, "CYP3A4": 0.3},
     metabolism_fraction={"CYP2D6": 0.6, "CYP2C9": 0.3})

# ── Antifungals ──
# Ketoconazole - strongest CYP3A4 inhibitor
_add("C1=CC=C(C=C1)C(CN2C=NC=N2)OC3=CC=C(C=C3)OC4=CC(=CC(=C4)Cl)Cl", "Ketoconazole",
     substrates=["CYP3A4"],
     inhibits=["CYP3A4"],
     inhibition_potency={"CYP3A4": 0.95},
     metabolism_fraction={"CYP3A4": 0.90})

# ── Opioids ──
# Codeine (needs CYP2D6 to become morphine)
_add("COC1=CC2=C(C=C1)C3CC4=CC=C(C(=C4C3C2O)O)OC", "Codeine",
     substrates=["CYP2D6", "CYP3A4"],
     metabolism_fraction={"CYP2D6": 0.5, "CYP3A4": 0.4})

# ── Diabetes ──
_add("CN(C)C(=N)NC(=N)N", "Metformin",
     substrates=[], metabolism_fraction={})  # Not CYP metabolized

# ── Antiplatelets ──
# Clopidogrel (needs CYP2C19 to be activated — prodrug)
_add("COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=CC=CS3", "Clopidogrel",
     substrates=["CYP2C19", "CYP3A4", "CYP1A2"],
     metabolism_fraction={"CYP2C19": 0.45, "CYP3A4": 0.3, "CYP1A2": 0.15})

# ── Natural products / food ──
# Grapefruit juice component (bergamottin) - CYP3A4 inhibitor
_add("CC(=CCC1=CC2=C(C=C1OC)OC(C2=O)CC=C(C)C)C", "Bergamottin (Grapefruit)",
     inhibits=["CYP3A4"],
     inhibition_potency={"CYP3A4": 0.7})

# Ethanol / Alcohol
_add("CCO", "Ethanol",
     substrates=["CYP2E1"],
     inhibits=["CYP1A2"],
     induces=["CYP2E1"],
     inhibition_potency={"CYP1A2": 0.4},
     metabolism_fraction={"CYP2E1": 0.3})

# St. John's Wort (hyperforin) - CYP3A4 inducer
_add("CC(C)CC(=O)C1=C(C(=C(C(=C1O)CC=C(C)C)O)CC=C(C)C)CC2CCC(C2)(CC=C(C)C)CC=C(C)C", "St. John's Wort",
     induces=["CYP3A4", "CYP2C9"],)


# ════════════════════════════════════════════════════════════════
# KNOWN DDI PAIRS
# ════════════════════════════════════════════════════════════════

@dataclass
class DDIPair:
    """A known drug-drug interaction pair."""
    drug_a_name: str
    drug_b_name: str
    interaction_type: str  # "metabolic", "pharmacodynamic", "both"
    mechanism: str
    clinical_effect: str
    severity: str  # "mild", "moderate", "severe", "contraindicated"
    cyp_involved: str  # Which CYP enzyme
    effect_on_victim: str  # "increased_levels", "decreased_levels", "additive_toxicity"
    recommendation: str

KNOWN_DDI_PAIRS = [
    DDIPair("Warfarin", "Aspirin", "pharmacodynamic",
            "Both inhibit clotting via different mechanisms (COX-1 + vitamin K antagonist)",
            "3-5x increased bleeding risk",
            "severe", "", "additive_toxicity",
            "Avoid combination or monitor INR closely"),
    DDIPair("Warfarin", "Ibuprofen", "both",
            "Ibuprofen inhibits CYP2C9 (warfarin's main enzyme) + antiplatelet effect",
            "Increased warfarin levels + bleeding risk",
            "severe", "CYP2C9", "increased_levels",
            "Use paracetamol instead of ibuprofen"),
    DDIPair("Caffeine", "Ethanol", "metabolic",
            "Alcohol inhibits CYP1A2, caffeine's primary metabolic enzyme",
            "Caffeine effects last 2-3x longer, masking alcohol impairment",
            "moderate", "CYP1A2", "increased_levels",
            "Be aware caffeine won't sober you up"),
    DDIPair("Simvastatin", "Bergamottin (Grapefruit)", "metabolic",
            "Bergamottin in grapefruit irreversibly inhibits intestinal CYP3A4",
            "Simvastatin levels increase 5-15x → rhabdomyolysis risk",
            "contraindicated", "CYP3A4", "increased_levels",
            "Avoid grapefruit with simvastatin entirely"),
    DDIPair("Atorvastatin", "Ketoconazole", "metabolic",
            "Ketoconazole is the strongest CYP3A4 inhibitor known",
            "Atorvastatin levels increase 3-6x → muscle toxicity",
            "contraindicated", "CYP3A4", "increased_levels",
            "Use alternative antifungal or switch statin to rosuvastatin"),
    DDIPair("Paracetamol", "Ethanol", "metabolic",
            "Chronic alcohol induces CYP2E1, producing more NAPQI (toxic metabolite)",
            "Liver toxicity at lower paracetamol doses (>2g/day becomes dangerous)",
            "severe", "CYP2E1", "increased_levels",
            "Limit paracetamol to 2g/day if drinking regularly"),
    DDIPair("Omeprazole", "Clopidogrel", "metabolic",
            "Omeprazole inhibits CYP2C19, which is needed to activate clopidogrel",
            "Reduced antiplatelet effect → increased heart attack/stent thrombosis risk",
            "severe", "CYP2C19", "decreased_levels",
            "Use pantoprazole instead (less CYP2C19 inhibition)"),
    DDIPair("Fluoxetine", "Codeine", "metabolic",
            "Fluoxetine strongly inhibits CYP2D6, blocking codeine → morphine conversion",
            "Codeine provides no pain relief (can't be converted to active form)",
            "moderate", "CYP2D6", "decreased_levels",
            "Use alternative analgesic (tramadol also affected)"),
    DDIPair("Ibuprofen", "Aspirin", "pharmacodynamic",
            "Ibuprofen blocks aspirin's access to COX-1 active site",
            "Aspirin's cardioprotective effect is negated",
            "moderate", "", "decreased_levels",
            "Take aspirin 30min before ibuprofen, or use paracetamol"),
    DDIPair("Metformin", "Ethanol", "pharmacodynamic",
            "Both impair hepatic lactate metabolism independently",
            "Risk of lactic acidosis (rare but fatal)",
            "moderate", "", "additive_toxicity",
            "Limit alcohol intake with metformin"),
]


# ════════════════════════════════════════════════════════════════
# CYP INTERACTION ENGINE
# ════════════════════════════════════════════════════════════════

class CYPInteractionEngine:
    """
    Assesses CYP-mediated drug interactions between two drugs.

    Returns:
    - Which enzymes are affected
    - How clearance changes for each drug
    - Matching known DDI pairs with clinical guidance
    """

    def __init__(self, drug_a_smiles: str, drug_b_smiles: str):
        self.profile_a = CYP_DATABASE.get(drug_a_smiles)
        self.profile_b = CYP_DATABASE.get(drug_b_smiles)

    def assess_interaction(self) -> dict:
        """
        Assess CYP-mediated interaction between two drugs.

        Returns dict with:
        - enzyme_effects: which enzymes are affected and how
        - clearance_change_a: fraction change in Drug A clearance (1.0 = no change)
        - clearance_change_b: fraction change in Drug B clearance
        - known_interaction: matching DDI pair if known
        - warnings: list of clinical warnings
        """
        result = {
            "enzyme_effects": [],
            "clearance_change_a": 1.0,
            "clearance_change_b": 1.0,
            "known_interaction": None,
            "warnings": [],
            "cyp_status": {},  # enzyme → status ("normal", "inhibited", "induced")
        }

        if not self.profile_a and not self.profile_b:
            result["warnings"].append("CYP profiles unknown for both drugs — interaction assessment limited")
            return result

        # Initialize all major CYP enzymes
        enzymes = ["CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9", "CYP2C19", "CYP2E1"]
        for e in enzymes:
            result["cyp_status"][e] = {"status": "normal", "activity": 1.0}

        # Check Drug A's effect on Drug B's metabolism
        if self.profile_a and self.profile_b:
            # Drug A inhibits enzymes that Drug B needs
            for enzyme in (self.profile_a.inhibits or []):
                potency = self.profile_a.inhibition_potency.get(enzyme, 0.5)
                result["cyp_status"][enzyme] = {
                    "status": "inhibited",
                    "activity": 1.0 - potency,
                    "inhibitor": self.profile_a.name,
                }

                # If Drug B is metabolized by this enzyme, its clearance drops
                b_fraction = self.profile_b.metabolism_fraction.get(enzyme, 0)
                if b_fraction > 0:
                    # Clearance reduction = fraction metabolized by this enzyme * inhibition
                    reduction = b_fraction * potency
                    result["clearance_change_b"] *= (1.0 - reduction)
                    result["enzyme_effects"].append({
                        "enzyme": enzyme,
                        "inhibitor": self.profile_a.name,
                        "victim": self.profile_b.name,
                        "effect": f"{self.profile_a.name} inhibits {enzyme} ({potency*100:.0f}%), reducing {self.profile_b.name} clearance by {reduction*100:.0f}%",
                        "severity": "high" if reduction > 0.5 else "moderate" if reduction > 0.2 else "low",
                    })

            # Drug B inhibits enzymes that Drug A needs
            for enzyme in (self.profile_b.inhibits or []):
                potency = self.profile_b.inhibition_potency.get(enzyme, 0.5)
                if result["cyp_status"][enzyme]["status"] == "normal":
                    result["cyp_status"][enzyme] = {
                        "status": "inhibited",
                        "activity": 1.0 - potency,
                        "inhibitor": self.profile_b.name,
                    }

                a_fraction = self.profile_a.metabolism_fraction.get(enzyme, 0)
                if a_fraction > 0:
                    reduction = a_fraction * potency
                    result["clearance_change_a"] *= (1.0 - reduction)
                    result["enzyme_effects"].append({
                        "enzyme": enzyme,
                        "inhibitor": self.profile_b.name,
                        "victim": self.profile_a.name,
                        "effect": f"{self.profile_b.name} inhibits {enzyme} ({potency*100:.0f}%), reducing {self.profile_a.name} clearance by {reduction*100:.0f}%",
                        "severity": "high" if reduction > 0.5 else "moderate" if reduction > 0.2 else "low",
                    })

            # Check for enzyme induction
            for enzyme in (self.profile_a.induces or []):
                result["cyp_status"][enzyme] = {
                    "status": "induced",
                    "activity": 2.0,  # Induction roughly doubles enzyme activity
                    "inducer": self.profile_a.name,
                }
                b_fraction = self.profile_b.metabolism_fraction.get(enzyme, 0)
                if b_fraction > 0:
                    result["clearance_change_b"] *= (1.0 + b_fraction)
                    result["enzyme_effects"].append({
                        "enzyme": enzyme,
                        "inhibitor": self.profile_a.name,
                        "victim": self.profile_b.name,
                        "effect": f"{self.profile_a.name} induces {enzyme}, increasing {self.profile_b.name} clearance (drug may be ineffective)",
                        "severity": "moderate",
                    })

            for enzyme in (self.profile_b.induces or []):
                if result["cyp_status"][enzyme]["status"] == "normal":
                    result["cyp_status"][enzyme] = {
                        "status": "induced",
                        "activity": 2.0,
                        "inducer": self.profile_b.name,
                    }
                a_fraction = self.profile_a.metabolism_fraction.get(enzyme, 0)
                if a_fraction > 0:
                    result["clearance_change_a"] *= (1.0 + a_fraction)
                    result["enzyme_effects"].append({
                        "enzyme": enzyme,
                        "inhibitor": self.profile_b.name,
                        "victim": self.profile_a.name,
                        "effect": f"{self.profile_b.name} induces {enzyme}, increasing {self.profile_a.name} clearance",
                        "severity": "moderate",
                    })

        # Check known DDI pairs
        names = set()
        if self.profile_a:
            names.add(self.profile_a.name)
        if self.profile_b:
            names.add(self.profile_b.name)

        for ddi in KNOWN_DDI_PAIRS:
            if {ddi.drug_a_name, ddi.drug_b_name} == names:
                result["known_interaction"] = {
                    "drug_a": ddi.drug_a_name,
                    "drug_b": ddi.drug_b_name,
                    "type": ddi.interaction_type,
                    "mechanism": ddi.mechanism,
                    "clinical_effect": ddi.clinical_effect,
                    "severity": ddi.severity,
                    "cyp_involved": ddi.cyp_involved,
                    "recommendation": ddi.recommendation,
                }
                result["warnings"].append(f"KNOWN INTERACTION: {ddi.clinical_effect}")
                break

        # Generate warnings based on clearance changes
        if result["clearance_change_b"] < 0.5:
            name = self.profile_b.name if self.profile_b else "Drug B"
            result["warnings"].append(
                f"DANGER: {name} clearance reduced by {(1-result['clearance_change_b'])*100:.0f}% — "
                f"drug levels will be {1/result['clearance_change_b']:.1f}x higher than expected"
            )
        elif result["clearance_change_b"] < 0.8:
            name = self.profile_b.name if self.profile_b else "Drug B"
            result["warnings"].append(
                f"CAUTION: {name} clearance reduced by {(1-result['clearance_change_b'])*100:.0f}% — "
                f"consider dose reduction"
            )

        if result["clearance_change_a"] < 0.5:
            name = self.profile_a.name if self.profile_a else "Drug A"
            result["warnings"].append(
                f"DANGER: {name} clearance reduced by {(1-result['clearance_change_a'])*100:.0f}% — "
                f"drug levels will be {1/result['clearance_change_a']:.1f}x higher than expected"
            )

        return result


# ════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("NeoGen CYP Enzyme Interaction Engine — Test")
    print("=" * 60)

    # Test 1: Omeprazole + Clopidogrel (CYP2C19 inhibition)
    print("\n--- Omeprazole + Clopidogrel ---")
    engine = CYPInteractionEngine(
        "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2",  # Omeprazole
        "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=CC=CS3",         # Clopidogrel
    )
    r = engine.assess_interaction()
    print(f"  CL change Drug A: {r['clearance_change_a']:.2f}x")
    print(f"  CL change Drug B: {r['clearance_change_b']:.2f}x")
    for e in r["enzyme_effects"]:
        print(f"  [{e['severity'].upper()}] {e['effect']}")
    if r["known_interaction"]:
        ki = r["known_interaction"]
        print(f"  KNOWN: {ki['clinical_effect']}")
        print(f"  RECOMMENDATION: {ki['recommendation']}")
    for w in r["warnings"]:
        print(f"  WARNING: {w}")

    # Test 2: Caffeine + Alcohol (CYP1A2 inhibition)
    print("\n--- Caffeine + Alcohol ---")
    engine2 = CYPInteractionEngine(
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CCO",                              # Ethanol
    )
    r2 = engine2.assess_interaction()
    print(f"  Caffeine CL change: {r2['clearance_change_a']:.2f}x")
    print(f"  Alcohol CL change: {r2['clearance_change_b']:.2f}x")
    for e in r2["enzyme_effects"]:
        print(f"  {e['effect']}")
    for w in r2["warnings"]:
        print(f"  WARNING: {w}")

    # Test 3: Fluoxetine + Codeine (CYP2D6 block)
    print("\n--- Fluoxetine + Codeine ---")
    engine3 = CYPInteractionEngine(
        "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",  # Fluoxetine
        "COC1=CC2=C(C=C1)C3CC4=CC=C(C(=C4C3C2O)O)OC",  # Codeine
    )
    r3 = engine3.assess_interaction()
    print(f"  Codeine CL change: {r3['clearance_change_b']:.2f}x")
    for e in r3["enzyme_effects"]:
        print(f"  {e['effect']}")
    if r3["known_interaction"]:
        print(f"  KNOWN: {r3['known_interaction']['clinical_effect']}")

    # Test 4: CYP status summary
    print("\n--- CYP Enzyme Status (Omeprazole + Clopidogrel) ---")
    for enzyme, status in r["cyp_status"].items():
        if status["status"] != "normal":
            print(f"  {enzyme}: {status['status']} (activity: {status['activity']:.0%})")

    print(f"\nDatabase: {len(CYP_DATABASE)} drugs, {len(KNOWN_DDI_PAIRS)} known DDI pairs")
    print("CYP engine test PASSED!")
