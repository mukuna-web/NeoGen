"""
Hepatic Metabolism Pathways
============================
Models the 3 phases of liver drug metabolism:

Phase I: Oxidation/Reduction (CYP enzymes)
   Drug → Oxidized metabolite (often more reactive/toxic!)
   Example: Paracetamol → NAPQI (toxic!) via CYP2E1

Phase II: Conjugation (UGT, GST, NAT, SULT)
   Metabolite → Conjugated form (water-soluble, ready for excretion)
   Example: NAPQI + Glutathione → Mercapturic acid (safe, excreted)

Phase III: Efflux (P-gp, MRP, BCRP transporters)
   Conjugated drug → pumped into bile or back into blood for renal excretion

Why this matters:
- Phase I can create TOXIC intermediates (reactive metabolites)
- If Phase II is overwhelmed (e.g., glutathione depletion), toxicity occurs
- This is EXACTLY how paracetamol overdose kills: too much NAPQI, not enough GSH

Reference: Parkinson "Biotransformation of Xenobiotics" (Casarett & Doull's Toxicology)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# METABOLIC PATHWAY DATABASE
# ════════════════════════════════════════════════════════════════

@dataclass
class Metabolite:
    """A metabolic product."""
    name: str
    toxicity: str  # "none", "low", "moderate", "high", "critical"
    reactive: bool  # True = reactive intermediate that can damage proteins/DNA
    detox_pathway: str  # How the body neutralizes it
    organ_damage: list[str] = field(default_factory=list)  # Organs affected if not detoxified


@dataclass
class MetabolicPathway:
    """Complete metabolic pathway for a drug."""
    drug_name: str
    smiles: str

    # Phase I
    phase1_enzymes: list[str] = field(default_factory=list)  # CYP enzymes involved
    phase1_fraction: float = 0.7  # Fraction metabolized by Phase I
    phase1_metabolites: list[Metabolite] = field(default_factory=list)
    phase1_rate: float = 1.0  # Relative rate (1.0 = normal)

    # Phase II
    phase2_pathways: list[str] = field(default_factory=list)  # UGT, GST, NAT, SULT
    phase2_fraction: float = 0.25  # Fraction directly conjugated
    phase2_capacity: float = 1.0  # 1.0 = normal, <1 = impaired
    gsh_requirement: float = 0.0  # Glutathione needed (0-1 scale, 1 = heavy)

    # Phase III
    phase3_transporters: list[str] = field(default_factory=list)  # P-gp, MRP2, BCRP
    biliary_excretion: float = 0.1  # Fraction excreted via bile

    # Toxicity risk
    has_reactive_metabolite: bool = False
    toxic_dose_threshold_mg: float = 10000  # Dose above which toxicity likely


# Known metabolic pathways
METABOLIC_DATABASE: dict[str, MetabolicPathway] = {
    # ── PARACETAMOL — The textbook example ──
    "CC(=O)NC1=CC=C(C=C1)O": MetabolicPathway(
        drug_name="Paracetamol",
        smiles="CC(=O)NC1=CC=C(C=C1)O",
        phase1_enzymes=["CYP2E1", "CYP1A2", "CYP3A4"],
        phase1_fraction=0.05,  # Only 5% goes through CYP → NAPQI
        phase1_metabolites=[
            Metabolite(
                name="NAPQI (N-acetyl-p-benzoquinone imine)",
                toxicity="critical",
                reactive=True,
                detox_pathway="Glutathione (GSH) conjugation",
                organ_damage=["liver"],
            ),
        ],
        phase2_pathways=["UGT (glucuronidation)", "SULT (sulfation)"],
        phase2_fraction=0.90,  # 90% safely conjugated directly
        phase2_capacity=1.0,
        gsh_requirement=0.7,  # Heavy GSH demand for NAPQI detox
        phase3_transporters=["MRP2"],
        biliary_excretion=0.05,
        has_reactive_metabolite=True,
        toxic_dose_threshold_mg=4000,  # >4g/day risks liver damage
    ),

    # ── ASPIRIN ──
    "CC(=O)OC1=CC=CC=C1C(=O)O": MetabolicPathway(
        drug_name="Aspirin",
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        phase1_enzymes=["Esterases"],  # Hydrolyzed to salicylic acid
        phase1_fraction=0.95,
        phase1_metabolites=[
            Metabolite(
                name="Salicylic acid",
                toxicity="low",
                reactive=False,
                detox_pathway="Glycine conjugation + glucuronidation",
                organ_damage=[],
            ),
        ],
        phase2_pathways=["Glycine conjugation", "UGT"],
        phase2_fraction=0.05,
        gsh_requirement=0.0,
        phase3_transporters=["OAT"],
        biliary_excretion=0.0,
        has_reactive_metabolite=False,
        toxic_dose_threshold_mg=6000,
    ),

    # ── CAFFEINE ──
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": MetabolicPathway(
        drug_name="Caffeine",
        smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        phase1_enzymes=["CYP1A2"],  # 95% by CYP1A2
        phase1_fraction=0.95,
        phase1_metabolites=[
            Metabolite(
                name="Paraxanthine",
                toxicity="none",
                reactive=False,
                detox_pathway="Further demethylation + acetylation",
                organ_damage=[],
            ),
            Metabolite(
                name="Theobromine",
                toxicity="none",
                reactive=False,
                detox_pathway="Demethylation",
                organ_damage=[],
            ),
        ],
        phase2_pathways=["NAT (acetylation)"],
        phase2_fraction=0.05,
        gsh_requirement=0.0,
        phase3_transporters=[],
        biliary_excretion=0.0,
        has_reactive_metabolite=False,
        toxic_dose_threshold_mg=5000,
    ),

    # ── IBUPROFEN ──
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": MetabolicPathway(
        drug_name="Ibuprofen",
        smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        phase1_enzymes=["CYP2C9", "CYP2C19"],
        phase1_fraction=0.60,
        phase1_metabolites=[
            Metabolite(
                name="Hydroxy-ibuprofen",
                toxicity="none",
                reactive=False,
                detox_pathway="Glucuronidation",
                organ_damage=[],
            ),
            Metabolite(
                name="Carboxy-ibuprofen",
                toxicity="none",
                reactive=False,
                detox_pathway="Glucuronidation",
                organ_damage=[],
            ),
        ],
        phase2_pathways=["UGT (glucuronidation)"],
        phase2_fraction=0.35,
        gsh_requirement=0.0,
        phase3_transporters=["MRP2"],
        biliary_excretion=0.05,
        has_reactive_metabolite=False,
        toxic_dose_threshold_mg=3200,
    ),

    # ── DICLOFENAC — Forms reactive acyl glucuronide ──
    "OC(=O)CC1=CC=CC(NC2=CC=CC=C2Cl)=C1Cl": MetabolicPathway(
        drug_name="Diclofenac",
        smiles="OC(=O)CC1=CC=CC(NC2=CC=CC=C2Cl)=C1Cl",
        phase1_enzymes=["CYP2C9", "CYP3A4"],
        phase1_fraction=0.55,
        phase1_metabolites=[
            Metabolite(
                name="4'-Hydroxy-diclofenac",
                toxicity="low",
                reactive=False,
                detox_pathway="Glucuronidation",
                organ_damage=[],
            ),
            Metabolite(
                name="5-Hydroxy-diclofenac",
                toxicity="moderate",
                reactive=True,
                detox_pathway="GSH conjugation",
                organ_damage=["liver"],
            ),
        ],
        phase2_pathways=["UGT (acyl glucuronidation)"],
        phase2_fraction=0.40,
        gsh_requirement=0.3,
        phase3_transporters=["MRP2", "P-gp"],
        biliary_excretion=0.30,
        has_reactive_metabolite=True,
        toxic_dose_threshold_mg=400,
    ),

    # ── OMEPRAZOLE ──
    "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2": MetabolicPathway(
        drug_name="Omeprazole",
        smiles="CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2",
        phase1_enzymes=["CYP2C19", "CYP3A4"],
        phase1_fraction=0.85,
        phase1_metabolites=[
            Metabolite(
                name="5-Hydroxy-omeprazole",
                toxicity="none",
                reactive=False,
                detox_pathway="Further oxidation + excretion",
                organ_damage=[],
            ),
        ],
        phase2_pathways=["UGT"],
        phase2_fraction=0.10,
        gsh_requirement=0.0,
        phase3_transporters=["P-gp"],
        biliary_excretion=0.0,
        has_reactive_metabolite=False,
        toxic_dose_threshold_mg=2000,
    ),
}


# ════════════════════════════════════════════════════════════════
# HEPATIC METABOLISM ENGINE
# ════════════════════════════════════════════════════════════════

class HepaticMetabolismEngine:
    """
    Simulates hepatic drug metabolism through Phase I/II/III.

    Key outputs:
    - Metabolite profile (what the liver creates)
    - Reactive metabolite risk (can the metabolite damage tissue?)
    - GSH depletion risk (is glutathione being consumed?)
    - Effective hepatic clearance (accounting for all phases)
    """

    def __init__(self, smiles: str, dose_mg: float = 100.0):
        self.smiles = smiles
        self.dose_mg = dose_mg

        if smiles in METABOLIC_DATABASE:
            self.pathway = METABOLIC_DATABASE[smiles]
            self.known = True
        else:
            self.pathway = self._estimate_pathway(smiles)
            self.known = False

    def _estimate_pathway(self, smiles: str) -> MetabolicPathway:
        """Estimate metabolic pathway for unknown drugs based on structural features."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return MetabolicPathway(drug_name="Unknown", smiles=smiles)

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hba = Descriptors.NumHAcceptors(mol)
            n_aromatic = Descriptors.NumAromaticRings(mol)

            # Estimate Phase I enzymes from molecular properties
            phase1_enzymes = []
            if mw > 300 and logp > 2:
                phase1_enzymes.append("CYP3A4")  # Largest substrate selectivity
            if n_aromatic > 0:
                phase1_enzymes.append("CYP1A2")
            if logp > 1 and mw < 400:
                phase1_enzymes.append("CYP2C9")

            if not phase1_enzymes:
                phase1_enzymes = ["CYP3A4"]

            # Estimate Phase II from MW and polarity
            phase2 = []
            if hba > 2:
                phase2.append("UGT (glucuronidation)")
            if logp < 2:
                phase2.append("SULT (sulfation)")

            if not phase2:
                phase2 = ["UGT (glucuronidation)"]

            # Check for structural alerts that suggest reactive metabolites
            reactive_alerts = [
                ("[NH2]c1ccccc1", "Aniline → hydroxylamine (reactive)"),
                ("[N+](=O)[O-]", "Nitro → nitroso (reactive)"),
                ("C=CC=O", "Michael acceptor → GSH conjugation"),
                ("c1ccsc1", "Thiophene → S-oxide (reactive)"),
            ]

            has_reactive = False
            for smarts, _ in reactive_alerts:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    has_reactive = True
                    break

            return MetabolicPathway(
                drug_name="Unknown",
                smiles=smiles,
                phase1_enzymes=phase1_enzymes,
                phase1_fraction=0.6 if logp > 2 else 0.4,
                phase2_pathways=phase2,
                phase2_fraction=0.3,
                gsh_requirement=0.5 if has_reactive else 0.1,
                phase3_transporters=["P-gp"] if mw > 400 else [],
                biliary_excretion=0.1 if mw > 400 else 0.02,
                has_reactive_metabolite=has_reactive,
            )

        except ImportError:
            return MetabolicPathway(drug_name="Unknown", smiles=smiles)

    def assess_metabolism(self, liver_conc_mg_l: float = 0.0) -> dict:
        """
        Full metabolic assessment at current liver concentration.

        Returns detailed breakdown of Phase I/II/III with risk assessment.
        """
        pw = self.pathway

        # ── Phase I Assessment ──
        phase1_risk = "none"
        reactive_metabolites = []
        for met in pw.phase1_metabolites:
            if met.reactive:
                reactive_metabolites.append({
                    "name": met.name,
                    "toxicity": met.toxicity,
                    "detox": met.detox_pathway,
                    "organs_at_risk": met.organ_damage,
                })
                if met.toxicity in ("high", "critical"):
                    phase1_risk = "high"
                elif met.toxicity == "moderate" and phase1_risk != "high":
                    phase1_risk = "moderate"

        # ── GSH Depletion Check ──
        # At high doses, glutathione can be depleted → reactive metabolites
        # accumulate → tissue damage
        gsh_depletion = 0.0
        gsh_status = "normal"
        gsh_warning = None

        if pw.gsh_requirement > 0 and self.dose_mg > 0:
            # GSH depletion is dose-dependent
            # Normal GSH pool is ~7 mmol in a 70kg adult
            # Paracetamol depletes ~30% of GSH at 2g dose
            gsh_depletion = min(1.0, (self.dose_mg / pw.toxic_dose_threshold_mg) * pw.gsh_requirement)

            if gsh_depletion > 0.7:
                gsh_status = "critically_depleted"
                gsh_warning = (f"CRITICAL: Glutathione depleted to {(1-gsh_depletion)*100:.0f}% of normal. "
                               f"Reactive metabolites ({reactive_metabolites[0]['name'] if reactive_metabolites else 'unknown'}) "
                               f"cannot be detoxified. Immediate NAC (N-acetylcysteine) recommended.")
            elif gsh_depletion > 0.5:
                gsh_status = "depleted"
                gsh_warning = (f"WARNING: Glutathione depleted to {(1-gsh_depletion)*100:.0f}% of normal. "
                               f"Risk of reactive metabolite accumulation increasing.")
            elif gsh_depletion > 0.3:
                gsh_status = "reduced"
                gsh_warning = f"Glutathione reduced to {(1-gsh_depletion)*100:.0f}%. Monitor liver function."

        # ── Phase II Assessment ──
        phase2_adequate = gsh_depletion < 0.5  # Phase II can handle load if GSH adequate

        # ── Phase III Assessment ──
        # Biliary excretion for large/conjugated molecules
        biliary_risk = "none"
        if pw.biliary_excretion > 0.3:
            biliary_risk = "possible cholestasis risk at high doses"

        # ── Overall Metabolic Risk ──
        if gsh_status == "critically_depleted" and pw.has_reactive_metabolite:
            overall_risk = "critical"
            overall_message = (f"HEPATOTOXICITY RISK: {pw.drug_name} at {self.dose_mg}mg produces "
                               f"reactive metabolites that exceed the liver's detoxification capacity. "
                               f"This is the mechanism of {pw.drug_name} overdose toxicity.")
        elif gsh_status == "depleted" and pw.has_reactive_metabolite:
            overall_risk = "high"
            overall_message = (f"Elevated liver damage risk. {pw.drug_name} dose ({self.dose_mg}mg) "
                               f"is approaching the toxic threshold ({pw.toxic_dose_threshold_mg}mg). "
                               f"Reactive metabolite detoxification is compromised.")
        elif pw.has_reactive_metabolite and self.dose_mg > pw.toxic_dose_threshold_mg * 0.5:
            overall_risk = "moderate"
            overall_message = (f"Moderate hepatic stress. {pw.drug_name} produces reactive metabolites "
                               f"via {', '.join(pw.phase1_enzymes)}. Current dose is within therapeutic range "
                               f"but approaching maximum safe dose.")
        elif pw.has_reactive_metabolite:
            overall_risk = "low"
            overall_message = (f"{pw.drug_name} produces reactive metabolites but at therapeutic doses "
                               f"({self.dose_mg}mg), the liver's GSH pool can handle detoxification.")
        else:
            overall_risk = "minimal"
            overall_message = (f"{pw.drug_name} is metabolized safely via "
                               f"{', '.join(pw.phase1_enzymes + pw.phase2_pathways)}. "
                               f"No reactive metabolites detected.")

        return {
            "drug_name": pw.drug_name,
            "known_pathway": self.known,
            "dose_mg": self.dose_mg,
            "toxic_threshold_mg": pw.toxic_dose_threshold_mg,
            "dose_fraction_of_toxic": round(self.dose_mg / pw.toxic_dose_threshold_mg, 3),

            "phase1": {
                "enzymes": pw.phase1_enzymes,
                "fraction": pw.phase1_fraction,
                "metabolites": [{"name": m.name, "toxicity": m.toxicity, "reactive": m.reactive}
                                for m in pw.phase1_metabolites],
                "reactive_risk": phase1_risk,
            },

            "phase2": {
                "pathways": pw.phase2_pathways,
                "fraction": pw.phase2_fraction,
                "adequate_capacity": phase2_adequate,
                "gsh_depletion": round(gsh_depletion, 3),
                "gsh_status": gsh_status,
                "gsh_warning": gsh_warning,
            },

            "phase3": {
                "transporters": pw.phase3_transporters,
                "biliary_excretion": pw.biliary_excretion,
                "risk": biliary_risk,
            },

            "reactive_metabolites": reactive_metabolites,
            "overall_risk": overall_risk,
            "clinical_message": overall_message,
        }


# ── Standalone test ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Hepatic Metabolism Engine — Test")
    print("=" * 60)

    # Test 1: Paracetamol at therapeutic dose
    print("\n--- Paracetamol 500mg (therapeutic) ---")
    eng = HepaticMetabolismEngine("CC(=O)NC1=CC=C(C=C1)O", dose_mg=500)
    result = eng.assess_metabolism()
    print(f"  Risk: {result['overall_risk']}")
    print(f"  GSH: {result['phase2']['gsh_status']} ({result['phase2']['gsh_depletion']*100:.0f}% depleted)")
    print(f"  Message: {result['clinical_message']}")

    # Test 2: Paracetamol at toxic dose
    print("\n--- Paracetamol 8000mg (OVERDOSE) ---")
    eng2 = HepaticMetabolismEngine("CC(=O)NC1=CC=C(C=C1)O", dose_mg=8000)
    result2 = eng2.assess_metabolism()
    print(f"  Risk: {result2['overall_risk']}")
    print(f"  GSH: {result2['phase2']['gsh_status']} ({result2['phase2']['gsh_depletion']*100:.0f}% depleted)")
    print(f"  Reactive metabolites: {[m['name'] for m in result2['reactive_metabolites']]}")
    print(f"  Message: {result2['clinical_message']}")
    if result2['phase2']['gsh_warning']:
        print(f"  GSH Warning: {result2['phase2']['gsh_warning']}")

    # Test 3: Caffeine (safe metabolism)
    print("\n--- Caffeine 200mg ---")
    eng3 = HepaticMetabolismEngine("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", dose_mg=200)
    result3 = eng3.assess_metabolism()
    print(f"  Risk: {result3['overall_risk']}")
    print(f"  Phase I enzymes: {result3['phase1']['enzymes']}")
    print(f"  Message: {result3['clinical_message']}")

    print("\nHepatic metabolism test PASSED!")
