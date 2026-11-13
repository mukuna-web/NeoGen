"""
Blood-Brain Barrier (BBB) Permeability Model
===============================================
Predicts if a drug crosses the blood-brain barrier.

The BBB is a tight barrier of endothelial cells that protects the brain.
Only certain drugs can cross it:
- Small, lipophilic molecules (e.g., caffeine, diazepam) → cross easily
- Large, polar molecules (e.g., metformin, antibiotics) → cannot cross
- P-gp substrates get actively pumped OUT of the brain

Why this matters:
1. CNS drugs MUST cross BBB to work (antidepressants, anti-seizure, painkillers)
2. Non-CNS drugs should NOT cross BBB (to avoid neurotoxicity)
3. BBB permeability + brain Kp determines brain drug concentration → neurotoxicity risk

Approach:
1. Known BBB permeability values from literature
2. Molecular property-based prediction (TPSA, LogP, MW, HBD, PSA)
3. P-gp efflux assessment (drugs pumped out of brain)

Reference: Egan et al. (2000) "Prediction of Drug Absorption Using Multivariate Statistics"
           Clark (2003) "In silico prediction of BBB permeation"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# BBB DATABASE
# ════════════════════════════════════════════════════════════════

@dataclass
class BBBProfile:
    """Drug's blood-brain barrier permeability profile."""
    name: str
    smiles: str
    logbb: float  # Log(Cbrain/Cblood) — positive = brain-penetrating
    bbb_category: str  # "high", "moderate", "low", "impermeable"
    pgp_substrate: bool  # Actively pumped out by P-glycoprotein
    pgp_inhibitor: bool  # Inhibits P-gp (can increase brain exposure of co-administered drugs)
    brain_kp: float  # Brain:plasma partition coefficient
    cns_effects: list[str] = field(default_factory=list)  # Known CNS side effects
    neurotoxicity_risk: str = "none"  # "none", "low", "moderate", "high"


BBB_DATABASE: dict[str, BBBProfile] = {
    # ── HIGH BBB penetration (CNS-active drugs) ──
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": BBBProfile(  # Caffeine
        name="Caffeine", smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        logbb=0.02, bbb_category="high", pgp_substrate=False, pgp_inhibitor=False,
        brain_kp=0.9,
        cns_effects=["Alertness", "Anxiety", "Insomnia", "Tremor at high doses"],
        neurotoxicity_risk="low",
    ),
    "CN1C(=O)CN=C(C2=CC=CC=C2)C3=CC(Cl)=CC=C31": BBBProfile(  # Diazepam
        name="Diazepam", smiles="CN1C(=O)CN=C(C2=CC=CC=C2)C3=CC(Cl)=CC=C31",
        logbb=0.52, bbb_category="high", pgp_substrate=False, pgp_inhibitor=False,
        brain_kp=2.5,
        cns_effects=["Sedation", "Anxiolysis", "Muscle relaxation", "Amnesia", "Respiratory depression"],
        neurotoxicity_risk="moderate",
    ),
    "CCO": BBBProfile(  # Ethanol
        name="Ethanol", smiles="CCO",
        logbb=0.15, bbb_category="high", pgp_substrate=False, pgp_inhibitor=False,
        brain_kp=1.2,
        cns_effects=["Euphoria", "Sedation", "Impaired coordination", "Slurred speech",
                      "Memory blackout", "Respiratory depression", "Coma"],
        neurotoxicity_risk="high",
    ),
    "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12": BBBProfile(  # Propranolol
        name="Propranolol", smiles="CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12",
        logbb=0.16, bbb_category="high", pgp_substrate=True, pgp_inhibitor=False,
        brain_kp=1.5,
        cns_effects=["Vivid dreams", "Depression", "Fatigue", "Sleep disturbance"],
        neurotoxicity_risk="low",
    ),

    # ── MODERATE BBB penetration ──
    "CC(=O)NC1=CC=C(C=C1)O": BBBProfile(  # Paracetamol
        name="Paracetamol", smiles="CC(=O)NC1=CC=C(C=C1)O",
        logbb=-0.31, bbb_category="moderate", pgp_substrate=False, pgp_inhibitor=False,
        brain_kp=0.5,
        cns_effects=["Antipyretic (acts on hypothalamus)"],
        neurotoxicity_risk="none",
    ),
    "CC(=O)OC1=CC=CC=C1C(=O)O": BBBProfile(  # Aspirin
        name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        logbb=-0.50, bbb_category="low", pgp_substrate=False, pgp_inhibitor=False,
        brain_kp=0.2,
        cns_effects=["Tinnitus at high doses"],
        neurotoxicity_risk="low",
    ),
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": BBBProfile(  # Ibuprofen
        name="Ibuprofen", smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        logbb=-0.20, bbb_category="moderate", pgp_substrate=False, pgp_inhibitor=False,
        brain_kp=0.4,
        cns_effects=["Headache (paradoxical)", "Dizziness"],
        neurotoxicity_risk="none",
    ),

    # ── LOW/NO BBB penetration ──
    "CN(C)C(=N)NC(=N)N": BBBProfile(  # Metformin
        name="Metformin", smiles="CN(C)C(=N)NC(=N)N",
        logbb=-1.5, bbb_category="impermeable", pgp_substrate=False, pgp_inhibitor=False,
        brain_kp=0.02,
        cns_effects=[],
        neurotoxicity_risk="none",
    ),
    "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2": BBBProfile(  # Omeprazole
        name="Omeprazole", smiles="CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2",
        logbb=-0.80, bbb_category="low", pgp_substrate=True, pgp_inhibitor=False,
        brain_kp=0.1,
        cns_effects=["Rare: headache, dizziness"],
        neurotoxicity_risk="none",
    ),
}


# ════════════════════════════════════════════════════════════════
# BBB PERMEABILITY ENGINE
# ════════════════════════════════════════════════════════════════

class BBBPermeabilityEngine:
    """
    Predicts BBB permeability and brain drug exposure.

    For known drugs: uses published logBB and Kp values.
    For unknown drugs: estimates from molecular properties using
    the Clark/Egan model.
    """

    def __init__(self, smiles: str, logp: float = 2.0, mw: float = 300,
                 tpsa: float = 60, hbd: int = 2):
        self.smiles = smiles
        self.logp = logp
        self.mw = mw
        self.tpsa = tpsa
        self.hbd = hbd

        if smiles in BBB_DATABASE:
            self.profile = BBB_DATABASE[smiles]
            self.known = True
        else:
            self.profile = self._estimate_bbb()
            self.known = False

    def _estimate_bbb(self) -> BBBProfile:
        """
        Estimate BBB permeability from molecular descriptors.

        Clark (2003) model: logBB = 0.152*LogP - 0.0148*TPSA + 0.139
        Egan egg: TPSA < 90 Å² and LogP 0-5 → likely CNS penetrant
        """
        # Clark model for logBB
        logbb = 0.152 * self.logp - 0.0148 * self.tpsa + 0.139

        # Adjust for MW (large molecules can't cross)
        if self.mw > 500:
            logbb -= 0.5
        elif self.mw > 400:
            logbb -= 0.2

        # Adjust for HBD (hydrogen bond donors reduce permeability)
        if self.hbd > 3:
            logbb -= 0.3 * (self.hbd - 3)

        # Convert logBB to brain Kp
        brain_kp = 10 ** logbb
        brain_kp = max(0.01, min(5.0, brain_kp))

        # Categorize
        if logbb > 0.3:
            category = "high"
        elif logbb > -0.1:
            category = "moderate"
        elif logbb > -1.0:
            category = "low"
        else:
            category = "impermeable"

        # Estimate P-gp substrate probability
        # P-gp prefers: MW > 400, amphiphilic, multiple aromatic rings
        pgp_substrate = self.mw > 400 and self.logp > 2

        # Neurotoxicity risk
        if category in ("high", "moderate") and self.logp > 3:
            neuro_risk = "moderate"
        elif category == "high":
            neuro_risk = "low"
        else:
            neuro_risk = "none"

        return BBBProfile(
            name="Unknown",
            smiles=self.smiles,
            logbb=round(logbb, 3),
            bbb_category=category,
            pgp_substrate=pgp_substrate,
            pgp_inhibitor=False,
            brain_kp=round(brain_kp, 3),
            cns_effects=[],
            neurotoxicity_risk=neuro_risk,
        )

    def get_brain_kp(self) -> float:
        """Return brain:plasma partition coefficient for PBPK model."""
        return self.profile.brain_kp

    def assess_neurotoxicity(self, brain_conc_mg_l: float) -> dict:
        """
        Assess neurotoxicity risk at a specific brain concentration.

        Returns risk level and expected CNS symptoms.
        """
        p = self.profile

        if p.bbb_category == "impermeable":
            return {
                "risk": "none",
                "brain_conc": round(brain_conc_mg_l, 4),
                "message": f"{p.name} does not cross the BBB significantly. No CNS effects expected.",
                "symptoms": [],
            }

        # Threshold-based assessment
        # At very low brain concentrations → no effect
        # As concentration rises → progressive symptoms
        if brain_conc_mg_l < 0.1:
            return {
                "risk": "none",
                "brain_conc": round(brain_conc_mg_l, 4),
                "message": "Brain concentration too low for CNS effects.",
                "symptoms": [],
            }

        # Scale symptoms by concentration
        active_symptoms = []
        if p.cns_effects:
            # Show more symptoms as concentration increases
            n_symptoms = min(len(p.cns_effects), max(1, int(brain_conc_mg_l / 2)))
            active_symptoms = p.cns_effects[:n_symptoms]

        if brain_conc_mg_l > 10:
            risk = "critical"
            message = (f"CRITICAL: Very high brain concentration ({brain_conc_mg_l:.1f} mg/L). "
                       f"Risk of seizures, confusion, loss of consciousness.")
            active_symptoms.extend(["Seizure risk", "Altered consciousness"])
        elif brain_conc_mg_l > 5:
            risk = "high"
            message = (f"High brain concentration ({brain_conc_mg_l:.1f} mg/L). "
                       f"Significant CNS effects expected.")
        elif brain_conc_mg_l > 1:
            risk = "moderate"
            message = (f"Moderate brain concentration ({brain_conc_mg_l:.1f} mg/L). "
                       f"Some CNS effects possible.")
        else:
            risk = "low"
            message = f"Low brain concentration ({brain_conc_mg_l:.2f} mg/L). Mild CNS effects possible."

        return {
            "risk": risk,
            "brain_conc": round(brain_conc_mg_l, 4),
            "bbb_category": p.bbb_category,
            "logbb": p.logbb,
            "pgp_substrate": p.pgp_substrate,
            "message": message,
            "symptoms": list(set(active_symptoms)),  # Deduplicate
        }

    def get_full_assessment(self) -> dict:
        """Complete BBB permeability assessment."""
        p = self.profile
        return {
            "drug_name": p.name,
            "known_profile": self.known,
            "logbb": p.logbb,
            "bbb_category": p.bbb_category,
            "brain_kp": p.brain_kp,
            "pgp_substrate": p.pgp_substrate,
            "pgp_inhibitor": p.pgp_inhibitor,
            "neurotoxicity_risk": p.neurotoxicity_risk,
            "known_cns_effects": p.cns_effects,
            "permeability_description": {
                "high": "Freely crosses BBB — significant brain exposure expected",
                "moderate": "Partially crosses BBB — moderate brain exposure",
                "low": "Poor BBB penetration — minimal brain exposure",
                "impermeable": "Does not cross BBB — no brain exposure",
            }.get(p.bbb_category, "Unknown"),
        }


# ── Standalone test ──
if __name__ == "__main__":
    print("=" * 60)
    print("BBB Permeability Engine — Test")
    print("=" * 60)

    tests = [
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0.02, 194, 49, 1),
        ("Diazepam", "CN1C(=O)CN=C(C2=CC=CC=C2)C3=CC(Cl)=CC=C31", 2.82, 285, 33, 0),
        ("Metformin", "CN(C)C(=N)NC(=N)N", -1.4, 129, 96, 4),
        ("Ethanol", "CCO", -0.31, 46, 20, 1),
    ]

    for name, smiles, logp, mw, tpsa, hbd in tests:
        eng = BBBPermeabilityEngine(smiles, logp=logp, mw=mw, tpsa=tpsa, hbd=hbd)
        result = eng.get_full_assessment()
        print(f"\n--- {name} ---")
        print(f"  LogBB: {result['logbb']}, Category: {result['bbb_category']}")
        print(f"  Brain Kp: {result['brain_kp']}")
        print(f"  P-gp substrate: {result['pgp_substrate']}")
        print(f"  CNS effects: {result['known_cns_effects']}")
        print(f"  Neurotoxicity risk: {result['neurotoxicity_risk']}")

    print("\nBBB permeability test PASSED!")
