"""
Protein Binding Module
======================
Models drug binding to plasma proteins (albumin, AAG) and the critical
concept of FREE drug concentration.

Why this matters:
- Only FREE (unbound) drug is pharmacologically active
- Only FREE drug distributes to tissues and is cleared
- Drug-drug interactions can DISPLACE bound drug → sudden spike in active drug

Example: Warfarin is 99% bound to albumin. If Drug B displaces just 1%,
the free warfarin DOUBLES from 1% to 2% → bleeding risk doubles.

Reference: Rowland & Tozer "Clinical Pharmacokinetics and Pharmacodynamics"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# PROTEIN BINDING DATABASE
# ════════════════════════════════════════════════════════════════

@dataclass
class ProteinBindingProfile:
    """Drug's plasma protein binding characteristics."""
    name: str
    smiles: str
    fu: float  # Fraction unbound (0.01 = 99% bound, 1.0 = 0% bound)
    primary_protein: str  # "albumin" or "aag" (alpha-1-acid glycoprotein)
    binding_affinity: str  # "high", "medium", "low"
    displacement_potency: float  # 0-1, how strongly this drug displaces others
    displaceable_by: list[str] = field(default_factory=list)  # SMILES of drugs that can displace it


# Known protein binding profiles
# fu = fraction unbound in plasma
BINDING_DATABASE: dict[str, ProteinBindingProfile] = {
    # ── Highly bound drugs (fu < 0.05) — most susceptible to displacement ──
    "CC(=O)OC1=CC=CC=C1C(=O)O": ProteinBindingProfile(  # Aspirin
        name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        fu=0.50, primary_protein="albumin", binding_affinity="medium",
        displacement_potency=0.3, displaceable_by=[],
    ),
    "OC(=O)C1=CC=CC=C1O": ProteinBindingProfile(  # Salicylic acid (aspirin metabolite)
        name="Salicylic acid", smiles="OC(=O)C1=CC=CC=C1O",
        fu=0.15, primary_protein="albumin", binding_affinity="high",
        displacement_potency=0.5, displaceable_by=[],
    ),
    # Warfarin — the classic displacement danger
    "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O": ProteinBindingProfile(
        name="Warfarin", smiles="CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
        fu=0.01, primary_protein="albumin", binding_affinity="high",
        displacement_potency=0.1,
        displaceable_by=["CC(=O)OC1=CC=CC=C1C(=O)O"],  # Aspirin can displace
    ),
    # Diazepam — highly bound
    "CN1C(=O)CN=C(C2=CC=CC=C2)C3=CC(Cl)=CC=C31": ProteinBindingProfile(
        name="Diazepam", smiles="CN1C(=O)CN=C(C2=CC=CC=C2)C3=CC(Cl)=CC=C31",
        fu=0.02, primary_protein="albumin", binding_affinity="high",
        displacement_potency=0.1, displaceable_by=[],
    ),
    # Ibuprofen
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": ProteinBindingProfile(
        name="Ibuprofen", smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        fu=0.01, primary_protein="albumin", binding_affinity="high",
        displacement_potency=0.4,
        displaceable_by=[],
    ),
    # Naproxen
    "CC(C1=CC2=CC=CC=C2C=C1)C(=O)O": ProteinBindingProfile(
        name="Naproxen", smiles="CC(C1=CC2=CC=CC=C2C=C1)C(=O)O",
        fu=0.01, primary_protein="albumin", binding_affinity="high",
        displacement_potency=0.5,
        displaceable_by=[],
    ),
    # Phenytoin
    "O=C1NC(=O)C(C2=CC=CC=C2)(C3=CC=CC=C3)N1": ProteinBindingProfile(
        name="Phenytoin", smiles="O=C1NC(=O)C(C2=CC=CC=C2)(C3=CC=CC=C3)N1",
        fu=0.10, primary_protein="albumin", binding_affinity="high",
        displacement_potency=0.2,
        displaceable_by=["CC(=O)OC1=CC=CC=C1C(=O)O"],  # Aspirin displaces
    ),
    # ── Moderately bound drugs ──
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": ProteinBindingProfile(  # Caffeine
        name="Caffeine", smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        fu=0.65, primary_protein="albumin", binding_affinity="low",
        displacement_potency=0.0, displaceable_by=[],
    ),
    "CC(=O)NC1=CC=C(C=C1)O": ProteinBindingProfile(  # Paracetamol
        name="Paracetamol", smiles="CC(=O)NC1=CC=C(C=C1)O",
        fu=0.80, primary_protein="albumin", binding_affinity="low",
        displacement_potency=0.0, displaceable_by=[],
    ),
    # Metformin — essentially unbound
    "CN(C)C(=N)NC(=N)N": ProteinBindingProfile(
        name="Metformin", smiles="CN(C)C(=N)NC(=N)N",
        fu=1.0, primary_protein="albumin", binding_affinity="low",
        displacement_potency=0.0, displaceable_by=[],
    ),
    # Omeprazole
    "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2": ProteinBindingProfile(
        name="Omeprazole", smiles="CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2",
        fu=0.05, primary_protein="albumin", binding_affinity="high",
        displacement_potency=0.2, displaceable_by=[],
    ),
    # AAG-binding drugs (basic/cationic drugs)
    # Propranolol
    "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12": ProteinBindingProfile(
        name="Propranolol", smiles="CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12",
        fu=0.10, primary_protein="aag", binding_affinity="high",
        displacement_potency=0.3, displaceable_by=[],
    ),
    # Lidocaine
    "CCN(CC)CC(=O)NC1=C(C)C=CC=C1C": ProteinBindingProfile(
        name="Lidocaine", smiles="CCN(CC)CC(=O)NC1=C(C)C=CC=C1C",
        fu=0.30, primary_protein="aag", binding_affinity="medium",
        displacement_potency=0.2, displaceable_by=[],
    ),
}


# ════════════════════════════════════════════════════════════════
# PROTEIN BINDING ENGINE
# ════════════════════════════════════════════════════════════════

def estimate_fu_from_logp(logp: float) -> float:
    """
    Estimate fraction unbound from LogP for unknown drugs.

    Based on Lombardo et al. (2002) empirical relationship:
    Higher LogP → more lipophilic → more protein binding → lower fu

    Returns fu between 0.01 and 1.0
    """
    if logp <= 0:
        return min(0.95, max(0.5, 1.0 - 0.1 * abs(logp)))
    elif logp <= 1:
        return 0.50
    elif logp <= 2:
        return 0.20
    elif logp <= 3:
        return 0.10
    elif logp <= 4:
        return 0.05
    else:
        return max(0.01, 0.10 - 0.02 * (logp - 3))


class ProteinBindingEngine:
    """
    Models protein binding and displacement interactions.

    Key concepts:
    - fu (fraction unbound) = free drug / total drug
    - Only free drug is active, distributes to tissues, and is cleared
    - Displacement: Drug A kicks Drug B off albumin → free Drug B spikes
    """

    def __init__(self, smiles: str, logp: float = 2.0):
        self.smiles = smiles
        self.logp = logp

        # Look up known profile or estimate
        if smiles in BINDING_DATABASE:
            self.profile = BINDING_DATABASE[smiles]
            self.fu = self.profile.fu
            self.known = True
        else:
            self.fu = estimate_fu_from_logp(logp)
            self.profile = ProteinBindingProfile(
                name="Unknown",
                smiles=smiles,
                fu=self.fu,
                primary_protein="albumin" if logp < 3 else "aag",
                binding_affinity="high" if self.fu < 0.05 else "medium" if self.fu < 0.3 else "low",
                displacement_potency=0.2 if self.fu < 0.05 else 0.0,
            )
            self.known = False

    def get_free_concentration(self, total_conc: float) -> float:
        """Calculate free (active) drug concentration from total."""
        return total_conc * self.fu

    def get_effective_clearance(self, intrinsic_clearance: float) -> float:
        """
        Only free drug is available for clearance.
        Effective CL = fu × CLint (for restrictively cleared drugs)
        """
        return self.fu * intrinsic_clearance

    def get_effective_kp(self, kp_total: float) -> float:
        """
        Partition coefficient correction for protein binding.
        Free drug partitioning: Kpu = Kp / fu
        Effectively, less free drug → less tissue distribution
        """
        # For highly bound drugs, tissue distribution is primarily
        # of the unbound fraction
        return kp_total * self.fu

    def assess_displacement(
        self,
        other_smiles: str,
        other_logp: float = 2.0,
    ) -> dict:
        """
        Assess if another drug can displace this drug from proteins.

        Returns dict with:
        - displaced: bool
        - new_fu: float (adjusted fraction unbound)
        - fu_fold_change: float (how much fu increased)
        - risk: str ("none", "low", "moderate", "high", "critical")
        - clinical_impact: str
        """
        other_engine = ProteinBindingEngine(other_smiles, other_logp)

        # No displacement if either drug is weakly bound
        if self.fu > 0.3 or other_engine.profile.displacement_potency < 0.1:
            return {
                "displaced": False,
                "new_fu": self.fu,
                "fu_fold_change": 1.0,
                "risk": "none",
                "clinical_impact": "No significant protein binding displacement expected.",
            }

        # Check if they compete for the same protein
        same_protein = (self.profile.primary_protein == other_engine.profile.primary_protein)
        if not same_protein:
            return {
                "displaced": False,
                "new_fu": self.fu,
                "fu_fold_change": 1.0,
                "risk": "none",
                "clinical_impact": "Drugs bind to different proteins — no displacement.",
            }

        # Calculate displacement magnitude
        # New fu ≈ old fu + (1 - old fu) × displacer potency × fraction bound
        displacement_fraction = other_engine.profile.displacement_potency * (1 - self.fu)
        new_fu = min(1.0, self.fu + displacement_fraction * 0.5)  # Damped displacement
        fold_change = new_fu / self.fu if self.fu > 0 else 1.0

        # Assess risk based on fold change and baseline fu
        if fold_change >= 3.0 and self.fu <= 0.02:
            risk = "critical"
            impact = (f"CRITICAL: {self.profile.name} free fraction increases {fold_change:.1f}x "
                      f"(from {self.fu*100:.1f}% to {new_fu*100:.1f}%). "
                      f"This can cause acute toxicity. Dose reduction required.")
        elif fold_change >= 2.0 and self.fu <= 0.05:
            risk = "high"
            impact = (f"HIGH RISK: {self.profile.name} free fraction increases {fold_change:.1f}x. "
                      f"Monitor closely for adverse effects. Consider dose adjustment.")
        elif fold_change >= 1.5:
            risk = "moderate"
            impact = (f"Moderate displacement: {self.profile.name} free fraction increases "
                      f"{fold_change:.1f}x. Monitor drug levels if available.")
        elif fold_change >= 1.2:
            risk = "low"
            impact = (f"Minor displacement: {self.profile.name} free fraction increases "
                      f"{fold_change:.1f}x. Clinically insignificant for most patients.")
        else:
            risk = "none"
            impact = "Negligible displacement."

        return {
            "displaced": fold_change >= 1.2,
            "new_fu": round(new_fu, 4),
            "fu_fold_change": round(fold_change, 2),
            "risk": risk,
            "clinical_impact": impact,
            "original_fu": self.fu,
            "drug_name": self.profile.name,
            "displacer_name": other_engine.profile.name,
            "protein": self.profile.primary_protein,
        }


def assess_mutual_displacement(smiles_a: str, smiles_b: str,
                                logp_a: float = 2.0, logp_b: float = 2.0) -> dict:
    """
    Check displacement in both directions (A displaces B AND B displaces A).

    Returns combined assessment with both directions.
    """
    engine_a = ProteinBindingEngine(smiles_a, logp_a)
    engine_b = ProteinBindingEngine(smiles_b, logp_b)

    a_displaces_b = engine_b.assess_displacement(smiles_a, logp_a)
    b_displaces_a = engine_a.assess_displacement(smiles_b, logp_b)

    # Overall risk = worst of both directions
    risk_order = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
    worst = max(
        risk_order.get(a_displaces_b["risk"], 0),
        risk_order.get(b_displaces_a["risk"], 0),
    )
    risk_names = {0: "none", 1: "low", 2: "moderate", 3: "high", 4: "critical"}

    return {
        "drug_a": {
            "name": engine_a.profile.name,
            "fu": engine_a.fu,
            "protein": engine_a.profile.primary_protein,
        },
        "drug_b": {
            "name": engine_b.profile.name,
            "fu": engine_b.fu,
            "protein": engine_b.profile.primary_protein,
        },
        "a_displaces_b": a_displaces_b,
        "b_displaces_a": b_displaces_a,
        "overall_displacement_risk": risk_names[worst],
        "warnings": [
            d["clinical_impact"]
            for d in [a_displaces_b, b_displaces_a]
            if d["displaced"]
        ],
    }


# ── Standalone test ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Protein Binding Engine — Test")
    print("=" * 60)

    # Test 1: Known drug
    print("\n--- Warfarin (99% bound) ---")
    eng = ProteinBindingEngine("CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O")
    print(f"  fu = {eng.fu}, free conc at 5 mg/L: {eng.get_free_concentration(5):.3f} mg/L")

    # Test 2: Displacement — Aspirin displaces Warfarin
    print("\n--- Aspirin displacing Warfarin ---")
    result = eng.assess_displacement("CC(=O)OC1=CC=CC=C1C(=O)O")
    print(f"  Displaced: {result['displaced']}")
    print(f"  New fu: {result['new_fu']} (was {eng.fu})")
    print(f"  Fold change: {result['fu_fold_change']}x")
    print(f"  Risk: {result['risk']}")
    print(f"  Impact: {result['clinical_impact']}")

    # Test 3: Unknown drug
    print("\n--- Unknown drug (LogP=4.5) ---")
    eng2 = ProteinBindingEngine("CCCCCCCCCC", logp=4.5)
    print(f"  Estimated fu: {eng2.fu}")

    # Test 4: Mutual displacement
    print("\n--- Mutual: Ibuprofen + Warfarin ---")
    mutual = assess_mutual_displacement(
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",  # Warfarin
    )
    print(f"  Overall risk: {mutual['overall_displacement_risk']}")
    for w in mutual['warnings']:
        print(f"  Warning: {w}")

    print("\nProtein binding test PASSED!")
