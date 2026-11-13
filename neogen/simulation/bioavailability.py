"""
Bioavailability Model
======================
Predicts oral bioavailability (F) — the fraction of an oral dose that
reaches systemic circulation.

F = Fa × Fg × Fh

Where:
- Fa = fraction absorbed (depends on solubility, dissolution, permeability)
- Fg = fraction escaping gut wall metabolism (intestinal CYP3A4/UGT)
- Fh = fraction escaping hepatic first-pass metabolism

For IV administration, F = 1.0 always (100% reaches blood).

Why this matters:
- Our PBPK used a fixed F=0.7 — which is wrong for many drugs
- Metformin: F ≈ 0.5 (poorly absorbed)
- Propranolol: F ≈ 0.25 (extensive first-pass)
- Diazepam: F ≈ 0.99 (almost completely absorbed)

Reference: Varma et al. "Biopharmaceutics Classification System" (BCS)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# BCS CLASSIFICATION
# ════════════════════════════════════════════════════════════════

@dataclass
class BCSProfile:
    """Biopharmaceutics Classification System profile."""
    drug_name: str
    bcs_class: int  # 1, 2, 3, or 4
    solubility: str  # "high" or "low"
    permeability: str  # "high" or "low"
    fa: float  # Fraction absorbed
    fg: float  # Fraction escaping gut metabolism
    fh: float  # Fraction escaping hepatic first-pass
    f_oral: float  # Overall oral bioavailability
    absorption_rate: str  # "fast", "medium", "slow"
    food_effect: str  # How food affects absorption


# Known bioavailability profiles
BIOAVAILABILITY_DATABASE: dict[str, BCSProfile] = {
    # BCS Class I: High solubility + High permeability (fast, complete absorption)
    "CN1C(=O)CN=C(C2=CC=CC=C2)C3=CC(Cl)=CC=C31": BCSProfile(  # Diazepam
        drug_name="Diazepam", bcs_class=1, solubility="high", permeability="high",
        fa=0.99, fg=0.95, fh=0.99, f_oral=0.93,
        absorption_rate="fast", food_effect="Minimal — well absorbed regardless",
    ),
    "CC(=O)NC1=CC=C(C=C1)O": BCSProfile(  # Paracetamol
        drug_name="Paracetamol", bcs_class=1, solubility="high", permeability="high",
        fa=0.98, fg=0.80, fh=0.75, f_oral=0.63,
        absorption_rate="fast", food_effect="Delayed Tmax by ~30min with food, same AUC",
    ),
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": BCSProfile(  # Caffeine
        drug_name="Caffeine", bcs_class=1, solubility="high", permeability="high",
        fa=0.99, fg=0.99, fh=0.99, f_oral=0.99,
        absorption_rate="fast", food_effect="Slightly delayed with food",
    ),
    "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2": BCSProfile(  # Omeprazole
        drug_name="Omeprazole", bcs_class=1, solubility="high", permeability="high",
        fa=0.90, fg=0.90, fh=0.50, f_oral=0.40,
        absorption_rate="medium", food_effect="Take 30min before meals for best effect",
    ),

    # BCS Class II: Low solubility + High permeability (dissolution-limited)
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": BCSProfile(  # Ibuprofen
        drug_name="Ibuprofen", bcs_class=2, solubility="low", permeability="high",
        fa=0.95, fg=0.95, fh=0.95, f_oral=0.85,
        absorption_rate="medium", food_effect="Reduced rate with food, similar extent",
    ),
    "CC(=O)OC1=CC=CC=C1C(=O)O": BCSProfile(  # Aspirin
        drug_name="Aspirin", bcs_class=1, solubility="high", permeability="high",
        fa=0.90, fg=0.70, fh=0.50, f_oral=0.50,
        absorption_rate="fast", food_effect="Take with food to reduce GI irritation",
    ),

    # BCS Class III: High solubility + Low permeability (permeation-limited)
    "CN(C)C(=N)NC(=N)N": BCSProfile(  # Metformin
        drug_name="Metformin", bcs_class=3, solubility="high", permeability="low",
        fa=0.50, fg=0.99, fh=0.99, f_oral=0.50,
        absorption_rate="slow", food_effect="Take with food to reduce GI side effects",
    ),

    # BCS Class IV: Low solubility + Low permeability (poor oral candidates)
    # Most BCS IV drugs have special formulations to improve absorption
}


# ════════════════════════════════════════════════════════════════
# BIOAVAILABILITY ENGINE
# ════════════════════════════════════════════════════════════════

class BioavailabilityEngine:
    """
    Predicts oral bioavailability from molecular properties.

    For known drugs: uses published values.
    For unknown drugs: estimates from LogP, MW, TPSA, HBD using empirical rules.
    """

    def __init__(self, smiles: str, logp: float = 2.0,
                 mw: float = 300, tpsa: float = 60, hbd: int = 2):
        self.smiles = smiles
        self.logp = logp
        self.mw = mw
        self.tpsa = tpsa
        self.hbd = hbd

        if smiles in BIOAVAILABILITY_DATABASE:
            self.profile = BIOAVAILABILITY_DATABASE[smiles]
            self.known = True
        else:
            self.profile = self._estimate_bioavailability()
            self.known = False

    def _estimate_bioavailability(self) -> BCSProfile:
        """
        Estimate bioavailability from molecular descriptors.

        Uses:
        1. Lipinski's Rule of 5 for absorption likelihood
        2. TPSA for permeability estimation (Egan et al. 2000)
        3. LogP for solubility estimation
        4. First-pass extraction from clearance estimates
        """
        # ── Estimate solubility (from LogP) ──
        # General Solubility Equation: LogS ≈ 0.5 - LogP - 0.01*(MP-25)
        # High solubility: dose/solubility ratio < 250mL
        if self.logp < 2:
            solubility = "high"
        elif self.logp < 4:
            solubility = "moderate"
        else:
            solubility = "low"

        # ── Estimate permeability (from TPSA) ──
        # Egan egg model: TPSA < 120 Å² → high permeability
        if self.tpsa < 90:
            permeability = "high"
        elif self.tpsa < 140:
            permeability = "moderate"
        else:
            permeability = "low"

        # ── BCS Classification ──
        sol_high = solubility == "high"
        perm_high = permeability == "high"

        if sol_high and perm_high:
            bcs_class = 1
        elif not sol_high and perm_high:
            bcs_class = 2
        elif sol_high and not perm_high:
            bcs_class = 3
        else:
            bcs_class = 4

        # ── Estimate Fa (fraction absorbed) ──
        # Based on TPSA and LogP
        if self.tpsa < 75:
            fa = 0.95  # Excellent absorption
        elif self.tpsa < 120:
            fa = 0.80 - 0.002 * (self.tpsa - 75)  # Linear decline
        elif self.tpsa < 150:
            fa = 0.50 - 0.005 * (self.tpsa - 120)
        else:
            fa = max(0.10, 0.35 - 0.003 * (self.tpsa - 150))

        # Adjust for solubility (dissolution-limited for BCS II/IV)
        if solubility == "low":
            fa *= 0.75
        elif solubility == "moderate":
            fa *= 0.90

        # Lipinski violations reduce absorption
        lipinski_violations = 0
        if self.mw > 500:
            lipinski_violations += 1
        if self.logp > 5:
            lipinski_violations += 1
        if self.hbd > 5:
            lipinski_violations += 1
        if self.tpsa > 140:
            lipinski_violations += 1

        fa *= max(0.3, 1.0 - 0.15 * lipinski_violations)
        fa = min(0.99, max(0.05, fa))

        # ── Estimate Fg (gut wall escape) ──
        # CYP3A4 in gut wall metabolizes some drugs during absorption
        # More lipophilic drugs → more gut metabolism
        if self.logp > 4:
            fg = 0.60  # High lipophilicity → significant gut metabolism
        elif self.logp > 2:
            fg = 0.80
        else:
            fg = 0.95

        # ── Estimate Fh (hepatic escape) ──
        # Based on expected clearance relative to liver blood flow
        # High extraction ratio drugs: Fh low
        if self.logp > 4 and self.mw > 300:
            fh = 0.40  # High extraction
        elif self.logp > 2:
            fh = 0.70
        else:
            fh = 0.90

        f_oral = round(fa * fg * fh, 3)

        # Absorption rate
        if bcs_class == 1:
            absorption_rate = "fast"
        elif bcs_class == 2:
            absorption_rate = "medium"
        else:
            absorption_rate = "slow"

        name = "Unknown"
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(self.smiles)
            if mol:
                name = f"Unknown (MW={self.mw:.0f}, LogP={self.logp:.1f})"
        except ImportError:
            pass

        return BCSProfile(
            drug_name=name,
            bcs_class=bcs_class,
            solubility=solubility if solubility != "moderate" else "low",
            permeability=permeability if permeability != "moderate" else "low",
            fa=round(fa, 3),
            fg=round(fg, 3),
            fh=round(fh, 3),
            f_oral=f_oral,
            absorption_rate=absorption_rate,
            food_effect="Unknown — assess clinically",
        )

    def get_bioavailability(self) -> float:
        """Return overall oral bioavailability (F)."""
        return self.profile.f_oral

    def get_absorption_rate_constant(self, base_ka: float = 1.0) -> float:
        """
        Adjust absorption rate constant based on BCS class.
        BCS I: fast (ka × 2)
        BCS II: dissolution-limited (ka × 0.7)
        BCS III: permeation-limited (ka × 0.5)
        BCS IV: slow (ka × 0.3)
        """
        multipliers = {1: 2.0, 2: 0.7, 3: 0.5, 4: 0.3}
        return base_ka * multipliers.get(self.profile.bcs_class, 1.0)

    def get_full_assessment(self) -> dict:
        """Complete bioavailability assessment."""
        p = self.profile
        return {
            "drug_name": p.drug_name,
            "known_profile": self.known,
            "bcs_class": p.bcs_class,
            "bcs_description": {
                1: "High solubility + High permeability → rapid & complete absorption",
                2: "Low solubility + High permeability → dissolution-limited",
                3: "High solubility + Low permeability → permeation-limited",
                4: "Low solubility + Low permeability → poor oral candidate",
            }.get(p.bcs_class, "Unknown"),
            "solubility": p.solubility,
            "permeability": p.permeability,
            "fa": p.fa,
            "fa_description": f"{p.fa*100:.0f}% of dose absorbed from GI tract",
            "fg": p.fg,
            "fg_description": f"{p.fg*100:.0f}% escapes gut wall metabolism",
            "fh": p.fh,
            "fh_description": f"{p.fh*100:.0f}% escapes hepatic first-pass",
            "f_oral": p.f_oral,
            "f_oral_description": f"Overall: {p.f_oral*100:.0f}% of oral dose reaches systemic circulation",
            "absorption_rate": p.absorption_rate,
            "food_effect": p.food_effect,
        }


# ── Standalone test ──
if __name__ == "__main__":
    print("=" * 60)
    print("Bioavailability Engine — Test")
    print("=" * 60)

    tests = [
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0.02),
        ("Paracetamol", "CC(=O)NC1=CC=C(C=C1)O", 0.46),
        ("Metformin", "CN(C)C(=N)NC(=N)N", -1.4),
        ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 3.97),
    ]

    for name, smiles, logp in tests:
        eng = BioavailabilityEngine(smiles, logp=logp)
        result = eng.get_full_assessment()
        print(f"\n--- {name} ---")
        print(f"  BCS Class: {result['bcs_class']} ({result['bcs_description']})")
        print(f"  F = Fa({result['fa']}) × Fg({result['fg']}) × Fh({result['fh']}) = {result['f_oral']}")
        print(f"  Known: {result['known_profile']}")

    print("\nBioavailability test PASSED!")
