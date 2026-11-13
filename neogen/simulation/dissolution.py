"""
Advanced Dissolution & Absorption Model (ADAM-like)
====================================================
Replaces the simple first-order ka with a mechanistic, multi-compartment
GI tract model that simulates drug dissolution and absorption as it
transits through stomach → duodenum → jejunum → ileum → colon.

This is what Simcyp's ADAM model does. Key features:
1. Compartmental transit (7 GI segments with different pH, surface area, transit time)
2. Dissolution kinetics (Noyes-Whitney equation — particle size, solubility, diffusion)
3. Permeability-limited absorption (Peff from PAMPA/Caco-2 or predicted)
4. Gut wall metabolism (CYP3A4/UGT in enterocytes)
5. Formulation effects (IR tablet, capsule, solution, enteric-coated)

This feeds into the PBPK model's gut compartment with a realistic
time-dependent absorption profile instead of a single ka.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


class Formulation(Enum):
    SOLUTION = "solution"           # Already dissolved — fastest absorption
    IR_TABLET = "ir_tablet"         # Immediate release tablet — needs disintegration + dissolution
    CAPSULE = "capsule"             # Capsule — disintegration then dissolution
    ENTERIC_COATED = "enteric"      # Acid-resistant coating — no stomach release
    SUSTAINED_RELEASE = "sr"        # Slow continuous release
    SUSPENSION = "suspension"       # Particles in liquid — partial dissolution


# ════════════════════════════════════════════════════════════════
# GI TRACT PHYSIOLOGY
# ════════════════════════════════════════════════════════════════

@dataclass
class GISegment:
    """Physiological parameters for one GI segment."""
    name: str
    pH: float                    # Luminal pH
    volume_mL: float             # Fluid volume
    surface_area_cm2: float      # Absorptive surface area
    transit_time_h: float        # Mean residence time
    radius_cm: float             # Segment radius (for SA calculation)
    bile_salt_mM: float = 0.0    # Bile salt concentration (aids dissolution)
    has_cyp3a4: bool = False     # Gut wall CYP3A4 present?
    has_ugt: bool = False        # Gut wall UGT present?
    cyp3a4_activity: float = 0.0 # Relative CYP3A4 activity (0-1)


# Reference: Yu & Amidon (1999), Willmann et al. (2004)
GI_TRACT = [
    GISegment("Stomach",     1.5, 250,   3.5,    0.25, 5.0,   0.0,  False, False, 0.0),
    GISegment("Duodenum",    6.0,  40, 150.0,    0.26, 1.75, 10.0,  True,  True,  0.4),
    GISegment("Jejunum_1",   6.2, 120, 550.0,    0.47, 1.75,  8.0,  True,  True,  1.0),
    GISegment("Jejunum_2",   6.4, 120, 550.0,    0.47, 1.75,  6.0,  True,  True,  0.8),
    GISegment("Ileum_1",     6.6, 120, 350.0,    0.47, 1.5,   4.0,  True,  True,  0.5),
    GISegment("Ileum_2",     7.0, 120, 350.0,    0.47, 1.5,   2.0,  True,  False, 0.3),
    GISegment("Ileum_3",     7.2, 120, 200.0,    0.47, 1.5,   1.0,  False, False, 0.1),
    GISegment("Colon_asc",   7.8, 150,  50.0,    4.00, 3.0,   0.5,  False, False, 0.0),
    GISegment("Colon_trans", 7.5, 100,  30.0,    6.00, 2.5,   0.2,  False, False, 0.0),
]


# ════════════════════════════════════════════════════════════════
# DRUG DISSOLUTION PROPERTIES
# ════════════════════════════════════════════════════════════════

@dataclass
class DissolutionProfile:
    """Drug-specific dissolution and absorption parameters."""
    # Solubility
    solubility_mg_mL: float = 1.0     # Intrinsic aqueous solubility at pH 7
    solubility_pH_dependent: bool = True
    pka: float = 7.0                   # For pH-solubility calculation
    is_acid: bool = True               # Weak acid or base

    # Dissolution kinetics (Noyes-Whitney)
    particle_radius_um: float = 25.0   # Initial particle radius
    density_g_cm3: float = 1.2         # Particle density
    diffusion_coeff: float = 5e-6      # Diffusion coefficient (cm²/s)

    # Permeability
    peff_cm_s: float = 2e-4            # Effective permeability (human jejunal)
    # Peff > 1e-4: high permeability (BCS I/II)
    # Peff < 1e-4: low permeability (BCS III/IV)

    # Gut wall metabolism
    fg: float = 1.0                    # Fraction escaping gut wall metabolism
    gut_cyp3a4_substrate: bool = False

    # Formulation
    formulation: str = "ir_tablet"
    disintegration_time_h: float = 0.25  # Time for tablet to break apart


# Known drug dissolution profiles
DISSOLUTION_DATABASE = {
    "aspirin": DissolutionProfile(
        solubility_mg_mL=3.0, pka=3.5, is_acid=True,
        particle_radius_um=20, peff_cm_s=5e-4,
        fg=0.95, formulation="ir_tablet", disintegration_time_h=0.15),
    "caffeine": DissolutionProfile(
        solubility_mg_mL=21.6, pka=14.0, is_acid=False,
        particle_radius_um=15, peff_cm_s=8e-4,
        fg=1.0, formulation="ir_tablet", disintegration_time_h=0.1),
    "paracetamol": DissolutionProfile(
        solubility_mg_mL=14.0, pka=9.5, is_acid=True,
        particle_radius_um=20, peff_cm_s=4e-4,
        fg=0.98, formulation="ir_tablet", disintegration_time_h=0.12),
    "ibuprofen": DissolutionProfile(
        solubility_mg_mL=0.021, pka=4.4, is_acid=True,
        particle_radius_um=30, peff_cm_s=6e-4,
        fg=0.95, formulation="ir_tablet", disintegration_time_h=0.2),
    "metformin": DissolutionProfile(
        solubility_mg_mL=300.0, pka=12.4, is_acid=False,
        particle_radius_um=10, peff_cm_s=0.3e-4,  # Low permeability (BCS III)
        fg=1.0, formulation="ir_tablet", disintegration_time_h=0.15),
    "atorvastatin": DissolutionProfile(
        solubility_mg_mL=0.001, pka=4.3, is_acid=True,
        particle_radius_um=5, peff_cm_s=3e-4,
        fg=0.7, gut_cyp3a4_substrate=True,
        formulation="ir_tablet", disintegration_time_h=0.2),
    "simvastatin": DissolutionProfile(
        solubility_mg_mL=0.0003, pka=4.0, is_acid=True,
        particle_radius_um=5, peff_cm_s=5e-4,
        fg=0.5, gut_cyp3a4_substrate=True,
        formulation="ir_tablet", disintegration_time_h=0.2),
    "amlodipine": DissolutionProfile(
        solubility_mg_mL=0.075, pka=8.6, is_acid=False,
        particle_radius_um=10, peff_cm_s=4e-4,
        fg=0.9, formulation="ir_tablet", disintegration_time_h=0.15),
    "omeprazole": DissolutionProfile(
        solubility_mg_mL=0.36, pka=4.0, is_acid=False,
        particle_radius_um=10, peff_cm_s=3e-4,
        fg=0.85, formulation="enteric", disintegration_time_h=0.0),
    "diclofenac": DissolutionProfile(
        solubility_mg_mL=0.002, pka=4.0, is_acid=True,
        particle_radius_um=20, peff_cm_s=8e-4,
        fg=0.95, formulation="ir_tablet", disintegration_time_h=0.2),
}


# ════════════════════════════════════════════════════════════════
# DISSOLUTION ENGINE
# ════════════════════════════════════════════════════════════════

class DissolutionEngine:
    """
    Mechanistic dissolution and absorption model.

    Simulates drug transit through 9 GI segments with:
    - pH-dependent solubility at each segment
    - Noyes-Whitney dissolution kinetics
    - Permeability-limited absorption
    - Gut wall CYP3A4 metabolism
    - Formulation effects
    """

    def __init__(self, drug: DissolutionProfile, dose_mg: float = 100.0,
                 fed_state: str = "fasted", ppi: bool = False):
        self.drug = drug
        self.dose_mg = dose_mg
        self.fed_state = fed_state
        self.ppi = ppi
        self.segments = GI_TRACT.copy()

        # Modify GI physiology for fed state
        if fed_state == "high_fat":
            self.segments[0].volume_mL = 500     # More fluid in stomach
            self.segments[0].transit_time_h = 1.0  # Slower gastric emptying
            self.segments[0].pH = 3.0             # Higher stomach pH with food
            # Bile salts increase in fed state
            for seg in self.segments[1:7]:
                seg.bile_salt_mM *= 2.0
        elif fed_state == "light_meal":
            self.segments[0].volume_mL = 350
            self.segments[0].transit_time_h = 0.5
            self.segments[0].pH = 2.5

        # PPI effect on stomach pH
        if ppi:
            self.segments[0].pH = 4.5

    def solubility_at_pH(self, pH: float) -> float:
        """
        Calculate pH-dependent solubility using Henderson-Hasselbalch.

        For weak acids: S(pH) = S0 × (1 + 10^(pH - pKa))
        For weak bases: S(pH) = S0 × (1 + 10^(pKa - pH))
        """
        s0 = self.drug.solubility_mg_mL

        if not self.drug.solubility_pH_dependent:
            return s0

        if self.drug.is_acid:
            return s0 * (1 + 10 ** (pH - self.drug.pka))
        else:
            return s0 * (1 + 10 ** (self.drug.pka - pH))

    def dissolution_rate(self, undissolved_mg: float, dissolved_mg: float,
                          segment: GISegment) -> float:
        """
        Noyes-Whitney dissolution rate.

        dM/dt = (D × SA × (Cs - C)) / h

        Where:
        - D = diffusion coefficient
        - SA = surface area of particles
        - Cs = saturation solubility
        - C = current dissolved concentration
        - h = diffusion layer thickness
        """
        if undissolved_mg <= 0:
            return 0.0

        # Saturation solubility at this segment's pH
        cs = self.solubility_at_pH(segment.pH)

        # Current dissolved concentration
        c_dissolved = dissolved_mg / max(segment.volume_mL * 0.001, 0.001)  # mg/mL

        # Saturation check
        if c_dissolved >= cs:
            return 0.0  # Already saturated

        # Surface area of remaining particles (assumes spherical, monodisperse)
        r = self.drug.particle_radius_um * 1e-4  # Convert to cm
        if undissolved_mg > 0:
            # N particles × surface area per particle
            vol_per_particle = (4/3) * np.pi * r**3 * self.drug.density_g_cm3  # g
            n_particles = (undissolved_mg * 1e-3) / max(vol_per_particle, 1e-15)
            sa = n_particles * 4 * np.pi * r**2  # Total surface area (cm²)
        else:
            sa = 0

        # Diffusion layer thickness (stagnant layer model)
        h = 30e-4  # 30 µm diffusion layer

        # Bile salt enhancement (increases effective solubility)
        bile_enhance = 1.0 + segment.bile_salt_mM * 0.1

        # Noyes-Whitney rate (mg/h)
        rate = (self.drug.diffusion_coeff * sa * (cs * bile_enhance - c_dissolved)) / h
        rate *= 3600  # Convert from per-second to per-hour

        return max(rate, 0.0)

    def absorption_rate(self, dissolved_mg: float, segment: GISegment) -> float:
        """
        Permeability-limited absorption rate.

        Rate = Peff × SA × C_dissolved × (1 - fg_loss)

        Where fg_loss accounts for gut wall CYP3A4 metabolism.
        """
        if dissolved_mg <= 0:
            return 0.0

        c_dissolved = dissolved_mg / max(segment.volume_mL * 0.001, 0.001)  # mg/mL → mg/L

        # Effective permeability (cm/s → cm/h)
        peff_h = self.drug.peff_cm_s * 3600

        # Absorption rate (mg/h)
        rate = peff_h * segment.surface_area_cm2 * c_dissolved * 1e-3  # mg/h

        # Gut wall metabolism
        fg = self.drug.fg
        if self.drug.gut_cyp3a4_substrate and segment.has_cyp3a4:
            fg *= (1.0 - segment.cyp3a4_activity * 0.3)  # Up to 30% loss per segment

        return rate * fg

    def simulate(self, n_points: int = 100) -> dict:
        """
        Simulate drug dissolution and absorption through the GI tract.

        Returns time-resolved absorption profile that replaces simple ka.
        """
        t_max = 24.0  # Hours
        dt = t_max / n_points

        n_seg = len(self.segments)

        # State: [undissolved, dissolved] per segment + total absorbed
        undissolved = np.zeros((n_points, n_seg))
        dissolved = np.zeros((n_points, n_seg))
        absorbed = np.zeros(n_points)
        cumulative_absorbed = np.zeros(n_points)

        # Initial conditions based on formulation
        if self.drug.formulation == "solution":
            dissolved[0, 0] = self.dose_mg  # All dissolved in stomach
        elif self.drug.formulation == "enteric":
            # Enteric-coated: releases in duodenum (segment 1)
            undissolved[0, 1] = self.dose_mg
        else:
            undissolved[0, 0] = self.dose_mg  # Solid in stomach

        time_h = np.linspace(0, t_max, n_points)

        for i in range(1, n_points):
            t = time_h[i]

            for j in range(n_seg):
                seg = self.segments[j]

                # Disintegration delay (for tablets)
                if self.drug.formulation in ("ir_tablet", "capsule") and j == 0:
                    if t < self.drug.disintegration_time_h:
                        undissolved[i, j] = undissolved[i-1, j]
                        dissolved[i, j] = dissolved[i-1, j]
                        continue

                # Dissolution
                diss_rate = self.dissolution_rate(undissolved[i-1, j], dissolved[i-1, j], seg)
                actual_diss = min(diss_rate * dt, undissolved[i-1, j])

                # Absorption (only dissolved drug absorbs)
                abs_rate = self.absorption_rate(dissolved[i-1, j], seg)
                actual_abs = min(abs_rate * dt, dissolved[i-1, j])

                # Transit to next segment
                transit_rate = 1.0 / max(seg.transit_time_h, 0.01)
                transit_undiss = undissolved[i-1, j] * transit_rate * dt
                transit_diss = dissolved[i-1, j] * transit_rate * dt

                # Update this segment
                undissolved[i, j] = max(0, undissolved[i-1, j] - actual_diss - transit_undiss)
                dissolved[i, j] = max(0, dissolved[i-1, j] + actual_diss - actual_abs - transit_diss)

                # Add transit from previous segment
                if j > 0:
                    prev_seg = self.segments[j-1]
                    prev_transit_rate = 1.0 / max(prev_seg.transit_time_h, 0.01)
                    undissolved[i, j] += undissolved[i-1, j-1] * prev_transit_rate * dt
                    dissolved[i, j] += dissolved[i-1, j-1] * prev_transit_rate * dt

                absorbed[i] += actual_abs

            cumulative_absorbed[i] = cumulative_absorbed[i-1] + absorbed[i]

        # Calculate effective absorption parameters
        total_absorbed = cumulative_absorbed[-1]
        f_absorbed = total_absorbed / self.dose_mg if self.dose_mg > 0 else 0

        # Effective ka (fit a first-order curve to the absorption profile)
        # Find time to 63.2% absorption (1 time constant)
        target = total_absorbed * 0.632
        t63_idx = np.searchsorted(cumulative_absorbed, target)
        t63 = time_h[min(t63_idx, len(time_h)-1)]
        effective_ka = 1.0 / max(t63, 0.1)  # 1/h

        # Tlag (time to first significant absorption >5%)
        threshold = total_absorbed * 0.05
        tlag_idx = np.searchsorted(cumulative_absorbed, threshold)
        tlag = time_h[min(tlag_idx, len(time_h)-1)]

        # Per-segment absorption fractions
        segment_absorption = {}
        for j, seg in enumerate(self.segments):
            seg_total = np.sum(absorbed * (dissolved[:, j] > 0.01))
            segment_absorption[seg.name] = round(seg_total / max(total_absorbed, 0.001) * 100, 1)

        return {
            "time_h": [float(x) for x in time_h],
            "absorption_rate_mg_h": [float(x) for x in absorbed],
            "cumulative_absorbed_mg": [float(x) for x in cumulative_absorbed],
            "total_absorbed_mg": round(total_absorbed, 2),
            "fraction_absorbed": round(f_absorbed, 3),
            "effective_ka": round(effective_ka, 3),
            "effective_tlag_h": round(tlag, 3),
            "segment_absorption_pct": segment_absorption,
            "formulation": self.drug.formulation,
            "fed_state": self.fed_state,
            "dissolution_limited": self.drug.solubility_mg_mL < 0.1,
            "permeability_limited": self.drug.peff_cm_s < 1e-4,
        }


def get_dissolution_profile(smiles: str = "", drug_name: str = "") -> DissolutionProfile:
    """Get dissolution profile for a drug."""
    name_lower = drug_name.lower().strip()
    for known_name, profile in DISSOLUTION_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            return profile

    # Predict from molecular properties
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)

            # Estimate solubility (ESOL-like)
            log_s = 0.5 - 0.01 * mw - 0.5 * logp + 0.003 * tpsa
            solubility = 10 ** log_s  # mol/L → mg/mL approximation
            solubility_mg_ml = solubility * mw / 1000

            # Estimate permeability
            if tpsa < 90 and mw < 500:
                peff = 4e-4  # High
            elif tpsa < 140:
                peff = 1.5e-4  # Moderate
            else:
                peff = 0.5e-4  # Low

            # Determine if acid or base
            has_cooh = "C(=O)O" in smiles or "C(O)=O" in smiles
            has_amine = smiles.count("N") > 0

            return DissolutionProfile(
                solubility_mg_mL=max(solubility_mg_ml, 0.0001),
                pka=4.0 if has_cooh else 8.5,
                is_acid=has_cooh,
                peff_cm_s=peff,
                fg=max(0.5, 1.0 - logp * 0.05) if logp > 3 else 1.0,
                gut_cyp3a4_substrate=logp > 3 and mw > 300,
            )
    except Exception:
        pass

    return DissolutionProfile()


def simulate_dissolution(
    smiles: str = "", drug_name: str = "",
    dose_mg: float = 100.0,
    fed_state: str = "fasted",
    ppi: bool = False,
) -> dict:
    """
    Main entry point: simulate dissolution and absorption.

    Returns dict with absorption profile and effective PK parameters.
    """
    profile = get_dissolution_profile(smiles, drug_name)
    engine = DissolutionEngine(profile, dose_mg, fed_state, ppi)
    result = engine.simulate()

    logger.info("Dissolution: F_abs=%.3f, ka_eff=%.2f, tlag=%.2fh, formulation=%s",
               result["fraction_absorbed"], result["effective_ka"],
               result["effective_tlag_h"], result["formulation"])

    return result
