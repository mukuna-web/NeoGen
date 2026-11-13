"""
Receptor Binding Kinetics
==========================
Models dynamic receptor occupancy over time using kon/koff kinetics
instead of static Emax model.

Key concepts:
1. Drug + Receptor ⇌ Drug-Receptor complex
   kon = association rate (1/µM/h), koff = dissociation rate (1/h)
   Kd = koff/kon = equilibrium dissociation constant (µM)

2. Occupancy(t) evolves dynamically:
   dRO/dt = kon × C(t) × (1 - RO) - koff × RO

3. This gives HYSTERESIS — effect lags behind concentration differently
   for drugs with slow kon (slow onset) vs slow koff (long duration)

Example: Omeprazole has slow kon but essentially zero koff (irreversible)
→ once bound, proton pump stays off for 24h regardless of drug levels
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReceptorTarget:
    """A specific receptor/enzyme target of a drug."""
    receptor_name: str            # e.g., "COX-1", "Beta-1 adrenergic"
    receptor_type: str            # "GPCR", "ion_channel", "enzyme", "nuclear", "kinase"
    organ: str                    # Primary organ
    cell_type: str                # Cell type expressing this receptor

    # Binding kinetics
    kon: float = 10.0             # Association rate (1/µM/h)
    koff: float = 1.0             # Dissociation rate (1/h)
    kd_uM: float = 0.1           # Equilibrium Kd (µM) = koff/kon

    # Action
    action: str = "antagonist"    # "agonist", "antagonist", "inhibitor", "inverse_agonist"
    reversible: bool = True       # Irreversible binding (aspirin COX, omeprazole PPI)

    # Downstream
    max_effect: float = 100.0     # Maximum pharmacological effect
    effect_unit: str = "%"
    ec50_occupancy: float = 0.5   # Receptor occupancy needed for 50% max effect

    # Clinical
    therapeutic_occupancy: float = 0.7  # Target occupancy for therapeutic effect
    description: str = ""


@dataclass
class ReceptorProfile:
    """Complete receptor profile for a drug."""
    drug_name: str
    targets: list = field(default_factory=list)
    selectivity_note: str = ""    # Selectivity between targets


RECEPTOR_DATABASE = {
    "aspirin": ReceptorProfile("aspirin", [
        ReceptorTarget("COX-1", "enzyme", "stomach", "platelet",
                      kon=50.0, koff=0.001, kd_uM=0.00002,  # Essentially irreversible
                      action="inhibitor", reversible=False,
                      max_effect=95, effect_unit="% platelet aggregation inhibition",
                      therapeutic_occupancy=0.9,
                      description="Acetylates Ser530 — IRREVERSIBLE. Platelet has no nucleus → can't make new COX-1 → effect lasts 7-10 days (platelet lifespan)"),
        ReceptorTarget("COX-2", "enzyme", "brain", "neuron",
                      kon=20.0, koff=0.5, kd_uM=0.025,
                      action="inhibitor", reversible=True,
                      max_effect=75, effect_unit="% PGE2 reduction",
                      therapeutic_occupancy=0.6,
                      description="Reversible COX-2 inhibition in CNS — pain and fever relief"),
    ], selectivity_note="100x more potent at COX-1 vs COX-2 at low doses"),

    "caffeine": ReceptorProfile("caffeine", [
        ReceptorTarget("Adenosine A2A", "GPCR", "brain", "neuron",
                      kon=5.0, koff=3.0, kd_uM=0.6,  # Moderate affinity, fast kinetics
                      action="antagonist", reversible=True,
                      max_effect=80, effect_unit="% wakefulness",
                      therapeutic_occupancy=0.5,
                      description="Blocks adenosine → prevents drowsiness signal. Fast on, fast off → short duration"),
        ReceptorTarget("Adenosine A1", "GPCR", "heart", "cardiomyocyte",
                      kon=3.0, koff=4.0, kd_uM=1.3,  # Lower affinity
                      action="antagonist", reversible=True,
                      max_effect=30, effect_unit="% HR increase",
                      therapeutic_occupancy=0.3,
                      description="A1 block removes heart rate brake → tachycardia"),
        ReceptorTarget("PDE", "enzyme", "lung", "smooth_muscle",
                      kon=1.0, koff=2.0, kd_uM=2.0,
                      action="inhibitor", reversible=True,
                      max_effect=20, effect_unit="% bronchodilation",
                      description="Phosphodiesterase inhibition → bronchodilation (weak)"),
    ]),

    "propranolol": ReceptorProfile("propranolol", [
        ReceptorTarget("Beta-1 adrenergic", "GPCR", "heart", "cardiomyocyte",
                      kon=30.0, koff=0.5, kd_uM=0.017,
                      action="antagonist", reversible=True,
                      max_effect=35, effect_unit="% HR reduction",
                      therapeutic_occupancy=0.7,
                      description="Blocks sympathetic drive to heart → reduces rate and contractility"),
        ReceptorTarget("Beta-2 adrenergic", "GPCR", "lung", "smooth_muscle",
                      kon=25.0, koff=0.6, kd_uM=0.024,
                      action="antagonist", reversible=True,
                      max_effect=40, effect_unit="% bronchoconstriction risk",
                      therapeutic_occupancy=0.5,
                      description="NON-SELECTIVE → blocks Beta-2 in lungs → can trigger asthma attack"),
    ], selectivity_note="Non-selective beta blocker — contraindicated in asthma"),

    "omeprazole": ReceptorProfile("omeprazole", [
        ReceptorTarget("H+/K+ ATPase", "enzyme", "stomach", "parietal_cell",
                      kon=100.0, koff=0.001, kd_uM=0.00001,  # Irreversible (covalent)
                      action="inhibitor", reversible=False,
                      max_effect=95, effect_unit="% acid suppression",
                      therapeutic_occupancy=0.7,
                      description="Covalent binding to proton pump. IRREVERSIBLE — new pumps must be synthesized (36h turnover). Single dose suppresses acid for 24h"),
    ]),

    "metoprolol": ReceptorProfile("metoprolol", [
        ReceptorTarget("Beta-1 adrenergic", "GPCR", "heart", "cardiomyocyte",
                      kon=35.0, koff=0.8, kd_uM=0.023,
                      action="antagonist", reversible=True,
                      max_effect=30, effect_unit="% HR reduction",
                      therapeutic_occupancy=0.6,
                      description="SELECTIVE Beta-1 blocker — less bronchospasm risk than propranolol"),
    ], selectivity_note="Beta-1 selective — safer in mild asthma/COPD than non-selective"),

    "atorvastatin": ReceptorProfile("atorvastatin", [
        ReceptorTarget("HMG-CoA reductase", "enzyme", "liver", "hepatocyte",
                      kon=40.0, koff=0.1, kd_uM=0.0025,  # Very tight binding
                      action="inhibitor", reversible=True,
                      max_effect=55, effect_unit="% LDL reduction",
                      therapeutic_occupancy=0.8,
                      description="Competitive inhibitor — binds 10,000x tighter than natural substrate. Slow off-rate → long duration"),
    ]),
}


def simulate_receptor_occupancy(
    conc_time_course: np.ndarray,  # Concentration at receptor site over time (µM)
    time_h: np.ndarray,
    target: ReceptorTarget,
) -> dict:
    """
    Simulate dynamic receptor occupancy using kon/koff kinetics.

    dRO/dt = kon × C(t) × (1 - RO) - koff × RO

    For irreversible: koff ≈ 0, occupancy only increases.
    """
    ro = np.zeros_like(time_h)  # Receptor occupancy (0-1)

    for i in range(1, len(time_h)):
        dt = time_h[i] - time_h[i - 1]
        c = conc_time_course[i]

        # Binding and unbinding
        binding = target.kon * c * (1.0 - ro[i - 1])
        unbinding = target.koff * ro[i - 1]

        dro = (binding - unbinding) * dt
        ro[i] = np.clip(ro[i - 1] + dro, 0, 1)

    # Calculate effect from occupancy
    effect = np.array([
        target.max_effect * (occ / (target.ec50_occupancy + occ)) if occ > 0 else 0
        for occ in ro
    ])

    # Summary statistics
    peak_occupancy = float(np.max(ro))
    peak_occ_time = float(time_h[np.argmax(ro)])
    peak_effect = float(np.max(effect))

    # Time above therapeutic occupancy
    above_therapeutic = ro >= target.therapeutic_occupancy
    if np.any(above_therapeutic):
        first_above = time_h[np.argmax(above_therapeutic)]
        last_above = time_h[len(above_therapeutic) - np.argmax(above_therapeutic[::-1]) - 1]
        therapeutic_duration = float(last_above - first_above)
    else:
        therapeutic_duration = 0.0

    # Residence time (receptor-bound drug half-life)
    residence_time = 1.0 / max(target.koff, 0.001)

    return {
        "receptor": target.receptor_name,
        "receptor_type": target.receptor_type,
        "organ": target.organ,
        "cell_type": target.cell_type,
        "action": target.action,
        "reversible": target.reversible,
        "description": target.description,

        # Time courses (sampled, ensure native Python floats)
        "time_h": [float(x) for x in time_h[::5]],
        "occupancy": [float(x) for x in ro[::5]],
        "effect": [float(x) for x in effect[::5]],

        # Kinetics
        "kon": target.kon,
        "koff": target.koff,
        "kd_uM": target.kd_uM,
        "residence_time_h": round(residence_time, 1),

        # Summary
        "peak_occupancy_pct": round(peak_occupancy * 100, 1),
        "peak_occupancy_time_h": round(peak_occ_time, 2),
        "peak_effect": round(peak_effect, 1),
        "effect_unit": target.effect_unit,
        "therapeutic_duration_h": round(therapeutic_duration, 1),
        "above_therapeutic_threshold": peak_occupancy >= target.therapeutic_occupancy,
    }


def assess_receptor_kinetics(
    smiles: str = "",
    drug_name: str = "",
    time_course: list = None,
    mw: float = 300,
) -> dict:
    """
    Full receptor binding kinetics assessment.

    Takes PBPK time course, converts concentrations to µM at each receptor site,
    and simulates dynamic occupancy for all targets.
    """
    name_lower = drug_name.lower().strip()

    profile = None
    for known_name, p in RECEPTOR_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            profile = p
            break

    if not profile or not time_course:
        return {"drug": drug_name or "Unknown", "assessed": False, "targets": []}

    time_h = np.array([tp["time_h"] for tp in time_course])

    # Get concentrations at each organ
    conc_map = {
        "brain": np.array([tp.get("brain_conc", 0) for tp in time_course]),
        "liver": np.array([tp.get("liver_conc", 0) for tp in time_course]),
        "kidney": np.array([tp.get("kidney_conc", 0) for tp in time_course]),
        "stomach": np.array([tp.get("gut_conc", 0) for tp in time_course]),
        "heart": np.array([tp.get("plasma_conc", 0) for tp in time_course]),
        "lung": np.array([tp.get("plasma_conc", 0) for tp in time_course]),
        "muscle": np.array([tp.get("muscle_conc", 0) for tp in time_course]),
    }

    results = []
    for target in profile.targets:
        # Get concentration at target organ (mg/L → µM)
        conc_mg_l = conc_map.get(target.organ, conc_map["heart"])
        conc_uM = conc_mg_l / max(mw, 100) * 1000  # mg/L → µM

        result = simulate_receptor_occupancy(conc_uM, time_h, target)
        results.append(result)

    return {
        "drug": profile.drug_name,
        "assessed": True,
        "selectivity_note": profile.selectivity_note,
        "n_targets": len(results),
        "targets": results,
    }
