"""
Effect Compartment Model (PK/PD Link)
=======================================
Models the TIME LAG between drug concentration in blood and drug EFFECT.

The problem:
  Blood concentration peaks at t=1h, but pain relief peaks at t=2h.
  Why? Drug needs time to travel from blood to the effect site (receptor).

The solution: Effect compartment model (Sheiner 1979)
  - Hypothetical compartment where drug effect occurs
  - Connected to plasma by a first-order rate constant (ke0)
  - Effect = f(Ce) where Ce = effect site concentration
  - Ce lags behind Cp (plasma concentration)

ke0 determines how fast Ce equilibrates with Cp:
  - High ke0 (>1 h⁻¹) → fast equilibration → minimal hysteresis
  - Low ke0 (<0.1 h⁻¹) → slow equilibration → large hysteresis

Clinical examples:
  - Morphine: ke0 = 0.12 h⁻¹ → effect peaks ~45 min after blood peak
  - Warfarin: ke0 = 0.003 h⁻¹ → INR changes over DAYS (not hours)
  - Diazepam: ke0 = 0.06 h⁻¹ → sedation peaks ~2h after blood peak
  - Propofol: ke0 = 2.4 h⁻¹ → almost instant effect (IV anesthetic)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EffectCompartmentParams:
    """Parameters for the effect compartment model."""
    drug_name: str
    ke0: float              # Effect site equilibration rate constant (1/h)
    ec50: float             # Concentration at 50% max effect (mg/L at effect site)
    emax: float = 100.0     # Maximum effect (%)
    hill: float = 1.0       # Hill coefficient (steepness of concentration-effect curve)
    baseline_effect: float = 0.0  # Effect at zero concentration
    effect_unit: str = "%"
    effect_name: str = ""   # e.g., "Pain relief", "BP reduction"
    t_peak_lag_h: float = 0.0  # Approximate lag between Cp_max and Ce_max


# ════════════════════════════════════════════════════════════════
# DRUG EFFECT COMPARTMENT DATABASE
# ════════════════════════════════════════════════════════════════

EFFECT_COMPARTMENT_DRUGS = {
    "morphine": EffectCompartmentParams(
        drug_name="Morphine", ke0=0.12, ec50=0.025, emax=100, hill=2.5,
        effect_name="Pain relief (analgesia)", effect_unit="%",
        t_peak_lag_h=0.75,
    ),
    "aspirin": EffectCompartmentParams(
        drug_name="Aspirin", ke0=0.8, ec50=5.0, emax=95, hill=1.5,
        effect_name="COX-1 inhibition (antiplatelet)", effect_unit="%",
        t_peak_lag_h=0.25,
    ),
    "ibuprofen": EffectCompartmentParams(
        drug_name="Ibuprofen", ke0=0.5, ec50=10.0, emax=85, hill=1.8,
        effect_name="Pain relief (COX inhibition)", effect_unit="%",
        t_peak_lag_h=0.5,
    ),
    "paracetamol": EffectCompartmentParams(
        drug_name="Paracetamol", ke0=0.6, ec50=8.0, emax=70, hill=1.5,
        effect_name="Pain relief / Antipyresis", effect_unit="%",
        t_peak_lag_h=0.3,
    ),
    "warfarin": EffectCompartmentParams(
        drug_name="Warfarin", ke0=0.003, ec50=0.3, emax=90, hill=1.2,
        effect_name="INR elevation (anticoagulation)", effect_unit="% of max INR change",
        t_peak_lag_h=72.0,  # 3 DAYS lag!
    ),
    "diazepam": EffectCompartmentParams(
        drug_name="Diazepam", ke0=0.06, ec50=0.15, emax=100, hill=2.0,
        effect_name="Sedation / Anxiolysis", effect_unit="%",
        t_peak_lag_h=2.0,
    ),
    "caffeine": EffectCompartmentParams(
        drug_name="Caffeine", ke0=1.5, ec50=3.0, emax=80, hill=1.5,
        effect_name="Alertness / Wakefulness", effect_unit="%",
        t_peak_lag_h=0.15,
    ),
    "metoprolol": EffectCompartmentParams(
        drug_name="Metoprolol", ke0=0.15, ec50=0.05, emax=30, hill=1.3,
        effect_name="Heart rate reduction", effect_unit="bpm reduction",
        t_peak_lag_h=1.0,
    ),
    "amlodipine": EffectCompartmentParams(
        drug_name="Amlodipine", ke0=0.01, ec50=0.003, emax=25, hill=1.5,
        effect_name="Blood pressure reduction", effect_unit="mmHg reduction",
        t_peak_lag_h=8.0,
    ),
    "omeprazole": EffectCompartmentParams(
        drug_name="Omeprazole", ke0=0.05, ec50=0.2, emax=95, hill=2.0,
        effect_name="Gastric acid suppression", effect_unit="%",
        t_peak_lag_h=2.0,
    ),
    "fluoxetine": EffectCompartmentParams(
        drug_name="Fluoxetine", ke0=0.001, ec50=0.01, emax=70, hill=1.0,
        effect_name="Antidepressant effect (mood improvement)", effect_unit="%",
        t_peak_lag_h=336.0,  # 2 WEEKS! (receptor downregulation needed)
    ),
}


# ════════════════════════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════════════════════════

def simulate_effect_compartment(
    plasma_concentrations: list[float] | np.ndarray,
    time_points_h: list[float] | np.ndarray,
    drug_name: str = "",
    ke0: float = None,
    ec50: float = None,
) -> dict:
    """
    Simulate effect compartment concentration and drug effect over time.

    Uses the differential equation:
        dCe/dt = ke0 × (Cp - Ce)

    Where Ce = effect site concentration, Cp = plasma concentration

    Then effect = Emax × Ce^hill / (EC50^hill + Ce^hill)

    Args:
        plasma_concentrations: Array of plasma concentrations at each time point
        time_points_h: Array of time points (hours)
        drug_name: Drug name (to look up ke0, EC50)
        ke0: Override ke0 if provided
        ec50: Override EC50 if provided

    Returns:
        Dict with effect site concentrations, effect values, and timing info
    """
    Cp = np.array(plasma_concentrations)
    t = np.array(time_points_h)

    # Look up drug parameters
    key = drug_name.lower().replace(" ", "_")
    params = EFFECT_COMPARTMENT_DRUGS.get(key)

    if params:
        _ke0 = ke0 or params.ke0
        _ec50 = ec50 or params.ec50
        _emax = params.emax
        _hill = params.hill
        _baseline = params.baseline_effect
        effect_name = params.effect_name
    else:
        _ke0 = ke0 or 0.5
        _ec50 = ec50 or max(Cp.max() * 0.5, 0.001)
        _emax = 100.0
        _hill = 1.0
        _baseline = 0.0
        effect_name = "Pharmacological effect"

    # Simulate effect compartment using Euler method
    Ce = np.zeros_like(Cp)
    dt_arr = np.diff(t, prepend=t[0])

    for i in range(1, len(t)):
        dt = dt_arr[i]
        if dt <= 0:
            dt = 0.01
        dCe = _ke0 * (Cp[i-1] - Ce[i-1]) * dt
        Ce[i] = max(0, Ce[i-1] + dCe)

    # Calculate effect using sigmoid Emax model
    effect = np.zeros_like(Ce)
    for i in range(len(Ce)):
        if Ce[i] > 0:
            effect[i] = _baseline + _emax * (Ce[i] ** _hill) / (_ec50 ** _hill + Ce[i] ** _hill)

    # Find key timepoints
    cp_max_idx = np.argmax(Cp)
    ce_max_idx = np.argmax(Ce)
    effect_max_idx = np.argmax(effect)

    hysteresis_lag = t[effect_max_idx] - t[cp_max_idx] if effect_max_idx > cp_max_idx else 0

    # Build time course for UI
    time_course = []
    step = max(1, len(t) // 200)  # Limit to 200 points
    for i in range(0, len(t), step):
        time_course.append({
            "time_h": round(float(t[i]), 3),
            "plasma_conc": round(float(Cp[i]), 4),
            "effect_site_conc": round(float(Ce[i]), 4),
            "effect": round(float(effect[i]), 2),
        })

    return {
        "drug": drug_name or "Unknown",
        "effect_name": effect_name,
        "parameters": {
            "ke0": _ke0,
            "ec50": _ec50,
            "emax": _emax,
            "hill": _hill,
            "t_half_ke0": round(0.693 / _ke0, 2) if _ke0 > 0 else 0,
        },
        "timing": {
            "cp_max_time_h": round(float(t[cp_max_idx]), 2),
            "cp_max_value": round(float(Cp[cp_max_idx]), 4),
            "ce_max_time_h": round(float(t[ce_max_idx]), 2),
            "ce_max_value": round(float(Ce[ce_max_idx]), 4),
            "effect_max_time_h": round(float(t[effect_max_idx]), 2),
            "effect_max_value": round(float(effect[effect_max_idx]), 2),
            "hysteresis_lag_h": round(hysteresis_lag, 2),
            "lag_description": (
                f"Effect peaks {hysteresis_lag:.1f}h AFTER blood concentration peaks"
                if hysteresis_lag > 0.1 else "Minimal lag — effect tracks blood concentration closely"
            ),
        },
        "time_course": time_course,
    }
