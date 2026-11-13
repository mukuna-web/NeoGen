"""
Enzyme Induction Time Course Model
=====================================
Models the DELAYED onset of CYP enzyme induction.

The problem:
  Our current DDI model applies CYP induction instantly. But in reality:
  - Enzyme INHIBITION is instant (drug blocks enzyme directly)
  - Enzyme INDUCTION takes 2-7 DAYS (drug upregulates gene transcription →
    new mRNA → new protein synthesis → accumulated enzyme)

The biology:
  1. Inducer binds nuclear receptor (PXR, CAR, or AhR)
  2. Receptor-ligand complex translocates to nucleus
  3. Binds to response elements on CYP gene promoter (takes hours)
  4. Increases mRNA transcription (takes 12-24h)
  5. mRNA translated to protein (takes 12-24h)
  6. New enzyme accumulates (rate depends on enzyme half-life)
  7. Steady-state induction at ~3-5 enzyme half-lives

Time to full induction depends on CYP enzyme half-life:
  - CYP3A4 t½ = 36h → full induction in 5-7 days
  - CYP1A2 t½ = 39h → full induction in 7-8 days
  - CYP2B6 t½ = 23h → full induction in 4-5 days
  - CYP2C9 t½ = 55h → full induction in 10-12 days

Clinical examples:
  - Rifampicin: starts inducing CYP3A4 in 2 days, full effect at 7 days
  - Carbamazepine autoinduction: t½ drops from 36h to 16h over 2 weeks
  - Smoking (CYP1A2): full induction over 3-7 days of regular smoking
  - St. John's Wort: CYP3A4 induction over 10-14 days

When STOPPING an inducer:
  - Enzyme levels return to baseline over 3-5 enzyme half-lives
  - If patient was on induced medication → levels RISE when inducer stopped → toxicity risk
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CYPEnzymeKinetics:
    """Turnover kinetics for a CYP enzyme."""
    name: str
    half_life_h: float        # Protein degradation half-life
    baseline_abundance: float  # pmol/mg microsomal protein (reference)
    max_fold_induction: float  # Maximum fold-increase with strong inducer


CYP_TURNOVER = {
    "CYP3A4": CYPEnzymeKinetics("CYP3A4", half_life_h=36, baseline_abundance=108, max_fold_induction=10),
    "CYP1A2": CYPEnzymeKinetics("CYP1A2", half_life_h=39, baseline_abundance=45, max_fold_induction=40),
    "CYP2B6": CYPEnzymeKinetics("CYP2B6", half_life_h=23, baseline_abundance=17, max_fold_induction=8),
    "CYP2C9": CYPEnzymeKinetics("CYP2C9", half_life_h=55, baseline_abundance=73, max_fold_induction=3),
    "CYP2C19": CYPEnzymeKinetics("CYP2C19", half_life_h=26, baseline_abundance=14, max_fold_induction=5),
    "CYP2E1": CYPEnzymeKinetics("CYP2E1", half_life_h=50, baseline_abundance=49, max_fold_induction=3),
}


@dataclass
class InducerProfile:
    """Drug that induces CYP enzymes."""
    name: str
    target_cyps: dict  # {CYP_name: fold_induction_at_steady_state}
    mechanism: str      # PXR, CAR, AhR
    clinical_note: str


INDUCER_PROFILES = {
    "rifampicin": InducerProfile(
        name="Rifampicin", mechanism="PXR (potent)",
        target_cyps={"CYP3A4": 8.0, "CYP2C9": 2.5, "CYP2C19": 3.0, "CYP2B6": 4.0},
        clinical_note="Most potent known CYP inducer. Reduces oral contraceptive, HIV drug, immunosuppressant levels by 80-95%.",
    ),
    "carbamazepine": InducerProfile(
        name="Carbamazepine", mechanism="PXR + CAR",
        target_cyps={"CYP3A4": 3.0, "CYP2B6": 2.5, "CYP1A2": 1.5},
        clinical_note="Autoinduction: induces its own metabolism. t½ drops from 36h to 16h over first 2 weeks.",
    ),
    "phenytoin": InducerProfile(
        name="Phenytoin", mechanism="PXR + CAR",
        target_cyps={"CYP3A4": 3.5, "CYP2C9": 2.0, "CYP2C19": 2.5},
        clinical_note="Strong inducer. Reduces warfarin, oral contraceptive, and cyclosporine levels significantly.",
    ),
    "st_johns_wort": InducerProfile(
        name="St. John's Wort", mechanism="PXR",
        target_cyps={"CYP3A4": 2.5, "CYP2C9": 1.5, "CYP1A2": 1.5},
        clinical_note="OTC supplement that causes clinically significant DDIs. Reduced cyclosporine levels → organ rejection cases reported.",
    ),
    "smoking": InducerProfile(
        name="Smoking (PAH)", mechanism="AhR",
        target_cyps={"CYP1A2": 3.0, "CYP1A1": 10.0},
        clinical_note="Polycyclic aromatic hydrocarbons in smoke induce CYP1A2. Quitting smoking → clozapine/theophylline levels rise → toxicity risk.",
    ),
    "efavirenz": InducerProfile(
        name="Efavirenz", mechanism="PXR + CAR",
        target_cyps={"CYP3A4": 1.5, "CYP2B6": 2.0},
        clinical_note="Autoinduction contributes to decreasing levels over first 2 weeks of therapy.",
    ),
    "chronic_alcohol": InducerProfile(
        name="Chronic Alcohol", mechanism="Protein stabilization",
        target_cyps={"CYP2E1": 3.0},
        clinical_note="Chronic alcohol stabilizes CYP2E1 protein (not transcriptional). Increases paracetamol → NAPQI conversion → enhanced hepatotoxicity risk.",
    ),
}


def simulate_induction_time_course(
    inducer_name: str,
    days_on_inducer: float = 7.0,
    days_off_inducer: float = 0.0,
    n_points: int = 100,
) -> dict:
    """
    Simulate CYP enzyme induction over time.

    The enzyme level follows:
        E(t) = E_baseline + (E_ss - E_baseline) × (1 - e^(-k_deg × t))

    Where:
        E_ss = E_baseline × fold_induction (steady-state induced level)
        k_deg = ln(2) / t½_enzyme (degradation rate constant)

    After stopping inducer:
        E(t) = E_ss × e^(-k_deg × t_off) + E_baseline × (1 - e^(-k_deg × t_off))
    """
    key = inducer_name.lower().replace(" ", "_").replace("'", "")
    profile = INDUCER_PROFILES.get(key)

    if not profile:
        return {"drug": inducer_name, "error": "No inducer profile available",
                "available": list(INDUCER_PROFILES.keys())}

    total_days = days_on_inducer + days_off_inducer
    t = [i * total_days / n_points for i in range(n_points + 1)]

    results_per_cyp = {}

    for cyp_name, fold_ss in profile.target_cyps.items():
        cyp_info = CYP_TURNOVER.get(cyp_name)
        if not cyp_info:
            continue

        k_deg = 0.693 / cyp_info.half_life_h  # per hour
        E_baseline = cyp_info.baseline_abundance
        E_ss = E_baseline * fold_ss

        enzyme_levels = []
        clearance_modifiers = []

        for day in t:
            t_hours = day * 24
            if day <= days_on_inducer:
                # Induction phase
                E = E_baseline + (E_ss - E_baseline) * (1 - math.exp(-k_deg * t_hours))
            else:
                # Deinduction phase
                t_off = (day - days_on_inducer) * 24
                # First calculate the level at the time inducer was stopped
                t_on_hours = days_on_inducer * 24
                E_at_stop = E_baseline + (E_ss - E_baseline) * (1 - math.exp(-k_deg * t_on_hours))
                # Then decay back to baseline
                E = E_baseline + (E_at_stop - E_baseline) * math.exp(-k_deg * t_off)

            fold_current = E / E_baseline
            cl_modifier = fold_current  # CL scales linearly with enzyme level

            enzyme_levels.append({
                "day": round(day, 2),
                "enzyme_level_pmol_mg": round(E, 1),
                "fold_vs_baseline": round(fold_current, 2),
                "clearance_modifier": round(cl_modifier, 2),
            })
            clearance_modifiers.append(cl_modifier)

        # Time to 50% and 90% of steady-state induction
        t_50 = cyp_info.half_life_h / 24  # 1 half-life
        t_90 = 3.32 * cyp_info.half_life_h / 24  # 3.32 half-lives
        t_full = 5 * cyp_info.half_life_h / 24  # ~5 half-lives

        results_per_cyp[cyp_name] = {
            "enzyme_half_life_h": cyp_info.half_life_h,
            "baseline_abundance": E_baseline,
            "steady_state_fold": fold_ss,
            "time_to_50_percent_days": round(t_50, 1),
            "time_to_90_percent_days": round(t_90, 1),
            "time_to_full_induction_days": round(t_full, 1),
            "current_fold": round(enzyme_levels[-1]["fold_vs_baseline"], 2),
            "time_course": enzyme_levels[::max(1, len(enzyme_levels)//50)],  # Limit points
        }

    return {
        "inducer": profile.name,
        "mechanism": profile.mechanism,
        "clinical_note": profile.clinical_note,
        "days_on": days_on_inducer,
        "days_off": days_off_inducer,
        "cyp_effects": results_per_cyp,
        "warnings": _generate_induction_warnings(profile, days_on_inducer, days_off_inducer),
    }


def _generate_induction_warnings(profile: InducerProfile, days_on: float, days_off: float) -> list[str]:
    """Generate clinical warnings."""
    warnings = []

    if days_on < 3:
        warnings.append(f"Induction is NOT yet at steady state. Full effect requires 5-7 days of continuous {profile.name} use.")

    if days_off > 0 and days_off < 7:
        warnings.append(f"Inducer recently stopped. CYP levels are DECLINING but not yet at baseline. Co-administered drug levels may be RISING → monitor for toxicity.")

    for cyp, fold in profile.target_cyps.items():
        if fold > 3:
            warnings.append(f"{cyp} induced {fold}x → drugs metabolized by {cyp} will have {int((1-1/fold)*100)}% LOWER levels. Dose increase may be needed.")

    return warnings
