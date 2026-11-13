"""
Pediatric Physiology Model
============================
Models age-dependent changes in physiology from neonate (0) to adult (18 years).

Key features:
1. Enzyme maturation — CYP enzymes develop at different rates after birth
2. Organ growth — organ volumes and blood flows scale with age/weight
3. Body composition — neonates have more water, less fat, less protein
4. Renal maturation — GFR doesn't reach adult levels until ~1 year
5. Protein binding — neonates have lower albumin + different binding affinity

This is critical for pediatric drug dosing — neonates can have 2-10x different
drug levels than adults given the same mg/kg dose.

Reference:
  Edginton et al., Clin Pharmacokinet 2006
  Johnson et al., Clin Pharmacokinet 2005
  Simcyp Pediatric Module
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PediatricPatient:
    """Age-specific physiological parameters."""
    age_years: float
    weight_kg: float
    sex: str = "male"

    def __post_init__(self):
        # Auto-compute weight if not provided
        if self.weight_kg <= 0:
            self.weight_kg = self._estimate_weight()

    def _estimate_weight(self):
        """Estimate weight from age using CDC growth curves (50th percentile)."""
        a = self.age_years
        if a < 0.08:    # Birth
            return 3.5
        elif a < 0.5:   # 0-6 months
            return 3.5 + a * 12
        elif a < 1:     # 6-12 months
            return 7 + (a - 0.5) * 6
        elif a < 6:     # 1-6 years
            return 10 + (a - 1) * 2.3
        elif a < 12:    # 6-12 years
            return 21 + (a - 6) * 3.5
        elif a < 18:    # 12-18 years
            return 42 + (a - 12) * 5
        else:
            return 70  # Adult

    @property
    def bsa(self):
        """Body Surface Area (m²) — Mosteller formula."""
        # For pediatrics, use height estimate
        h = self._estimate_height_cm()
        return math.sqrt(h * self.weight_kg / 3600)

    def _estimate_height_cm(self):
        """Estimate height from age."""
        a = self.age_years
        if a < 0.08:
            return 50
        elif a < 1:
            return 50 + a * 25
        elif a < 6:
            return 75 + (a - 1) * 7
        elif a < 12:
            return 110 + (a - 6) * 6
        elif a < 18:
            return 146 + (a - 12) * 5 if self.sex == "male" else 146 + (a - 12) * 3
        else:
            return 176 if self.sex == "male" else 163


# ════════════════════════════════════════════════════════════════
# CYP ENZYME MATURATION
# ════════════════════════════════════════════════════════════════

def get_cyp_maturation_factor(cyp_name: str, age_years: float) -> float:
    """
    Get CYP enzyme maturation factor (fraction of adult activity).

    Returns 0.0-1.0 where 1.0 = adult level.

    Maturation curves from Edginton 2006, Johnson 2005, Hines 2008:
    - CYP3A4: Reaches adult by ~2 years. But CYP3A7 (fetal form) high at birth
    - CYP1A2: Very low at birth. Reaches adult by ~4 years
    - CYP2D6: Low at birth. Reaches adult by ~5 years
    - CYP2C9: Low at birth. Reaches adult by ~6 months
    - CYP2C19: Very low at birth. Reaches adult by ~10 years (slow!)
    - CYP2E1: Surge after birth. Reaches adult by ~1 year
    """
    a = age_years

    # Hill equation: fraction = age^n / (T50^n + age^n)
    maturation_params = {
        "CYP3A4":  {"T50": 0.5,  "n": 2.0, "birth_frac": 0.30},
        "CYP3A5":  {"T50": 0.5,  "n": 2.0, "birth_frac": 0.30},
        "CYP3A7":  {"T50": 0.2,  "n": 2.0, "birth_frac": 1.50, "decreases": True},
        "CYP1A2":  {"T50": 1.5,  "n": 1.5, "birth_frac": 0.05},
        "CYP2D6":  {"T50": 1.0,  "n": 1.5, "birth_frac": 0.10},
        "CYP2C9":  {"T50": 0.25, "n": 2.0, "birth_frac": 0.15},
        "CYP2C19": {"T50": 3.0,  "n": 1.0, "birth_frac": 0.10},
        "CYP2E1":  {"T50": 0.2,  "n": 3.0, "birth_frac": 0.20},
        "CYP2B6":  {"T50": 1.0,  "n": 1.5, "birth_frac": 0.10},
        "CYP2A6":  {"T50": 2.0,  "n": 1.5, "birth_frac": 0.05},
    }

    params = maturation_params.get(cyp_name, {"T50": 1.0, "n": 1.5, "birth_frac": 0.20})

    if params.get("decreases", False):
        # CYP3A7 decreases with age (fetal enzyme replaced by CYP3A4)
        fraction = params["birth_frac"] * (params["T50"] ** params["n"]) / (a ** params["n"] + params["T50"] ** params["n"])
        return max(fraction, 0.01)

    # Standard maturation (increases with age)
    if a <= 0:
        return params["birth_frac"]

    matured = a ** params["n"] / (params["T50"] ** params["n"] + a ** params["n"])
    fraction = params["birth_frac"] + (1.0 - params["birth_frac"]) * matured

    return min(fraction, 1.0)


# ════════════════════════════════════════════════════════════════
# ORGAN MATURATION
# ════════════════════════════════════════════════════════════════

def get_organ_scaling(age_years: float, weight_kg: float, sex: str = "male") -> dict:
    """
    Get age-specific organ volumes and blood flows.

    Uses allometric scaling + maturation corrections.
    Reference: Edginton et al., Clin Pharmacokinet 2006
    """
    a = age_years
    w = weight_kg
    adult_w = 70.0

    # Organ volume as fraction of body weight (changes with age)
    liver_frac = 0.040 if a > 1 else 0.050  # Neonatal liver is proportionally larger
    kidney_frac = 0.0044 if a > 1 else 0.007
    brain_frac = 0.021 if a > 2 else (0.10 if a < 0.08 else 0.06)  # Brain huge in neonates
    muscle_frac = 0.40 if a > 18 else (0.25 if a < 1 else 0.25 + (a - 1) * 0.009)
    fat_frac = 0.21 if a > 18 else (0.11 if a < 0.5 else 0.11 + min(a, 18) * 0.006)

    # Calculate organ volumes
    volumes = {
        "liver": w * liver_frac * 1000 / 1050,  # Convert to L (density ~1050 g/L)
        "kidney": w * kidney_frac * 1000 / 1050,
        "brain": w * brain_frac * 1000 / 1040,
        "muscle": w * muscle_frac,
        "fat": w * fat_frac,
        "gut": w * 0.020,
        "lung": w * 0.008,
        "skin": w * 0.037,
        "spleen": w * 0.002,
        "plasma": w * 0.045,
    }

    # Cardiac output (allometric scaling)
    # CO scales with weight^0.75
    co_adult = 390  # L/h for 70kg adult
    co = co_adult * (w / adult_w) ** 0.75

    # GFR maturation (doesn't reach adult levels until ~1 year)
    gfr_maturation = 1.0
    if a < 0.08:  # Birth
        gfr_maturation = 0.15  # Neonatal GFR ~15% of adult
    elif a < 0.25:
        gfr_maturation = 0.15 + (a - 0.08) * 3.5
    elif a < 0.5:
        gfr_maturation = 0.75
    elif a < 1:
        gfr_maturation = 0.85
    elif a < 2:
        gfr_maturation = 0.95
    else:
        gfr_maturation = 1.0

    gfr = 7.2 * gfr_maturation * (w / adult_w) ** 0.75  # L/h

    # Albumin concentration (lower in neonates)
    albumin_gl = 40.0  # Adult
    if a < 0.08:
        albumin_gl = 30.0
    elif a < 1:
        albumin_gl = 33.0
    elif a < 6:
        albumin_gl = 37.0

    # AAG concentration (lower in neonates, higher in inflammation)
    aag_gl = 0.8
    if a < 0.08:
        aag_gl = 0.25
    elif a < 1:
        aag_gl = 0.40

    # Body water (higher in neonates)
    total_body_water_frac = 0.60  # Adult
    if a < 0.08:
        total_body_water_frac = 0.80
    elif a < 1:
        total_body_water_frac = 0.70
    elif a < 6:
        total_body_water_frac = 0.65

    return {
        "volumes": volumes,
        "cardiac_output_lh": round(co, 1),
        "gfr_lh": round(gfr, 2),
        "gfr_maturation": round(gfr_maturation, 2),
        "albumin_gl": albumin_gl,
        "aag_gl": aag_gl,
        "total_body_water_fraction": total_body_water_frac,
        "age_years": a,
        "weight_kg": w,
    }


# ════════════════════════════════════════════════════════════════
# PEDIATRIC DOSE ADJUSTMENT
# ════════════════════════════════════════════════════════════════

def recommend_pediatric_dose(
    adult_dose_mg: float,
    age_years: float,
    weight_kg: float,
    primary_cyp: str = "CYP3A4",
    primary_elimination: str = "hepatic",
) -> dict:
    """
    Recommend pediatric dose based on maturation-adjusted clearance.

    Methods:
    1. Simple weight-based: dose × (child_weight / 70)
    2. BSA-based: dose × (child_BSA / 1.73)
    3. Maturation-adjusted: considers CYP maturation + renal maturation
    """
    patient = PediatricPatient(age_years, weight_kg)
    weight_kg = patient.weight_kg  # Use auto-estimated if 0
    organs = get_organ_scaling(age_years, weight_kg)

    # Weight-based
    dose_weight = adult_dose_mg * (weight_kg / 70.0)

    # BSA-based
    dose_bsa = adult_dose_mg * (patient.bsa / 1.73)

    # Maturation-adjusted
    if primary_elimination == "hepatic":
        mat_factor = get_cyp_maturation_factor(primary_cyp, age_years)
        # Also scale by liver size relative to body weight
        liver_scale = organs["volumes"]["liver"] / (weight_kg * 0.026)  # Adult liver ~2.6% BW
        effective_cl_ratio = mat_factor * liver_scale * (weight_kg / 70.0) ** 0.75
    else:
        effective_cl_ratio = organs["gfr_maturation"] * (weight_kg / 70.0) ** 0.75

    dose_maturation = adult_dose_mg * effective_cl_ratio

    # Pick the most conservative (lowest) dose for safety
    recommended = min(dose_weight, dose_bsa, dose_maturation)

    return {
        "adult_dose_mg": adult_dose_mg,
        "age_years": age_years,
        "weight_kg": weight_kg,
        "bsa_m2": round(patient.bsa, 2),
        "dose_weight_based_mg": round(dose_weight, 1),
        "dose_bsa_based_mg": round(dose_bsa, 1),
        "dose_maturation_adjusted_mg": round(dose_maturation, 1),
        "recommended_dose_mg": round(recommended, 1),
        "cyp_maturation_factor": round(get_cyp_maturation_factor(primary_cyp, age_years), 2),
        "gfr_maturation_factor": round(organs["gfr_maturation"], 2),
        "primary_cyp": primary_cyp,
        "primary_elimination": primary_elimination,
        "warning": ("NEONATAL — Extreme caution. Immature hepatic and renal function." if age_years < 0.08
                    else "INFANT — Dose adjustment critical. Monitor closely." if age_years < 1
                    else "CHILD — Standard pediatric dosing with monitoring." if age_years < 12
                    else "ADOLESCENT — Approaching adult physiology."),
        "cyp_maturation_table": {
            cyp: round(get_cyp_maturation_factor(cyp, age_years), 2)
            for cyp in ["CYP3A4", "CYP1A2", "CYP2D6", "CYP2C9", "CYP2C19", "CYP2E1"]
        },
    }
