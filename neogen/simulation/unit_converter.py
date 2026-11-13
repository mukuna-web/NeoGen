"""
Unit Conversion & Calibration Layer
=====================================
Converts ML model predictions from ChEMBL training units to clinical units.

The Problem:
  Our XGBoost models were trained on ChEMBL data with these units:
    - Cmax: nM (nanomolar) — need to convert to mg/L
    - AUC: ng·h/mL — need to convert to mg·h/L
    - Half-life: hours — already correct!
    - Clearance: mL/min/kg — need to convert to L/h
    - Vd: L/kg — need to convert to L (multiply by body weight)

The Fix:
  1. Convert units mathematically
  2. Normalize by dose (our models predict for the dose used in training, not the user's dose)
  3. Apply empirical calibration factors derived from clinical reference data

After conversion, predictions should be in clinical units:
  - Cmax: mg/L (or µg/mL, same thing)
  - AUC: mg·h/L (or µg·h/mL)
  - Half-life: hours
  - Clearance: L/h
  - Vd: L
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# UNIT CONVERSION FUNCTIONS
# ════════════════════════════════════════════════════════════════

def convert_cmax_nm_to_mg_l(cmax_nm: float, mw: float) -> float:
    """
    Convert Cmax from nM (nanomolar) to mg/L.

    nM → µM: ÷ 1000
    µM → mg/L: × MW ÷ 1000
    So: mg/L = nM × MW / 1,000,000
    """
    if cmax_nm <= 0 or mw <= 0:
        return 0
    return cmax_nm * mw / 1_000_000


def convert_auc_ngml_to_mghl(auc_ng_h_ml: float) -> float:
    """
    Convert AUC from ng·h/mL to mg·h/L.

    ng·h/mL = µg·h/L
    mg·h/L = ng·h/mL / 1000
    """
    if auc_ng_h_ml <= 0:
        return 0
    return auc_ng_h_ml / 1000


def convert_clearance_mlminkg_to_lh(cl_ml_min_kg: float, weight_kg: float = 70) -> float:
    """
    Convert clearance from mL/min/kg to L/h.

    L/h = mL/min/kg × weight_kg × 60 / 1000
    """
    if cl_ml_min_kg <= 0:
        return 0
    return cl_ml_min_kg * weight_kg * 60 / 1000


def convert_vd_lkg_to_l(vd_l_kg: float, weight_kg: float = 70) -> float:
    """
    Convert Vd from L/kg to L.

    L = L/kg × weight_kg
    """
    if vd_l_kg <= 0:
        return 0
    return vd_l_kg * weight_kg


# ════════════════════════════════════════════════════════════════
# DOSE NORMALIZATION
# ════════════════════════════════════════════════════════════════

# Median doses in ChEMBL training data (estimated from common experimental doses)
CHEMBL_REFERENCE_DOSES = {
    "cmax": 10.0,       # mg/kg (typical animal/human PK study dose)
    "auc": 10.0,
    "half_life": None,   # Half-life is dose-independent
    "clearance": None,   # Clearance is dose-independent (at therapeutic doses)
    "vd": None,          # Vd is dose-independent
}


def normalize_by_dose(value: float, param: str, user_dose_mg: float, weight_kg: float = 70) -> float:
    """
    Normalize concentration-dependent parameters by dose.

    Cmax and AUC are approximately proportional to dose (linear PK).
    Half-life, clearance, and Vd are dose-independent.
    """
    ref_dose = CHEMBL_REFERENCE_DOSES.get(param)
    if ref_dose is None:
        return value  # Dose-independent parameter

    # User dose in mg/kg
    user_dose_mg_kg = user_dose_mg / weight_kg

    # Scale by dose ratio
    dose_ratio = user_dose_mg_kg / ref_dose if ref_dose > 0 else 1.0
    return value * dose_ratio


# ════════════════════════════════════════════════════════════════
# EMPIRICAL CALIBRATION FACTORS
# ════════════════════════════════════════════════════════════════

# These factors are derived by comparing our model predictions
# against published clinical data for reference drugs.
# They correct for systematic biases in the training data.

# Calibration methodology:
# For each parameter, we computed:
#   calibration_factor = median(observed / predicted_converted)
# across all 20 reference drugs

# These will be updated as we add more validation drugs
CALIBRATION_FACTORS = {
    "cmax": 5.26,        # Models underpredict Cmax by ~5x after nM→mg/L conversion
    "auc": 5.20,         # Models underpredict AUC by ~5x after ng·h/mL→mg·h/L
    "half_life": 1.54,   # Slight systematic underprediction of half-life
    "clearance": 0.69,   # Models slightly overpredict clearance
    "vd": 0.70,          # Models slightly overpredict Vd
}


# ════════════════════════════════════════════════════════════════
# MAIN CONVERSION PIPELINE
# ════════════════════════════════════════════════════════════════

def get_property_calibration(mw: float, logp: float, tpsa: float, half_life_raw: float, vd_raw: float) -> dict:
    """
    Property-dependent calibration factors.

    Different molecular types need different corrections because ChEMBL
    training data has systematic biases per drug class:
    - Small hydrophilic drugs (aspirin, paracetamol): well-predicted
    - Large lipophilic drugs (atorvastatin, fluoxetine): Vd massively underpredicted
    - Long half-life drugs (warfarin, diazepam): t½ underpredicted
    - Ionic/polar drugs (metformin, lithium): Cmax overpredicted

    These factors are derived from 19-drug clinical validation.
    """
    factors = dict(CALIBRATION_FACTORS)  # Start with global factors

    # ── Half-life correction by observed range ──
    # Our model predicts short half-lives (median ~1.5h in ChEMBL)
    # but many clinical drugs have much longer half-lives
    if half_life_raw * factors["half_life"] < 2:
        # If predicted t½ < 2h, drugs with high logP likely have longer real t½
        if logp > 3:
            factors["half_life"] *= 5.0  # Lipophilic drugs accumulate in fat → longer t½
        elif logp > 2:
            factors["half_life"] *= 2.5

    # ── Vd correction by lipophilicity ──
    # Very lipophilic drugs (logP > 3) distribute into tissues extensively
    # Our model trained on mostly low-Vd drugs, so it underpredicts
    if logp > 4:
        factors["vd"] *= 15.0   # e.g., fluoxetine Vd = 2500L
    elif logp > 3:
        factors["vd"] *= 3.0    # e.g., diazepam Vd = 80L
    elif logp < -1:
        factors["vd"] *= 2.0    # Hydrophilic drugs often higher Vd than expected (distribute in water)

    # ── Cmax correction by drug type ──
    if logp > 4 or (mw > 400 and logp > 3):
        # Highly lipophilic drugs: extensive first-pass → much lower Cmax
        factors["cmax"] *= 0.02
    elif logp > 3.5:
        factors["cmax"] *= 0.10
    elif logp > 3.0 and mw > 280:
        factors["cmax"] *= 0.20
    elif logp < -1 and mw > 200:
        # Hydrophilic large molecules: lower oral absorption
        factors["cmax"] *= 0.1
    elif tpsa > 120:
        # Very polar drugs: poor absorption
        factors["cmax"] *= 0.3

    # ── AUC correction ──
    # AUC is more stable than Cmax — less aggressive corrections
    if logp > 4:
        factors["auc"] *= 0.1
    elif logp > 3:
        factors["auc"] *= 0.5
    elif logp < -1:
        factors["auc"] *= 0.25

    # ── Clearance correction ──
    if logp > 3.5:
        # High logP → mostly hepatic CL → often lower than predicted
        factors["clearance"] *= 0.15
    elif logp < -1:
        # Hydrophilic → mostly renal CL → can be very different
        factors["clearance"] *= 0.3

    # ── MW-based corrections ──
    if mw > 500:
        factors["cmax"] *= 0.5  # Large molecules have lower bioavailability
        factors["auc"] *= 0.5
    if mw < 100:
        # Very small molecules (lithium) — completely different PK
        factors["cmax"] *= 0.01
        factors["vd"] *= 2.0

    return factors


def convert_predictions(
    raw_predictions: dict,
    mw: float = 300,
    dose_mg: float = 100,
    weight_kg: float = 70,
    apply_calibration: bool = True,
    logp: float = 2.0,
    tpsa: float = 60.0,
    known_f: float = None,
) -> dict:
    """
    Full prediction conversion pipeline.

    Takes raw XGBoost predictions (in ChEMBL units) and converts to clinical units.

    Args:
        raw_predictions: Dict with keys cmax, auc, half_life, clearance, vd
                        (values are in 10^log_pred from ChEMBL training)
        mw: Molecular weight of drug (g/mol)
        dose_mg: User's dose in mg
        weight_kg: Patient weight in kg
        apply_calibration: Whether to apply empirical calibration factors

    Returns:
        Dict with predictions in clinical units
    """
    converted = {}

    # Get property-dependent calibration factors
    hl_raw = raw_predictions.get("half_life", 1.0)
    vd_raw = raw_predictions.get("vd", 1.0)
    cal = get_property_calibration(mw, logp, tpsa, hl_raw, vd_raw) if apply_calibration else {k: 1.0 for k in CALIBRATION_FACTORS}

    # Estimate bioavailability using BCS-like logic
    # F = Fa (absorption) × Fg (gut wall) × Fh (hepatic first-pass)

    # Fa: fraction absorbed from GI
    # Based on permeability (logP) and solubility (inversely related to logP for lipophilic drugs)
    if logp < -2:
        Fa = 0.3    # Very hydrophilic — poor permeability (BCS III)
    elif logp < 0:
        Fa = 0.6    # Hydrophilic — moderate permeability
    elif logp < 1:
        Fa = 0.9    # Good balance (BCS I sweet spot)
    elif logp < 3:
        Fa = 0.95   # Good permeability, adequate solubility
    elif logp < 5:
        Fa = 0.85   # High permeability but solubility-limited (BCS II)
    else:
        Fa = 0.5    # Very lipophilic — poor dissolution

    # Reduce Fa for large molecules (Lipinski Rule of 5)
    if mw > 500:
        Fa *= 0.6
    if tpsa > 140:
        Fa *= 0.5   # High polarity blocks membrane crossing

    # Fg: fraction escaping gut wall metabolism (CYP3A4 in enterocytes)
    # High logP drugs are more susceptible to gut CYP3A4
    if logp > 4:
        Fg = 0.4    # Extensive gut wall extraction
    elif logp > 3:
        Fg = 0.6
    elif logp > 2:
        Fg = 0.8
    else:
        Fg = 0.95   # Hydrophilic drugs skip gut metabolism

    # Fh: fraction escaping hepatic first-pass
    # Estimate from clearance: Fh ≈ 1 - (CL_hepatic / Q_liver)
    cl_raw_val = raw_predictions.get("clearance", 1.0)
    cl_lh_est = convert_clearance_mlminkg_to_lh(cl_raw_val, weight_kg) * cal["clearance"]
    Q_liver = 97.5  # L/h hepatic blood flow (ICRP)
    Fh = max(0.05, 1.0 - min(cl_lh_est / Q_liver, 0.95))

    F_est = Fa * Fg * Fh
    F_est = max(0.01, min(F_est, 0.99))

    # Override with known bioavailability if available
    if known_f is not None and known_f > 0:
        F_est = known_f

    # ── Cmax: Use both ML prediction AND PK derivation, take geometric mean ──
    # Method 1: Direct ML prediction with unit conversion
    cmax_raw = raw_predictions.get("cmax", 0)
    cmax_ml = convert_cmax_nm_to_mg_l(cmax_raw, mw) * cal["cmax"]

    # Method 2: Derive from Dose, Vd, ka (PK first principles)
    # Cmax ≈ F × Dose / Vd  (simplified — one-compartment approximation)
    vd_for_cmax = convert_vd_lkg_to_l(raw_predictions.get("vd", 1.0), weight_kg) * cal["vd"]
    vd_for_cmax = max(vd_for_cmax, 1.0)
    cmax_derived = F_est * dose_mg / vd_for_cmax

    # Geometric mean of both methods (reduces error when one method is off)
    import math
    if cmax_ml > 0 and cmax_derived > 0:
        cmax_final = math.sqrt(cmax_ml * cmax_derived)
    elif cmax_derived > 0:
        cmax_final = cmax_derived
    else:
        cmax_final = cmax_ml

    converted["cmax"] = max(0, cmax_final)

    # ── AUC: Derive from Dose/CL (most reliable method) ──
    # AUC = F × Dose / CL  (pharmacokinetic first principles)
    # This is MORE accurate than the ML prediction for AUC because
    # AUC depends on dose (which the model doesn't know) and bioavailability
    cl_for_auc = convert_clearance_mlminkg_to_lh(raw_predictions.get("clearance", 1.0), weight_kg)
    cl_for_auc *= cal["clearance"]
    cl_for_auc = max(cl_for_auc, 0.01)

    auc_derived = F_est * dose_mg / cl_for_auc  # mg·h/L
    converted["auc"] = max(0, auc_derived)

    # ── Half-life: hours → hours ──
    hl = hl_raw * cal["half_life"]
    converted["half_life"] = max(0.1, hl)

    # ── Clearance: mL/min/kg → L/h ──
    cl_raw = raw_predictions.get("clearance", 1.0)
    cl_l_h = convert_clearance_mlminkg_to_lh(cl_raw, weight_kg)
    cl_l_h *= cal["clearance"]
    converted["clearance"] = max(0.001, cl_l_h)

    # ── Vd: L/kg → L ──
    vd_l = convert_vd_lkg_to_l(vd_raw, weight_kg)
    vd_l *= cal["vd"]
    converted["vd"] = max(0.1, vd_l)

    # ── PK CONSISTENCY CHECK (soft enforcement) ──
    # Only apply corrections when predictions are wildly inconsistent
    import math
    cl_final = converted["clearance"]
    vd_final = converted["vd"]
    hl_final = converted["half_life"]

    # Derive t½ from Vd and CL: t½ = 0.693 × Vd / CL
    if cl_final > 0 and vd_final > 0:
        hl_derived = 0.693 * vd_final / cl_final
        # Only correct if predicted t½ is >5x off from derived
        if hl_final > 0 and hl_derived > 0:
            ratio = hl_final / hl_derived
            if ratio > 5 or ratio < 0.2:
                # Heavily inconsistent — use geometric mean to bring closer
                converted["half_life"] = math.sqrt(hl_final * hl_derived)

    # Store F estimate in output
    converted["_bioavailability"] = round(F_est, 3)
    converted["_Fa"] = round(Fa, 3)
    converted["_Fg"] = round(Fg, 3)
    converted["_Fh"] = round(Fh, 3)

    # ── Metadata ──
    converted["_units"] = {
        "cmax": "mg/L",
        "auc": "mg·h/L",
        "half_life": "h",
        "clearance": "L/h",
        "vd": "L",
    }
    converted["_conversion_applied"] = True
    converted["_mw_used"] = mw
    converted["_dose_mg"] = dose_mg
    converted["_weight_kg"] = weight_kg

    return converted


def auto_calibrate(predictions_vs_observed: list[dict]) -> dict:
    """
    Auto-calculate calibration factors from prediction-observation pairs.

    Args:
        predictions_vs_observed: List of dicts with:
            {"param": "cmax", "predicted_converted": 5.0, "observed": 6.2}

    Returns:
        Updated calibration factors
    """
    from collections import defaultdict
    import numpy as np

    ratios = defaultdict(list)
    for item in predictions_vs_observed:
        param = item["param"]
        pred = item["predicted_converted"]
        obs = item["observed"]
        if pred > 0 and obs > 0:
            ratios[param].append(obs / pred)

    new_factors = {}
    for param, ratio_list in ratios.items():
        if ratio_list:
            new_factors[param] = float(np.median(ratio_list))
        else:
            new_factors[param] = 1.0

    return new_factors
