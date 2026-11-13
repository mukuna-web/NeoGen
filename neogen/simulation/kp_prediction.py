"""
Tissue Partition Coefficient (Kp) Prediction
==============================================
Implements the Rodgers & Rowland (2005/2006) method for predicting
tissue-to-plasma partition coefficients.

This is the gold standard method used by Simcyp, GastroPlus, and PK-Sim.
It accounts for actual tissue composition rather than just logP.

The key insight: Kp depends on how much drug partitions into each tissue
component (water, neutral lipids, phospholipids, proteins) based on the
drug's physicochemical properties (logP, pKa, ionization state, binding).

Two methods:
  Method 1 (Rodgers & Rowland 2005): For moderate-to-strong bases and cations
  Method 2 (Rodgers & Rowland 2006): For acids, neutrals, zwitterions, and weak bases

Both use the same tissue composition data (ICRP 89 + literature).

Reference:
  Rodgers T, Rowland M. J Pharm Sci 2005; 94:1259-1276
  Rodgers T, Rowland M. Pharm Res 2006; 23:56-68
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# TISSUE COMPOSITION DATA (ICRP 89 + Rodgers & Rowland)
# ════════════════════════════════════════════════════════════════

@dataclass
class TissueComposition:
    """Fractional volumes of tissue components."""
    name: str
    f_water: float          # Fraction intracellular water
    f_ew: float             # Fraction extracellular water
    f_neutral_lipid: float  # Fraction neutral lipids
    f_phospholipid: float   # Fraction phospholipids
    f_acidic_phospholipid: float  # Fraction acidic phospholipids (AP)
    pH_iw: float = 7.0     # Intracellular water pH


# Tissue composition from Rodgers & Rowland 2005/2006
TISSUES = {
    "adipose": TissueComposition("Adipose", f_water=0.017, f_ew=0.135, f_neutral_lipid=0.853,
                                  f_phospholipid=0.0023, f_acidic_phospholipid=0.0004, pH_iw=7.0),
    "bone": TissueComposition("Bone", f_water=0.346, f_ew=0.1, f_neutral_lipid=0.017,
                               f_phospholipid=0.0017, f_acidic_phospholipid=0.0009, pH_iw=7.0),
    "brain": TissueComposition("Brain", f_water=0.62, f_ew=0.162, f_neutral_lipid=0.039,
                                f_phospholipid=0.0532, f_acidic_phospholipid=0.0406, pH_iw=7.0),
    "gut": TissueComposition("Gut", f_water=0.282, f_ew=0.265, f_neutral_lipid=0.038,
                              f_phospholipid=0.0125, f_acidic_phospholipid=0.0111, pH_iw=7.0),
    "heart": TissueComposition("Heart", f_water=0.456, f_ew=0.32, f_neutral_lipid=0.014,
                                f_phospholipid=0.0111, f_acidic_phospholipid=0.0127, pH_iw=7.0),
    "kidney": TissueComposition("Kidney", f_water=0.483, f_ew=0.273, f_neutral_lipid=0.012,
                                 f_phospholipid=0.0124, f_acidic_phospholipid=0.0113, pH_iw=7.0),
    "liver": TissueComposition("Liver", f_water=0.573, f_ew=0.161, f_neutral_lipid=0.014,
                                f_phospholipid=0.024, f_acidic_phospholipid=0.0488, pH_iw=7.0),
    "lung": TissueComposition("Lung", f_water=0.446, f_ew=0.336, f_neutral_lipid=0.003,
                               f_phospholipid=0.009, f_acidic_phospholipid=0.0068, pH_iw=7.0),
    "muscle": TissueComposition("Muscle", f_water=0.63, f_ew=0.118, f_neutral_lipid=0.01,
                                 f_phospholipid=0.0073, f_acidic_phospholipid=0.0064, pH_iw=7.0),
    "skin": TissueComposition("Skin", f_water=0.382, f_ew=0.291, f_neutral_lipid=0.06,
                               f_phospholipid=0.0044, f_acidic_phospholipid=0.0033, pH_iw=7.0),
    "spleen": TissueComposition("Spleen", f_water=0.579, f_ew=0.207, f_neutral_lipid=0.0077,
                                 f_phospholipid=0.0113, f_acidic_phospholipid=0.0091, pH_iw=7.0),
    "pancreas": TissueComposition("Pancreas", f_water=0.463, f_ew=0.12, f_neutral_lipid=0.041,
                                   f_phospholipid=0.009, f_acidic_phospholipid=0.0046, pH_iw=7.0),
    "rbc": TissueComposition("RBC", f_water=0.603, f_ew=0.0, f_neutral_lipid=0.0017,
                              f_phospholipid=0.0029, f_acidic_phospholipid=0.0044, pH_iw=7.22),
}

# Plasma composition
PLASMA_PH = 7.4
PLASMA_F_WATER = 0.945
PLASMA_F_NEUTRAL_LIPID = 0.0023
PLASMA_F_PHOSPHOLIPID = 0.0009


def predict_kp_all_tissues(
    logp: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
    fu_plasma: float = 0.5,
    bp_ratio: float = 1.0,
) -> dict:
    """
    Predict Kp for all tissues using Rodgers & Rowland method.

    Args:
        logp: Partition coefficient (log scale)
        pka: Acid dissociation constant
        drug_type: "acid", "base", "neutral", "zwitterion"
        fu_plasma: Fraction unbound in plasma
        bp_ratio: Blood-to-plasma ratio

    Returns:
        Dict of tissue_name: Kp values
    """
    # LogP to partition coefficients
    P = 10 ** logp  # octanol:water partition coefficient
    logP_vow = logp - 0.7  # vegetable oil:water (for neutral lipids)
    P_vow = 10 ** logP_vow

    # Ionization at plasma pH
    if drug_type == "base":
        # For bases: fraction unionized at pH 7.4
        # fu_ion = 1 / (1 + 10^(pka - pH))
        ratio_plasma = 10 ** (pka - PLASMA_PH)
    elif drug_type == "acid":
        ratio_plasma = 10 ** (PLASMA_PH - pka)
    else:
        ratio_plasma = 0  # Neutral — fully unionized

    results = {}

    for tissue_name, tc in TISSUES.items():
        # Ionization ratio in tissue (different pH)
        if drug_type == "base":
            ratio_tissue = 10 ** (pka - tc.pH_iw)
        elif drug_type == "acid":
            ratio_tissue = 10 ** (tc.pH_iw - pka)
        else:
            ratio_tissue = 0

        if drug_type in ("neutral", "acid", "zwitterion"):
            # Method 2: Rodgers & Rowland 2006
            # Kp = (f_ew + (1 + ratio_tissue)/(1 + ratio_plasma) × f_iw
            #       + P_vow × f_nl + (0.3P + 0.7) × f_pl) / (f_ew_p + P_vow × f_nl_p + (0.3P + 0.7) × f_pl_p) × 1/fu
            # Simplified version:

            kp_u = (tc.f_ew
                    + (1 + ratio_tissue) / max(1 + ratio_plasma, 0.01) * tc.f_water
                    + P_vow * tc.f_neutral_lipid
                    + (0.3 * P + 0.7) * tc.f_phospholipid)

            kp_u_plasma = (PLASMA_F_WATER
                           + P_vow * PLASMA_F_NEUTRAL_LIPID
                           + (0.3 * P + 0.7) * PLASMA_F_PHOSPHOLIPID)

            kp = kp_u / (kp_u_plasma * fu_plasma) if kp_u_plasma * fu_plasma > 0 else 1.0

        else:
            # Method 1: Rodgers & Rowland 2005 (for strong bases)
            # Includes binding to acidic phospholipids (AP)
            ka_ap = (P ** 0.65) * 0.5  # Association constant for AP binding

            kp_u = (tc.f_ew
                    + (1 + ratio_tissue) / max(1 + ratio_plasma, 0.01) * (tc.f_water + ka_ap * tc.f_acidic_phospholipid)
                    + P_vow * tc.f_neutral_lipid
                    + (0.3 * P + 0.7) * tc.f_phospholipid)

            kp_u_plasma = (PLASMA_F_WATER
                           + P_vow * PLASMA_F_NEUTRAL_LIPID
                           + (0.3 * P + 0.7) * PLASMA_F_PHOSPHOLIPID)

            kp = kp_u / (kp_u_plasma * fu_plasma) if kp_u_plasma * fu_plasma > 0 else 1.0

        # Clip to reasonable range
        kp = max(0.01, min(kp, 1000))
        results[tissue_name] = round(kp, 3)

    # Calculate Vss (steady-state volume of distribution)
    # Vss = Vp + Σ(Vt × Kp,t) where Vp = plasma volume
    tissue_volumes = {
        "adipose": 18.2, "bone": 4.0, "brain": 1.45, "gut": 1.7,
        "heart": 0.33, "kidney": 0.31, "liver": 1.8, "lung": 0.50,
        "muscle": 29.0, "skin": 3.3, "spleen": 0.15, "pancreas": 0.10,
    }
    vp = 3.0  # Plasma volume (L)
    vss = vp
    for tissue, vol in tissue_volumes.items():
        kp_t = results.get(tissue, 1.0)
        vss += vol * kp_t

    # RBC partition
    kp_rbc = results.get("rbc", 1.0)
    vrbc = 2.5  # RBC volume
    vss += vrbc * kp_rbc

    results["_vss_predicted_L"] = round(vss, 1)
    results["_method"] = "Rodgers & Rowland 2005/2006"
    results["_drug_type"] = drug_type

    return results


def get_kp_for_pbpk(
    logp: float,
    pka: float = 7.0,
    drug_type: str = "neutral",
    fu_plasma: float = 0.5,
) -> dict:
    """
    Get Kp values formatted for direct use in PBPK ODE.

    Returns dict with keys matching PBPK compartment names.
    """
    all_kp = predict_kp_all_tissues(logp, pka, drug_type, fu_plasma)

    return {
        "Kp_gut": all_kp.get("gut", 1.0),
        "Kp_liver": all_kp.get("liver", 1.5),
        "Kp_kidney": all_kp.get("kidney", 1.0),
        "Kp_brain": all_kp.get("brain", 0.3),
        "Kp_muscle": all_kp.get("muscle", 0.5),
        "Kp_fat": all_kp.get("adipose", 0.2),
        "Kp_lung": all_kp.get("lung", 0.8),
        "Kp_spleen": all_kp.get("spleen", 1.2),
        "Kp_skin": all_kp.get("skin", 0.5),
        "Kp_bone": all_kp.get("bone", 0.3),
        "Kp_rbc": all_kp.get("rbc", 1.0),
        "Kp_gut_wall": all_kp.get("gut", 1.0),
        "Kp_pancreas": all_kp.get("pancreas", 0.8),
        "Vss_predicted": all_kp.get("_vss_predicted_L", 50),
        "_method": "Rodgers_Rowland",
    }
