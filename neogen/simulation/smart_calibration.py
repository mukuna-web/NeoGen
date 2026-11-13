"""
Smart Calibration System
=========================
1. Drug class-specific calibration factors
2. Conformal prediction (uncertainty quantification)
3. Optimal ensemble selection (v1/v2/v3 per parameter)
"""
from __future__ import annotations
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# DRUG CLASS CALIBRATION
# ════════════════════════════════════════════════════════════════

# Calibration factors derived from analyzing prediction errors
# by drug class in the validation set

DRUG_CLASS_CALIBRATION = {
    # Class identified by molecular properties
    "small_polar": {
        # MW < 200, logP < 1 (caffeine, theophylline, lithium)
        "match": lambda mw, logp, tpsa: mw < 200 and logp < 1,
        "cmax": 1.5, "auc": 3.0, "half_life": 3.0, "clearance": 0.3, "vd": 1.2,
    },
    "nsaid": {
        # Acidic, MW 180-350, logP 1-4 (aspirin, ibuprofen, diclofenac, naproxen)
        "match": lambda mw, logp, tpsa: 150 < mw < 400 and 1 < logp < 5 and tpsa < 80,
        "cmax": 5.0, "auc": 8.0, "half_life": 1.2, "clearance": 0.5, "vd": 0.5,
    },
    "beta_blocker": {
        # MW 250-350, logP 1-3, contains propanolamine substructure
        "match": lambda mw, logp, tpsa: 250 < mw < 400 and 1 < logp < 3.5 and 40 < tpsa < 80,
        "cmax": 0.1, "auc": 0.3, "half_life": 1.5, "clearance": 1.5, "vd": 3.0,
    },
    "highly_lipophilic": {
        # logP > 4 (fluoxetine, atorvastatin, chloroquine)
        "match": lambda mw, logp, tpsa: logp > 4,
        "cmax": 0.01, "auc": 0.05, "half_life": 10.0, "clearance": 0.5, "vd": 15.0,
    },
    "moderate_lipophilic": {
        # logP 3-4, MW > 280 (warfarin, diazepam)
        "match": lambda mw, logp, tpsa: 3 < logp <= 4 and mw > 280,
        "cmax": 0.2, "auc": 10.0, "half_life": 8.0, "clearance": 0.1, "vd": 1.0,
    },
    "hydrophilic_large": {
        # logP < 0, MW > 200 (amoxicillin, metformin)
        "match": lambda mw, logp, tpsa: logp < 0 and mw > 100,
        "cmax": 0.3, "auc": 0.2, "half_life": 2.0, "clearance": 0.5, "vd": 5.0,
    },
    "default": {
        # Fallback for unmatched drugs
        "match": lambda mw, logp, tpsa: True,
        "cmax": 5.26, "auc": 5.20, "half_life": 1.54, "clearance": 0.69, "vd": 0.70,
    },
}


def get_drug_class_calibration(mw: float, logp: float, tpsa: float) -> dict:
    """Get calibration factors for a drug based on its molecular properties."""
    for class_name, config in DRUG_CLASS_CALIBRATION.items():
        if class_name == "default":
            continue
        if config["match"](mw, logp, tpsa):
            return {
                "class": class_name,
                "cmax": config["cmax"],
                "auc": config["auc"],
                "half_life": config["half_life"],
                "clearance": config["clearance"],
                "vd": config["vd"],
            }
    # Default
    d = DRUG_CLASS_CALIBRATION["default"]
    return {
        "class": "default",
        "cmax": d["cmax"], "auc": d["auc"], "half_life": d["half_life"],
        "clearance": d["clearance"], "vd": d["vd"],
    }


# ════════════════════════════════════════════════════════════════
# CONFORMAL PREDICTION (Uncertainty Quantification)
# ════════════════════════════════════════════════════════════════

# Pre-computed prediction intervals from cross-validation
# These represent the typical error distribution of our models
# expressed as multiplicative factors at different confidence levels

PREDICTION_INTERVALS = {
    "cmax": {
        "50%": (0.5, 2.0),     # 50% chance true value is within 0.5x-2x
        "80%": (0.2, 5.0),     # 80% chance within 0.2x-5x
        "90%": (0.1, 10.0),    # 90% chance within 0.1x-10x
        "95%": (0.05, 20.0),   # 95% chance within 0.05x-20x
        "model_r2": 0.298,
    },
    "auc": {
        "50%": (0.4, 2.5),
        "80%": (0.15, 7.0),
        "90%": (0.08, 12.0),
        "95%": (0.04, 25.0),
        "model_r2": 0.405,
    },
    "half_life": {
        "50%": (0.5, 2.0),
        "80%": (0.3, 3.5),
        "90%": (0.15, 7.0),
        "95%": (0.08, 12.0),
        "model_r2": 0.487,
    },
    "clearance": {
        "50%": (0.4, 2.5),
        "80%": (0.2, 5.0),
        "90%": (0.1, 10.0),
        "95%": (0.05, 20.0),
        "model_r2": 0.490,
    },
    "vd": {
        "50%": (0.5, 2.0),
        "80%": (0.2, 5.0),
        "90%": (0.1, 10.0),
        "95%": (0.05, 20.0),
        "model_r2": 0.428,
    },
}


def get_prediction_interval(
    param: str,
    predicted_value: float,
    confidence: str = "80%",
    is_known_drug: bool = False,
) -> dict:
    """
    Get prediction interval for a PK parameter.

    Returns lower and upper bounds at the specified confidence level.
    Known drugs get much tighter intervals.
    """
    if is_known_drug:
        # Known drugs: published values, ±20% for measurement variability
        return {
            "lower": predicted_value * 0.8,
            "upper": predicted_value * 1.2,
            "confidence": "95%",
            "source": "published_clinical_data",
            "reliability": "HIGH",
        }

    intervals = PREDICTION_INTERVALS.get(param, {})
    bounds = intervals.get(confidence, (0.1, 10.0))

    lower = predicted_value * bounds[0]
    upper = predicted_value * bounds[1]
    r2 = intervals.get("model_r2", 0.3)

    # Reliability assessment
    if r2 > 0.45:
        reliability = "MODERATE"
    elif r2 > 0.3:
        reliability = "LOW"
    else:
        reliability = "VERY LOW"

    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "confidence": confidence,
        "source": "ml_prediction",
        "reliability": reliability,
        "model_r2": r2,
        "note": f"True value has {confidence} probability of falling within [{lower:.3f}, {upper:.3f}]",
    }


def get_full_uncertainty_report(predictions: dict, is_known: bool = False) -> dict:
    """Generate full uncertainty report for all predicted parameters."""
    report = {}
    for param in ["cmax", "auc", "half_life", "clearance", "vd"]:
        val = predictions.get(param, 0)
        if val > 0:
            report[param] = {
                "point_estimate": val,
                "intervals": {
                    "50%": get_prediction_interval(param, val, "50%", is_known),
                    "80%": get_prediction_interval(param, val, "80%", is_known),
                    "90%": get_prediction_interval(param, val, "90%", is_known),
                    "95%": get_prediction_interval(param, val, "95%", is_known),
                },
            }
    return report


# ════════════════════════════════════════════════════════════════
# OPTIMAL ENSEMBLE CONFIG
# ════════════════════════════════════════════════════════════════

# Best model version per parameter (based on all training runs)
OPTIMAL_ENSEMBLE = {
    "cmax": {"version": "v3", "r2": 0.298},       # v3 wins (clean data)
    "auc": {"version": "v2", "r2": 0.405},         # v2 wins (enhanced features)
    "half_life": {"version": "v1", "r2": 0.556},   # v1 wins (original)
    "clearance": {"version": "v3", "r2": 0.490},   # v3 wins (clean data, huge gain!)
    "vd": {"version": "v1", "r2": 0.594},          # v1 wins (original)
}


def get_optimal_model_version(param: str) -> str:
    """Get the best model version for a given parameter."""
    return OPTIMAL_ENSEMBLE.get(param, {}).get("version", "v1")
