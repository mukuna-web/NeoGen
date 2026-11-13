"""
In Vitro to In Vivo Extrapolation (IVIVE)
==========================================
Predicts in vivo hepatic clearance from in vitro CYP microsomal data.

The gold standard for predicting drug clearance in pharma:
1. Measure CLint in human liver microsomes (HLM) or hepatocytes
2. Scale up using MPPGL (mg microsomal protein per gram liver) and liver weight
3. Apply the well-stirred liver model to get in vivo hepatic clearance

Formula:
    CLint,in_vivo = CLint,in_vitro × MPPGL × Liver_weight
    CLh = (Qh × fu × CLint,in_vivo) / (Qh + fu × CLint,in_vivo)

Where:
    Qh = hepatic blood flow (~90 L/h)
    fu = fraction unbound in plasma
    MPPGL = 40 mg/g (human)
    Liver weight = 1800 g (70 kg human)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IVIVEParams:
    """Parameters for IVIVE scaling."""
    # Scaling factors (human reference values)
    MPPGL: float = 40.0            # mg microsomal protein per g liver (human)
    liver_weight_g: float = 1800.0   # Liver weight (g) for 70 kg human
    hepatocellularity: float = 120e6  # Hepatocytes per g liver
    Q_hepatic: float = 90.0         # Hepatic blood flow (L/h)

    # Blood-plasma ratio
    Rb: float = 1.0                 # Blood:plasma concentration ratio

    # Age scaling of MPPGL
    @classmethod
    def for_patient(cls, weight_kg: float = 70, age_years: int = 30, sex: str = "male"):
        p = cls()
        w_ratio = weight_kg / 70.0
        p.liver_weight_g = 1800.0 * w_ratio

        # MPPGL decreases with age
        if age_years > 60:
            p.MPPGL *= max(0.6, 1.0 - (age_years - 60) * 0.008)

        # Sex differences
        if sex == "female":
            p.liver_weight_g *= 0.85
            p.Q_hepatic *= 0.85

        # Hepatic blood flow decreases with age
        if age_years > 40:
            p.Q_hepatic *= max(0.6, 1.0 - (age_years - 40) * 0.005)

        p.Q_hepatic *= w_ratio ** 0.75  # Allometric scaling

        return p


# Known in vitro clearance values (µL/min/mg protein or µL/min/10^6 cells)
CLINT_DATABASE = {
    # drug: (CLint_microsomes µL/min/mg, CLint_hepatocytes µL/min/M cells, source)
    "aspirin":       (None, None, "esterase — not CYP mediated"),
    "caffeine":      (3.2, 2.5, "CYP1A2"),
    "paracetamol":   (12.0, 8.0, "UGT/SULT/CYP2E1"),
    "ibuprofen":     (8.5, 6.0, "CYP2C9"),
    "diclofenac":    (32.0, 25.0, "CYP2C9/UGT2B7"),
    "warfarin":      (0.8, 0.5, "CYP2C9/CYP3A4"),
    "midazolam":     (250.0, 180.0, "CYP3A4 — high extraction"),
    "atorvastatin":  (50.0, 35.0, "CYP3A4/UGT"),
    "simvastatin":   (180.0, 120.0, "CYP3A4"),
    "metformin":     (1.0, 0.8, "Minimal metabolism — renal elimination"),
    "propranolol":   (85.0, 60.0, "CYP2D6/CYP1A2"),
    "omeprazole":    (40.0, 30.0, "CYP2C19/CYP3A4"),
    "losartan":      (15.0, 10.0, "CYP2C9/CYP3A4"),
    "metoprolol":    (45.0, 30.0, "CYP2D6"),
    "verapamil":     (350.0, 250.0, "CYP3A4 — very high extraction"),
    "nifedipine":    (200.0, 150.0, "CYP3A4"),
    "dextromethorphan": (120.0, 80.0, "CYP2D6/CYP3A4"),
    "codeine":       (25.0, 18.0, "CYP2D6/CYP3A4/UGT2B7"),
    "diazepam":      (3.5, 2.5, "CYP3A4/CYP2C19 — slow"),
    "phenytoin":     (5.0, 3.5, "CYP2C9/CYP2C19"),
}


def well_stirred_model(
    clint_invitro: float,       # µL/min/mg protein (microsomes)
    fu: float = 0.5,            # Fraction unbound in plasma
    params: IVIVEParams = None,
    use_hepatocytes: bool = False,
) -> dict:
    """
    Well-stirred liver model for IVIVE.

    Scales in vitro CLint to in vivo hepatic clearance.
    """
    if params is None:
        params = IVIVEParams()

    if clint_invitro is None or clint_invitro <= 0:
        return {
            "cl_hepatic_L_h": 0.0,
            "extraction_ratio": 0.0,
            "method": "no_data",
        }

    # Scale CLint from in vitro to in vivo
    if use_hepatocytes:
        # Hepatocyte scaling: µL/min/10^6 cells → L/h
        clint_invivo = (clint_invitro * 1e-6 * 60  # µL/min → L/h per 10^6 cells
                       * params.hepatocellularity  # cells/g liver
                       * params.liver_weight_g)     # g liver
    else:
        # Microsomal scaling: µL/min/mg → L/h
        clint_invivo = (clint_invitro * 1e-6 * 60  # µL/min → L/h per mg protein
                       * params.MPPGL               # mg protein/g liver
                       * params.liver_weight_g)      # g liver

    # Well-stirred model
    qh = params.Q_hepatic
    fu_b = fu / params.Rb  # Fraction unbound in blood

    # CLh = (Qh × fu_b × CLint) / (Qh + fu_b × CLint)
    cl_hepatic = (qh * fu_b * clint_invivo) / (qh + fu_b * clint_invivo)

    # Extraction ratio
    eh = cl_hepatic / qh

    # Hepatic bioavailability (first-pass)
    fh = 1.0 - eh

    return {
        "cl_hepatic_L_h": round(cl_hepatic, 3),
        "clint_invivo_L_h": round(clint_invivo, 2),
        "extraction_ratio": round(eh, 3),
        "hepatic_bioavailability": round(fh, 3),
        "scaling_method": "hepatocytes" if use_hepatocytes else "microsomes",
        "fu_used": fu,
        "qh_used": round(qh, 1),
    }


def predict_clearance_ivive(
    smiles: str = "",
    drug_name: str = "",
    fu: float = 0.5,
    weight_kg: float = 70,
    age_years: int = 30,
    sex: str = "male",
) -> dict:
    """
    Predict hepatic clearance using IVIVE if in vitro data is available.

    Falls back to ML-predicted clearance if no in vitro data exists.
    """
    name_lower = drug_name.lower().strip()
    params = IVIVEParams.for_patient(weight_kg, age_years, sex)

    for known_name, (clint_mic, clint_hep, source) in CLINT_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            result = {"drug": known_name, "source": source}

            if clint_mic is not None:
                mic_result = well_stirred_model(clint_mic, fu, params, use_hepatocytes=False)
                result["microsomal"] = mic_result

            if clint_hep is not None:
                hep_result = well_stirred_model(clint_hep, fu, params, use_hepatocytes=True)
                result["hepatocyte"] = hep_result

            # Use hepatocyte result if available (more physiologically relevant)
            if clint_hep is not None:
                result["recommended_cl"] = result["hepatocyte"]["cl_hepatic_L_h"]
                result["recommended_method"] = "hepatocyte_well_stirred"
            elif clint_mic is not None:
                result["recommended_cl"] = result["microsomal"]["cl_hepatic_L_h"]
                result["recommended_method"] = "microsomal_well_stirred"
            else:
                result["recommended_cl"] = None
                result["recommended_method"] = "no_invitro_data"

            logger.info("IVIVE for %s: CLh=%.2f L/h (method=%s)",
                       known_name, result.get("recommended_cl", 0), result["recommended_method"])
            return result

    return {
        "drug": drug_name or "Unknown",
        "source": "no_in_vitro_data",
        "recommended_cl": None,
        "recommended_method": "use_ml_prediction",
    }
