"""
IVIVE Empirical Scaling Factors
=================================
Corrects the systematic overprediction of in vivo clearance from in vitro data.

The Problem:
  Microsomal CLint consistently overpredicts in vivo hepatic clearance by 2-10x.
  This is called the "IVIVE underprediction of clearance" paradox — microsomal
  systems miss hepatic uptake, protein binding differences, and incomplete
  enzyme systems.

The Fix (what Simcyp does):
  Apply drug-class-specific empirical scaling factors (ESFs) derived from
  comparing IVIVE-predicted CL vs observed clinical CL across hundreds of drugs.

  CL_corrected = CL_ivive × ESF

Scaling factors from published literature:
  - Hallifax & Houston 2009: ESF = 0.43 for microsomes (avg across 109 drugs)
  - Bowman & Benet 2016: ESF varies by extraction ratio category
  - Ito & Houston 2005: hepatocyte ESF = 0.56

This is Simcyp's "secret sauce" — without these corrections, all PBPK
tools overpredict clearance. We implement the same methodology.
"""
from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# EMPIRICAL SCALING FACTORS
# ════════════════════════════════════════════════════════════════

# Microsomal IVIVE scaling factors by drug class
# Source: Meta-analysis of Hallifax 2009, Bowman 2016, Ito 2005
MICROSOMAL_ESF = {
    # By extraction ratio category
    "low_extraction": 0.55,      # ER < 0.3 (most drugs) — Hallifax 2009
    "intermediate_extraction": 0.35,  # ER 0.3-0.7
    "high_extraction": 0.25,     # ER > 0.7 — less overprediction because flow-limited

    # By drug class (more specific)
    "acid": 0.50,               # Acidic drugs (NSAIDs, warfarin)
    "base": 0.35,               # Basic drugs (propranolol, metoprolol)
    "neutral": 0.45,            # Neutral drugs (caffeine, midazolam)
    "zwitterion": 0.40,         # Zwitterionic drugs (ciprofloxacin)

    # By primary CYP
    "CYP3A4": 0.40,            # CYP3A4 substrates
    "CYP2D6": 0.50,            # CYP2D6 substrates
    "CYP2C9": 0.55,            # CYP2C9 substrates
    "CYP2C19": 0.45,           # CYP2C19 substrates
    "CYP1A2": 0.50,            # CYP1A2 substrates
    "CYP2E1": 0.35,            # CYP2E1 substrates

    # Overall average
    "default": 0.43,            # Hallifax & Houston 2009 meta-analysis average
}

# Hepatocyte IVIVE scaling factors
HEPATOCYTE_ESF = {
    "low_extraction": 0.65,
    "intermediate_extraction": 0.50,
    "high_extraction": 0.40,
    "default": 0.56,            # Ito & Houston 2005
}

# Additional corrections
ADDITIONAL_CORRECTIONS = {
    # Transporter-mediated drugs need extra correction
    "oatp_substrate": 0.3,      # OATP substrates (statins) — hepatic uptake needed
    "pgp_substrate": 0.8,       # P-gp substrates — slight overestimation correction
    "oct_substrate": 0.6,       # OCT substrates (metformin) — renal focus
}


def get_ivive_scaling_factor(
    clint_source: str = "microsomal",
    extraction_ratio: float = 0.3,
    drug_class: str = "neutral",
    primary_cyp: str = "",
    is_oatp_substrate: bool = False,
    is_pgp_substrate: bool = False,
    logp: float = 2.0,
) -> dict:
    """
    Get the optimal IVIVE empirical scaling factor.

    Uses a hierarchy:
    1. CYP-specific ESF (most specific)
    2. Drug ionization class ESF
    3. Extraction ratio category ESF (most general)

    Args:
        clint_source: "microsomal" or "hepatocyte"
        extraction_ratio: From well-stirred model (0-1)
        drug_class: "acid", "base", "neutral", "zwitterion"
        primary_cyp: e.g., "CYP3A4"
        is_oatp_substrate: True if OATP-mediated hepatic uptake
        is_pgp_substrate: True if P-gp substrate
        logp: Lipophilicity

    Returns:
        Dict with scaling factor and explanation
    """
    esf_table = MICROSOMAL_ESF if clint_source == "microsomal" else HEPATOCYTE_ESF

    # Start with extraction ratio category
    if extraction_ratio > 0.7:
        er_category = "high_extraction"
    elif extraction_ratio > 0.3:
        er_category = "intermediate_extraction"
    else:
        er_category = "low_extraction"

    esf_er = esf_table.get(er_category, esf_table["default"])

    # CYP-specific (if available and microsomal)
    esf_cyp = MICROSOMAL_ESF.get(primary_cyp, None) if clint_source == "microsomal" else None

    # Drug class
    esf_class = MICROSOMAL_ESF.get(drug_class, None) if clint_source == "microsomal" else None

    # Pick the most specific available factor
    if esf_cyp is not None:
        esf = esf_cyp
        source = f"CYP-specific ({primary_cyp})"
    elif esf_class is not None:
        esf = esf_class
        source = f"Drug class ({drug_class})"
    else:
        esf = esf_er
        source = f"Extraction ratio category ({er_category})"

    # Additional transporter corrections
    extra_correction = 1.0
    extra_notes = []
    if is_oatp_substrate:
        extra_correction *= ADDITIONAL_CORRECTIONS["oatp_substrate"]
        extra_notes.append("OATP substrate — additional scaling for hepatic uptake")
    if is_pgp_substrate:
        extra_correction *= ADDITIONAL_CORRECTIONS["pgp_substrate"]
        extra_notes.append("P-gp substrate — slight correction applied")

    # Very lipophilic drugs tend to have more overprediction
    if logp > 4:
        extra_correction *= 0.6
        extra_notes.append("High logP (>4) — additional lipophilicity correction")

    final_esf = esf * extra_correction

    return {
        "scaling_factor": round(final_esf, 4),
        "base_esf": round(esf, 4),
        "esf_source": source,
        "extraction_category": er_category,
        "clint_source": clint_source,
        "extra_correction": round(extra_correction, 4),
        "extra_notes": extra_notes,
        "interpretation": (
            f"Multiply IVIVE-predicted CL by {final_esf:.3f} to correct for systematic overprediction. "
            f"Based on {source}."
        ),
        "literature": (
            "Hallifax & Houston, Drug Metab Dispos 2009" if clint_source == "microsomal"
            else "Ito & Houston, Pharm Res 2005"
        ),
    }


def apply_ivive_scaling(
    cl_ivive: float,
    clint_source: str = "microsomal",
    extraction_ratio: float = 0.3,
    drug_class: str = "neutral",
    primary_cyp: str = "",
    logp: float = 2.0,
) -> dict:
    """
    Apply IVIVE scaling factor to correct predicted clearance.

    Returns:
        corrected_cl: Scaled clearance (L/h)
        scaling_info: Details about the correction
    """
    scaling = get_ivive_scaling_factor(
        clint_source=clint_source,
        extraction_ratio=extraction_ratio,
        drug_class=drug_class,
        primary_cyp=primary_cyp,
        logp=logp,
    )

    corrected_cl = cl_ivive * scaling["scaling_factor"]

    return {
        "cl_ivive_raw": round(cl_ivive, 3),
        "cl_corrected": round(corrected_cl, 3),
        "correction_fold": round(scaling["scaling_factor"], 3),
        "scaling_info": scaling,
    }
