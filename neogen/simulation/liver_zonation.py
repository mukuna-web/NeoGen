"""
Liver Zonation Model
=====================
The liver lobule has 3 metabolic zones along the portal-central axis:

Zone 1 (Periportal):
  - First to receive blood from portal vein
  - Highest oxygen concentration (~85 mmHg)
  - Dominant in: gluconeogenesis, urea synthesis, bile acid uptake
  - CYP expression: LOW
  - Toxicity: resistant (good oxygen, good GSH)

Zone 2 (Midlobular):
  - Intermediate between Zone 1 and 3
  - Moderate oxygen (~50 mmHg)
  - Transitional metabolism

Zone 3 (Centrilobular / Perivenous):
  - Last to receive blood — lowest oxygen (~35 mmHg)
  - Dominant in: CYP metabolism (3A4, 2E1, 1A2), lipogenesis, glutamine synthesis
  - Most drug metabolism happens here
  - Most VULNERABLE to toxicity (hypoxia + reactive metabolites + depleted GSH)
  - Paracetamol kills Zone 3 first → ALT rises → necrosis spreads outward

Why this matters:
- Different drugs are metabolized at different rates across zones
- Toxicity patterns are zone-specific (helps predict liver injury type)
- Disease states affect zones differently (cirrhosis destroys Zone 3 first)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ZoneParameters:
    """Parameters for a single liver zone."""
    name: str
    fraction_of_liver: float    # Volume fraction (Zone 1+2+3 = 1.0)
    oxygen_tension: float       # mmHg
    cyp3a4_expression: float    # Relative to average (1.0 = average)
    cyp2e1_expression: float
    cyp1a2_expression: float
    cyp2d6_expression: float
    cyp2c9_expression: float
    cyp2c19_expression: float
    ugt_expression: float       # Phase II glucuronidation
    gsh_concentration: float    # Glutathione (mM) — detox shield
    blood_flow_fraction: float  # Fraction of hepatic blood flow


# ════════════════════════════════════════════════════════════════
# ZONE DEFINITIONS (Healthy Liver)
# ════════════════════════════════════════════════════════════════

HEALTHY_ZONES = {
    "zone_1": ZoneParameters(
        name="Zone 1 (Periportal)",
        fraction_of_liver=0.35,
        oxygen_tension=85,
        cyp3a4_expression=0.3,    # Low CYP in Zone 1
        cyp2e1_expression=0.2,
        cyp1a2_expression=0.3,
        cyp2d6_expression=0.5,
        cyp2c9_expression=0.4,
        cyp2c19_expression=0.4,
        ugt_expression=0.8,       # UGT relatively uniform
        gsh_concentration=8.0,    # High GSH (good detox)
        blood_flow_fraction=0.40,
    ),
    "zone_2": ZoneParameters(
        name="Zone 2 (Midlobular)",
        fraction_of_liver=0.35,
        oxygen_tension=50,
        cyp3a4_expression=1.0,
        cyp2e1_expression=1.0,
        cyp1a2_expression=1.0,
        cyp2d6_expression=1.0,
        cyp2c9_expression=1.0,
        cyp2c19_expression=1.0,
        ugt_expression=1.0,
        gsh_concentration=5.0,
        blood_flow_fraction=0.35,
    ),
    "zone_3": ZoneParameters(
        name="Zone 3 (Centrilobular)",
        fraction_of_liver=0.30,
        oxygen_tension=35,
        cyp3a4_expression=2.0,    # Highest CYP expression!
        cyp2e1_expression=3.0,    # CYP2E1 very concentrated in Zone 3
        cyp1a2_expression=2.0,
        cyp2d6_expression=1.5,
        cyp2c9_expression=1.8,
        cyp2c19_expression=1.8,
        ugt_expression=1.2,
        gsh_concentration=3.0,    # Low GSH — vulnerable to reactive metabolites!
        blood_flow_fraction=0.25,
    ),
}


# ════════════════════════════════════════════════════════════════
# ZONE-SPECIFIC DRUG METABOLISM
# ════════════════════════════════════════════════════════════════

@dataclass
class ZonalMetabolismResult:
    """Results of zone-specific metabolism calculation."""
    zone: str
    cl_contribution: float           # Fraction of total CL from this zone
    metabolite_formation_rate: float  # Relative rate of metabolite formation
    reactive_metabolite_risk: float   # Risk of toxic intermediates (0-1)
    gsh_depletion_rate: float        # Rate of GSH consumption
    oxygen_sufficiency: float        # Whether O2 is sufficient for metabolism (0-1)
    toxicity_vulnerability: float    # Overall toxicity vulnerability (0-1)


def calculate_zonal_metabolism(
    primary_cyp: str,
    cl_hepatic: float,
    drug_name: str = "",
    reactive_metabolite: bool = False,
    disease_state: str = "healthy",
) -> dict:
    """
    Calculate metabolism distribution across liver zones.

    Args:
        primary_cyp: Primary metabolizing enzyme (e.g., "CYP3A4")
        cl_hepatic: Total hepatic clearance (L/h)
        drug_name: Drug name for specific profiles
        reactive_metabolite: Whether drug forms reactive intermediates
        disease_state: "healthy", "cirrhosis_a", "cirrhosis_b", "cirrhosis_c",
                      "nafld", "nash", "alcoholic"

    Returns:
        Dict with per-zone metabolism data and toxicity risk
    """
    zones = _get_zones_for_disease(disease_state)

    # Map CYP name to zone expression attribute
    cyp_attr_map = {
        "CYP3A4": "cyp3a4_expression",
        "CYP2E1": "cyp2e1_expression",
        "CYP1A2": "cyp1a2_expression",
        "CYP2D6": "cyp2d6_expression",
        "CYP2C9": "cyp2c9_expression",
        "CYP2C19": "cyp2c19_expression",
    }

    cyp_attr = cyp_attr_map.get(primary_cyp.upper(), "cyp3a4_expression")

    # Calculate per-zone metabolism
    zone_results = {}
    total_weighted_expression = 0

    for zone_key, zone in zones.items():
        expression = getattr(zone, cyp_attr, 1.0)
        weighted = expression * zone.fraction_of_liver * zone.blood_flow_fraction
        total_weighted_expression += weighted

    for zone_key, zone in zones.items():
        expression = getattr(zone, cyp_attr, 1.0)
        weighted = expression * zone.fraction_of_liver * zone.blood_flow_fraction

        # Fraction of total clearance from this zone
        cl_fraction = weighted / total_weighted_expression if total_weighted_expression > 0 else 0.33

        # Oxygen-limited metabolism (Zone 3 can be O2-limited)
        o2_factor = min(1.0, zone.oxygen_tension / 40.0)

        # Reactive metabolite risk — higher in zones with high CYP + low GSH
        rm_risk = 0.0
        if reactive_metabolite:
            rm_risk = (expression / 2.0) * (1.0 - zone.gsh_concentration / 10.0)
            rm_risk = max(0, min(1.0, rm_risk))

        # GSH depletion rate (higher metabolism + reactive metabolites = more GSH consumed)
        gsh_depletion = cl_fraction * (1.0 + 2.0 * rm_risk)

        # Overall toxicity vulnerability
        vulnerability = (1.0 - zone.gsh_concentration / 10.0) * (1.0 + rm_risk) * (1.0 - o2_factor * 0.3)
        vulnerability = max(0, min(1.0, vulnerability))

        zone_results[zone_key] = {
            "name": zone.name,
            "cl_fraction": round(cl_fraction, 3),
            "cl_absolute": round(cl_fraction * cl_hepatic, 3),
            "cyp_expression": round(expression, 2),
            "oxygen_tension": zone.oxygen_tension,
            "gsh_level": zone.gsh_concentration,
            "reactive_metabolite_risk": round(rm_risk, 3),
            "gsh_depletion_rate": round(gsh_depletion, 3),
            "oxygen_sufficiency": round(o2_factor, 3),
            "toxicity_vulnerability": round(vulnerability, 3),
        }

    # Determine primary toxicity zone
    max_vulnerability = max(
        zone_results.values(), key=lambda x: x["toxicity_vulnerability"]
    )

    return {
        "zones": zone_results,
        "primary_metabolism_zone": "zone_3",  # Almost always Zone 3
        "primary_toxicity_zone": max_vulnerability["name"],
        "total_cl_hepatic": round(cl_hepatic, 3),
        "disease_state": disease_state,
        "clinical_pattern": _determine_injury_pattern(zone_results, reactive_metabolite),
        "alt_prediction": _predict_alt_pattern(zone_results, cl_hepatic, reactive_metabolite),
    }


def _get_zones_for_disease(disease_state: str) -> dict:
    """Get zone parameters modified by disease state."""
    import copy
    zones = {k: copy.deepcopy(v) for k, v in HEALTHY_ZONES.items()}

    if disease_state == "healthy":
        return zones

    if disease_state in ("cirrhosis_a", "cirrhosis_b", "cirrhosis_c"):
        severity = {"cirrhosis_a": 0.3, "cirrhosis_b": 0.5, "cirrhosis_c": 0.8}[disease_state]
        # Cirrhosis destroys Zone 3 first, then Zone 2
        zones["zone_3"].fraction_of_liver *= (1 - severity * 0.8)
        zones["zone_3"].cyp3a4_expression *= (1 - severity * 0.7)
        zones["zone_3"].cyp2e1_expression *= (1 - severity * 0.7)
        zones["zone_3"].gsh_concentration *= (1 - severity * 0.6)
        zones["zone_3"].blood_flow_fraction *= (1 - severity * 0.5)
        zones["zone_2"].fraction_of_liver *= (1 - severity * 0.4)
        zones["zone_2"].cyp3a4_expression *= (1 - severity * 0.3)

    elif disease_state == "nafld":
        # Fat accumulation primarily in Zone 3
        zones["zone_3"].cyp2e1_expression *= 1.5  # CYP2E1 induced by fat!
        zones["zone_3"].gsh_concentration *= 0.7

    elif disease_state == "nash":
        zones["zone_3"].cyp2e1_expression *= 2.0
        zones["zone_3"].gsh_concentration *= 0.4
        zones["zone_3"].fraction_of_liver *= 0.8
        zones["zone_2"].gsh_concentration *= 0.7

    elif disease_state == "alcoholic":
        # Alcohol induces CYP2E1 massively in Zone 3
        zones["zone_3"].cyp2e1_expression *= 4.0
        zones["zone_3"].cyp3a4_expression *= 0.7
        zones["zone_3"].gsh_concentration *= 0.3  # Severely depleted
        zones["zone_3"].oxygen_tension *= 0.6  # Hypoxia

    return zones


def _determine_injury_pattern(zone_results: dict, reactive_metabolite: bool) -> str:
    """Determine the expected liver injury pattern."""
    z3_vuln = zone_results["zone_3"]["toxicity_vulnerability"]
    z1_vuln = zone_results["zone_1"]["toxicity_vulnerability"]

    if reactive_metabolite and z3_vuln > 0.6:
        return "Centrilobular necrosis (Zone 3) — typical of reactive metabolite injury (paracetamol, halothane)"
    elif z3_vuln > 0.5:
        return "Zone 3 predominant injury — consistent with hypoxic/metabolic damage"
    elif z1_vuln > 0.5:
        return "Zone 1 predominant injury — unusual, seen in phosphorus poisoning and eclampsia"
    else:
        return "Low hepatotoxicity risk at current exposure levels"


def _predict_alt_pattern(zone_results: dict, cl_hepatic: float, reactive_metabolite: bool) -> dict:
    """Predict ALT (alanine aminotransferase) elevation pattern."""
    z3 = zone_results["zone_3"]

    # Baseline ALT
    alt_baseline = 25  # U/L (normal)

    # ALT rises proportional to Zone 3 damage
    if reactive_metabolite and z3["toxicity_vulnerability"] > 0.5:
        alt_peak_multiplier = 1 + z3["toxicity_vulnerability"] * 50  # Can reach 50x ULN
        time_to_peak_h = 24 + (1 - z3["toxicity_vulnerability"]) * 48
    else:
        alt_peak_multiplier = 1 + z3["toxicity_vulnerability"] * 5
        time_to_peak_h = 48

    return {
        "baseline_alt": alt_baseline,
        "predicted_peak_alt": round(alt_baseline * alt_peak_multiplier),
        "time_to_peak_h": round(time_to_peak_h),
        "pattern": (
            "Acute hepatocellular injury" if alt_peak_multiplier > 10
            else "Mild transaminase elevation" if alt_peak_multiplier > 3
            else "Within normal limits"
        ),
    }
