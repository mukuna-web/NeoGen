"""
Mechanistic CYP Inhibition Model
===================================
Models competitive, uncompetitive, and mechanism-based (time-dependent)
CYP enzyme inhibition for accurate DDI prediction.

Three types of inhibition:

1. REVERSIBLE COMPETITIVE (most common):
   - Inhibitor competes with substrate for enzyme active site
   - Effect is immediate and reversible
   - Magnitude depends on inhibitor concentration and Ki
   - fold_change = 1 + [I]/Ki (for competitive)
   - Example: fluconazole inhibiting CYP2C9

2. REVERSIBLE UNCOMPETITIVE:
   - Inhibitor binds enzyme-substrate complex
   - Less common for CYP enzymes
   - fold_change = 1 + [I]/Ki (same equation, different Ki)

3. MECHANISM-BASED (TIME-DEPENDENT) INHIBITION (TDI):
   - Inhibitor is metabolized to a reactive intermediate that
     PERMANENTLY inactivates the enzyme (covalent binding)
   - Effect is IRREVERSIBLE — enzyme is destroyed
   - Effect ACCUMULATES over time (unlike competitive)
   - Recovery requires synthesis of NEW enzyme (days)
   - Much more clinically significant than competitive inhibition
   - Parameters: kinact (max inactivation rate) and KI (concentration at 50% kinact)
   - Example: clarithromycin inhibiting CYP3A4 (mechanism-based)

Clinical importance:
   - Competitive: DDI magnitude depends on inhibitor concentration (dose-dependent)
   - Mechanism-based: DDI can persist DAYS after inhibitor is stopped
     (until new enzyme is synthesized)
   - FDA requires both types to be assessed for new drugs

Reference: FDA DDI Guidance 2020, EMA DDI Guideline 2012
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CYPInhibitorProfile:
    """Complete CYP inhibition profile for a drug."""
    name: str
    # Reversible inhibition Ki values (µM) per CYP
    ki_competitive: dict = field(default_factory=dict)
    # Mechanism-based inhibition parameters per CYP
    # Each entry: {"kinact": float (1/h), "KI": float (µM)}
    tdi_params: dict = field(default_factory=dict)
    # Clinical Cmax unbound (µM) — for DDI magnitude prediction
    cmax_u_um: float = 0.0
    # Intestinal concentration (µM) — for gut CYP3A4 inhibition
    igut_um: float = 0.0
    # Notes
    clinical_note: str = ""


# ════════════════════════════════════════════════════════════════
# INHIBITOR DATABASE
# ════════════════════════════════════════════════════════════════

INHIBITOR_DATABASE = {
    "ketoconazole": CYPInhibitorProfile(
        name="Ketoconazole",
        ki_competitive={"CYP3A4": 0.015, "CYP2C9": 12.0, "CYP1A2": 50.0},
        tdi_params={},
        cmax_u_um=0.04, igut_um=150,
        clinical_note="Potent CYP3A4 competitive inhibitor. FDA 'strong' CYP3A4 inhibitor. Used as index inhibitor in DDI studies.",
    ),
    "itraconazole": CYPInhibitorProfile(
        name="Itraconazole",
        ki_competitive={"CYP3A4": 0.003},
        tdi_params={},
        cmax_u_um=0.002, igut_um=5,
        clinical_note="Very potent CYP3A4 inhibitor. Plus active metabolite OH-itraconazole also inhibits.",
    ),
    "fluconazole": CYPInhibitorProfile(
        name="Fluconazole",
        ki_competitive={"CYP2C9": 7.0, "CYP2C19": 3.0, "CYP3A4": 10.0},
        tdi_params={},
        cmax_u_um=35.0, igut_um=200,
        clinical_note="Moderate CYP2C9/2C19 inhibitor. Dose-dependent: 100mg → moderate, 400mg → strong.",
    ),
    "ritonavir": CYPInhibitorProfile(
        name="Ritonavir",
        ki_competitive={"CYP3A4": 0.02, "CYP2D6": 2.5},
        tdi_params={"CYP3A4": {"kinact": 1.2, "KI": 0.1}},  # Also mechanism-based!
        cmax_u_um=0.15, igut_um=100,
        clinical_note="Dual CYP3A4 inhibitor: competitive + mechanism-based. Used as pharmacoenhancer in HIV (boosted PIs).",
    ),
    "clarithromycin": CYPInhibitorProfile(
        name="Clarithromycin",
        ki_competitive={"CYP3A4": 3.0},
        tdi_params={"CYP3A4": {"kinact": 1.5, "KI": 15.0}},
        cmax_u_um=3.0, igut_um=250,
        clinical_note="Strong CYP3A4 inhibitor — primarily mechanism-based. Effect persists 3-5 days after stopping.",
    ),
    "erythromycin": CYPInhibitorProfile(
        name="Erythromycin",
        ki_competitive={"CYP3A4": 20.0},
        tdi_params={"CYP3A4": {"kinact": 0.6, "KI": 25.0}},
        cmax_u_um=1.0, igut_um=300,
        clinical_note="Moderate CYP3A4 inhibitor — mechanism-based component significant.",
    ),
    "fluoxetine": CYPInhibitorProfile(
        name="Fluoxetine",
        ki_competitive={"CYP2D6": 0.07, "CYP2C19": 1.0, "CYP3A4": 5.0, "CYP2C9": 12.0},
        tdi_params={"CYP2D6": {"kinact": 0.1, "KI": 1.0}},
        cmax_u_um=0.04, igut_um=50,
        clinical_note="Strong CYP2D6 inhibitor — both competitive and mechanism-based. DDI persists weeks (long t½).",
    ),
    "paroxetine": CYPInhibitorProfile(
        name="Paroxetine",
        ki_competitive={"CYP2D6": 0.05, "CYP2B6": 3.0},
        tdi_params={"CYP2D6": {"kinact": 0.8, "KI": 0.5}},
        cmax_u_um=0.03, igut_um=40,
        clinical_note="Most potent CYP2D6 inhibitor — strong mechanism-based component.",
    ),
    "cimetidine": CYPInhibitorProfile(
        name="Cimetidine",
        ki_competitive={"CYP1A2": 200, "CYP2D6": 50, "CYP2C19": 30, "CYP3A4": 300},
        tdi_params={},
        cmax_u_um=10.0, igut_um=800,
        clinical_note="Weak pan-CYP inhibitor. Clinically relevant mainly for CYP1A2 (theophylline) and CYP2D6.",
    ),
    "grapefruit_juice": CYPInhibitorProfile(
        name="Grapefruit juice (furanocoumarins)",
        ki_competitive={},
        tdi_params={"CYP3A4": {"kinact": 3.0, "KI": 0.3}},
        cmax_u_um=0.0, igut_um=50,
        clinical_note="Selective GUT CYP3A4 mechanism-based inhibitor. Does NOT affect hepatic CYP3A4. Effect lasts 24-72h.",
    ),
    "verapamil": CYPInhibitorProfile(
        name="Verapamil",
        ki_competitive={"CYP3A4": 1.0, "CYP1A2": 20},
        tdi_params={},
        cmax_u_um=0.1, igut_um=200,
        clinical_note="Moderate CYP3A4 inhibitor. Also P-gp inhibitor.",
    ),
    "amiodarone": CYPInhibitorProfile(
        name="Amiodarone",
        ki_competitive={"CYP2D6": 2.0, "CYP2C9": 5.0, "CYP3A4": 1.0},
        tdi_params={"CYP3A4": {"kinact": 0.3, "KI": 5.0}},
        cmax_u_um=0.02, igut_um=30,
        clinical_note="Multi-CYP inhibitor. Very long t½ (58 days) — DDI persists for weeks after stopping.",
    ),
}


# ════════════════════════════════════════════════════════════════
# DDI PREDICTION
# ════════════════════════════════════════════════════════════════

def predict_ddi_magnitude(
    inhibitor_name: str,
    victim_cyp: str,
    inhibitor_conc_um: float = None,
    days_on_inhibitor: float = 1.0,
    site: str = "liver",
) -> dict:
    """
    Predict the magnitude of a CYP-mediated drug-drug interaction.

    Uses the FDA basic model:
      R = 1 + [I]u / Ki   (for competitive)
      R = kinact × [I]u / (KI + [I]u) × 1/kdeg + 1   (for mechanism-based)

    Where R = fold increase in victim drug AUC when co-administered with inhibitor.

    Args:
        inhibitor_name: Name of the inhibiting drug
        victim_cyp: CYP being inhibited (e.g., "CYP3A4")
        inhibitor_conc_um: Inhibitor concentration (µM). If None, uses Cmax,u
        days_on_inhibitor: Days of inhibitor exposure (for TDI accumulation)
        site: "liver" or "gut"

    Returns:
        DDI prediction with fold-change, severity, and clinical guidance
    """
    key = inhibitor_name.lower().replace(" ", "_")
    profile = INHIBITOR_DATABASE.get(key)

    if not profile:
        return {"inhibitor": inhibitor_name, "error": "Not in inhibitor database",
                "available": list(INHIBITOR_DATABASE.keys())}

    # Use provided concentration or default to Cmax,u
    if inhibitor_conc_um is not None:
        conc = inhibitor_conc_um
    elif site == "gut":
        conc = profile.igut_um
    else:
        conc = profile.cmax_u_um

    result = {
        "inhibitor": profile.name,
        "victim_cyp": victim_cyp,
        "inhibitor_concentration_um": round(conc, 4),
        "site": site,
        "mechanisms": [],
    }

    total_r = 1.0  # Fold increase in AUC

    # ── Competitive inhibition ──
    ki = profile.ki_competitive.get(victim_cyp)
    if ki is not None and conc > 0:
        r_competitive = 1 + conc / ki
        total_r *= r_competitive
        result["mechanisms"].append({
            "type": "Competitive (reversible)",
            "ki_um": ki,
            "r_value": round(r_competitive, 2),
            "auc_fold_increase": round(r_competitive, 2),
            "onset": "Immediate",
            "offset": "Clears with inhibitor elimination",
        })

    # ── Mechanism-based (time-dependent) inhibition ──
    tdi = profile.tdi_params.get(victim_cyp)
    if tdi is not None and conc > 0:
        kinact = tdi["kinact"]  # Max inactivation rate (1/h)
        KI = tdi["KI"]         # Concentration at 50% kinact (µM)

        # Rate of inactivation
        kobs = kinact * conc / (KI + conc)

        # CYP degradation rate constant (kdeg)
        # CYP3A4 t½ = 36h, CYP2D6 t½ = 51h, CYP1A2 t½ = 39h
        cyp_half_lives = {
            "CYP3A4": 36, "CYP2D6": 51, "CYP1A2": 39,
            "CYP2C9": 55, "CYP2C19": 26, "CYP2B6": 23,
        }
        t_half_cyp = cyp_half_lives.get(victim_cyp, 36)
        kdeg = 0.693 / t_half_cyp  # 1/h

        # Steady-state fold increase for TDI:
        # R_tdi = kdeg / (kdeg + kobs) at steady state (inverted because enzyme is depleted)
        # But R should be expressed as fold increase in AUC
        # R = 1 / (kdeg / (kdeg + kobs)) = (kdeg + kobs) / kdeg
        r_tdi_ss = (kdeg + kobs) / kdeg

        # Time-dependent — effect builds up over enzyme half-lives
        # At time t: fraction inactivated = 1 - e^(-kobs × t)
        # But we need to account for enzyme synthesis too
        # Steady-state fraction remaining = kdeg / (kdeg + kobs)
        t_hours = days_on_inhibitor * 24
        n_halflives = t_hours / t_half_cyp
        fraction_of_ss = 1 - 0.5 ** n_halflives

        r_tdi_current = 1 + (r_tdi_ss - 1) * fraction_of_ss

        total_r *= r_tdi_current

        result["mechanisms"].append({
            "type": "Mechanism-based (time-dependent, irreversible)",
            "kinact": kinact,
            "KI_um": KI,
            "kobs": round(kobs, 4),
            "r_steady_state": round(r_tdi_ss, 2),
            "r_current": round(r_tdi_current, 2),
            "days_to_steady_state": round(5 * t_half_cyp / 24, 1),
            "enzyme_half_life_h": t_half_cyp,
            "fraction_of_steady_state": round(fraction_of_ss, 2),
            "onset": f"Builds over {5 * t_half_cyp / 24:.0f} days",
            "offset": f"Persists {5 * t_half_cyp / 24:.0f} days AFTER inhibitor stopped (new enzyme needed)",
        })

    # ── Total DDI magnitude ──
    result["total_r"] = round(total_r, 2)
    result["auc_fold_increase"] = round(total_r, 2)
    result["victim_cl_reduction_pct"] = round((1 - 1/total_r) * 100, 1) if total_r > 1 else 0

    # ── Severity classification (FDA) ──
    if total_r >= 5:
        severity = "STRONG"
        guidance = "Contraindicated or major dose reduction required. AUC increases ≥5-fold."
    elif total_r >= 2:
        severity = "MODERATE"
        guidance = f"Dose adjustment likely needed. Reduce victim drug dose by ~{(1-1/total_r)*100:.0f}%."
    elif total_r >= 1.25:
        severity = "WEAK"
        guidance = "Monitor for increased effects. Dose adjustment may not be needed."
    else:
        severity = "NO SIGNIFICANT INTERACTION"
        guidance = "No dose adjustment expected."

    result["severity"] = severity
    result["guidance"] = guidance
    result["clinical_note"] = profile.clinical_note

    return result


def get_inhibitor_profile(name: str) -> CYPInhibitorProfile | None:
    """Look up an inhibitor profile."""
    key = name.lower().replace(" ", "_")
    return INHIBITOR_DATABASE.get(key)
