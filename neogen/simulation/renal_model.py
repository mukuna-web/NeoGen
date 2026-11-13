"""
Full Renal Tubular Model
=========================
Models the 3 mechanisms kidneys use to handle drugs:

1. Glomerular Filtration (passive)
   - Only FREE drug filtered (bound drug stays in blood)
   - GFR ≈ 120 mL/min (7.2 L/h) in healthy adults
   - Size-limited: molecules >60 kDa don't pass

2. Tubular Secretion (active transport)
   - OAT1/OAT3 pump organic anions into urine (penicillins, methotrexate, NSAIDs)
   - OCT2/MATE1 pump organic cations into urine (metformin, cisplatin, cimetidine)
   - Michaelis-Menten kinetics: saturatable at high concentrations

3. Tubular Reabsorption (passive, pH-dependent)
   - Drug in urine can passively diffuse back into blood
   - Only UNIONIZED drug is reabsorbed (Henderson-Hasselbalch)
   - Urine pH 4.5-8.0: acidic urine traps basic drugs, alkaline urine traps acidic drugs
   - This is why sodium bicarb treats aspirin overdose (alkalinizes urine)

Net renal clearance = Filtration + Secretion - Reabsorption
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RenalDrugProfile:
    """Drug-specific renal handling parameters."""
    name: str
    fu: float = 1.0              # Fraction unbound in plasma
    mw: float = 300              # Molecular weight (Da)
    pka: float = 7.0             # Acid dissociation constant
    acid_base: str = "neutral"   # "acid", "base", "neutral", "zwitterion"
    logP: float = 1.0            # Lipophilicity (affects reabsorption)

    # Tubular secretion parameters
    oat_substrate: bool = False  # OAT1/OAT3 substrate (anion transporters)
    oct_substrate: bool = False  # OCT2/MATE1 substrate (cation transporters)
    clsec_max: float = 0.0      # Max secretion clearance (L/h) — Vmax/Km
    km_sec: float = 100.0       # Michaelis constant for secretion (mg/L)

    # Reabsorption
    reabsorption_fraction: float = 0.5  # Fraction reabsorbed at reference pH


# ════════════════════════════════════════════════════════════════
# RENAL DRUG DATABASE
# ════════════════════════════════════════════════════════════════

RENAL_PROFILES = {
    # OAT substrates (organic anions)
    "penicillin_g": RenalDrugProfile(
        name="Penicillin G", fu=0.40, mw=334, pka=2.7, acid_base="acid",
        logP=1.83, oat_substrate=True, clsec_max=18.0, km_sec=50,
        reabsorption_fraction=0.1
    ),
    "amoxicillin": RenalDrugProfile(
        name="Amoxicillin", fu=0.82, mw=365, pka=2.4, acid_base="acid",
        logP=0.87, oat_substrate=True, clsec_max=8.0, km_sec=80,
        reabsorption_fraction=0.15
    ),
    "methotrexate": RenalDrugProfile(
        name="Methotrexate", fu=0.50, mw=454, pka=4.7, acid_base="acid",
        logP=-1.85, oat_substrate=True, clsec_max=12.0, km_sec=30,
        reabsorption_fraction=0.05
    ),
    "furosemide": RenalDrugProfile(
        name="Furosemide", fu=0.01, mw=330, pka=3.8, acid_base="acid",
        logP=2.03, oat_substrate=True, clsec_max=20.0, km_sec=40,
        reabsorption_fraction=0.3
    ),
    "aspirin": RenalDrugProfile(
        name="Aspirin", fu=0.50, mw=180, pka=3.5, acid_base="acid",
        logP=1.19, oat_substrate=False, clsec_max=2.0, km_sec=100,
        reabsorption_fraction=0.6
    ),

    # OCT substrates (organic cations)
    "metformin": RenalDrugProfile(
        name="Metformin", fu=1.0, mw=129, pka=11.5, acid_base="base",
        logP=-1.43, oct_substrate=True, clsec_max=25.0, km_sec=200,
        reabsorption_fraction=0.0  # Too hydrophilic to reabsorb
    ),
    "cimetidine": RenalDrugProfile(
        name="Cimetidine", fu=0.80, mw=252, pka=6.8, acid_base="base",
        logP=0.40, oct_substrate=True, clsec_max=15.0, km_sec=100,
        reabsorption_fraction=0.2
    ),
    "cisplatin": RenalDrugProfile(
        name="Cisplatin", fu=0.10, mw=300, pka=5.0, acid_base="neutral",
        logP=-2.19, oct_substrate=True, clsec_max=5.0, km_sec=20,
        reabsorption_fraction=0.05
    ),
    "atenolol": RenalDrugProfile(
        name="Atenolol", fu=0.95, mw=266, pka=9.6, acid_base="base",
        logP=0.16, oct_substrate=True, clsec_max=4.0, km_sec=150,
        reabsorption_fraction=0.1
    ),

    # Primarily filtered (minimal secretion)
    "caffeine": RenalDrugProfile(
        name="Caffeine", fu=0.65, mw=194, pka=14.0, acid_base="neutral",
        logP=-0.07, oat_substrate=False, oct_substrate=False,
        clsec_max=0.5, km_sec=200, reabsorption_fraction=0.85
    ),
    "paracetamol": RenalDrugProfile(
        name="Paracetamol", fu=0.80, mw=151, pka=9.5, acid_base="neutral",
        logP=0.46, oat_substrate=False, oct_substrate=False,
        clsec_max=0.0, km_sec=200, reabsorption_fraction=0.7
    ),
    "ibuprofen": RenalDrugProfile(
        name="Ibuprofen", fu=0.01, mw=206, pka=4.9, acid_base="acid",
        logP=3.97, oat_substrate=False, oct_substrate=False,
        clsec_max=1.0, km_sec=100, reabsorption_fraction=0.9
    ),
    "warfarin": RenalDrugProfile(
        name="Warfarin", fu=0.01, mw=308, pka=5.1, acid_base="acid",
        logP=2.70, oat_substrate=False, oct_substrate=False,
        clsec_max=0.0, km_sec=100, reabsorption_fraction=0.95
    ),
    "lithium": RenalDrugProfile(
        name="Lithium", fu=1.0, mw=7, pka=14.0, acid_base="neutral",
        logP=-3.0, oat_substrate=False, oct_substrate=False,
        clsec_max=0.0, km_sec=100, reabsorption_fraction=0.8
    ),
    "digoxin": RenalDrugProfile(
        name="Digoxin", fu=0.75, mw=781, pka=7.2, acid_base="neutral",
        logP=1.26, oat_substrate=False, oct_substrate=False,
        clsec_max=3.0, km_sec=50, reabsorption_fraction=0.3
    ),
}


# ════════════════════════════════════════════════════════════════
# RENAL CLEARANCE CALCULATIONS
# ════════════════════════════════════════════════════════════════

def calculate_fraction_unionized(pka: float, ph: float, acid_base: str) -> float:
    """
    Henderson-Hasselbalch equation — calculates fraction of drug in
    unionized (non-charged) form. Only unionized drug can passively
    cross tubular membranes for reabsorption.

    For acids: fu = 1 / (1 + 10^(pH - pKa))
    For bases: fu = 1 / (1 + 10^(pKa - pH))
    """
    if acid_base == "neutral":
        return 0.5  # Neutral drugs — moderate reabsorption
    elif acid_base == "acid":
        return 1.0 / (1.0 + 10 ** (ph - pka))
    elif acid_base == "base":
        return 1.0 / (1.0 + 10 ** (pka - ph))
    else:  # zwitterion
        return 0.3


def calculate_gfr_clearance(
    gfr: float,  # mL/min
    fu: float,   # fraction unbound
    mw: float,   # molecular weight
) -> float:
    """
    Glomerular filtration clearance.
    CL_filt = GFR × fu × sieving_coefficient

    Only free drug is filtered. Large molecules (>60 kDa) are blocked.
    """
    # Convert GFR to L/h
    gfr_lh = gfr * 60 / 1000  # mL/min → L/h

    # Sieving coefficient — drops off for MW > 10,000
    if mw < 10000:
        sieving = 1.0
    elif mw < 60000:
        sieving = max(0, 1.0 - (mw - 10000) / 50000)
    else:
        sieving = 0.0

    return gfr_lh * fu * sieving


def calculate_secretion_clearance(
    profile: RenalDrugProfile,
    plasma_conc: float = 1.0,  # mg/L
    oat_inhibition: float = 1.0,   # 1.0 = normal, 0 = fully inhibited
    oct_inhibition: float = 1.0,
) -> float:
    """
    Active tubular secretion — Michaelis-Menten kinetics.
    CL_sec = (Vmax / (Km + C)) × inhibition_factor

    Saturatable: at high concentrations, secretion plateaus.
    """
    if profile.clsec_max <= 0:
        return 0.0

    # Michaelis-Menten
    cl_sec = profile.clsec_max * profile.km_sec / (profile.km_sec + plasma_conc)

    # Apply transporter inhibition
    if profile.oat_substrate:
        cl_sec *= oat_inhibition
    if profile.oct_substrate:
        cl_sec *= oct_inhibition

    return cl_sec


def calculate_reabsorption_clearance(
    profile: RenalDrugProfile,
    urine_ph: float = 6.0,
    urine_flow: float = 1.0,  # mL/min (normal 0.5-2.0)
) -> float:
    """
    Passive tubular reabsorption — pH-dependent.

    Only UNIONIZED drug is lipophilic enough to diffuse back into blood.
    High urine flow = less time for reabsorption = less drug reabsorbed.
    Acidic urine traps basic drugs (keeps them ionized = not reabsorbed).
    Alkaline urine traps acidic drugs.
    """
    # Fraction unionized at urine pH
    f_unionized = calculate_fraction_unionized(
        profile.pka, urine_ph, profile.acid_base
    )

    # Base reabsorption fraction modulated by:
    # 1. Ionization state (more unionized = more reabsorbed)
    # 2. Lipophilicity (logP > 1 = highly reabsorbed)
    # 3. Urine flow (high flow = less time = less reabsorption)
    logp_factor = min(1.0, max(0.0, (profile.logP + 1) / 4))  # Normalize to 0-1
    flow_factor = 1.0 / urine_flow  # Inverse relationship

    reabsorption = profile.reabsorption_fraction * f_unionized * logp_factor * min(flow_factor, 2.0)

    return min(reabsorption, 0.99)  # Cap at 99%


def calculate_net_renal_clearance(
    profile: RenalDrugProfile,
    gfr: float = 120.0,          # mL/min
    urine_ph: float = 6.0,
    urine_flow: float = 1.0,     # mL/min
    plasma_conc: float = 1.0,    # mg/L
    oat_inhibition: float = 1.0,
    oct_inhibition: float = 1.0,
) -> dict:
    """
    Calculate net renal clearance = filtration + secretion - reabsorption.

    Returns breakdown of all three components.
    """
    cl_filt = calculate_gfr_clearance(gfr, profile.fu, profile.mw)
    cl_sec = calculate_secretion_clearance(profile, plasma_conc, oat_inhibition, oct_inhibition)
    reabsorption_frac = calculate_reabsorption_clearance(profile, urine_ph, urine_flow)

    # Net CL = (filtration + secretion) × (1 - reabsorption_fraction)
    cl_gross = cl_filt + cl_sec
    cl_reabsorbed = cl_gross * reabsorption_frac
    cl_net = cl_gross - cl_reabsorbed

    return {
        "cl_filtration": round(cl_filt, 3),
        "cl_secretion": round(cl_sec, 3),
        "cl_reabsorption": round(cl_reabsorbed, 3),
        "cl_net_renal": round(max(cl_net, 0), 3),
        "reabsorption_fraction": round(reabsorption_frac, 3),
        "fraction_filtered": round(cl_filt / cl_gross if cl_gross > 0 else 0, 3),
        "fraction_secreted": round(cl_sec / cl_gross if cl_gross > 0 else 0, 3),
        "gfr_used": gfr,
        "urine_ph": urine_ph,
        "transport_mechanism": (
            "OAT1/OAT3 (anion)" if profile.oat_substrate
            else "OCT2/MATE1 (cation)" if profile.oct_substrate
            else "Passive filtration only"
        ),
        "clinical_notes": _generate_renal_notes(profile, gfr, urine_ph, reabsorption_frac),
    }


def _generate_renal_notes(
    profile: RenalDrugProfile,
    gfr: float,
    urine_ph: float,
    reabsorption_frac: float,
) -> list[str]:
    """Generate clinical notes about renal handling."""
    notes = []

    if gfr < 60:
        notes.append(f"Reduced GFR ({gfr} mL/min) — dose adjustment may be needed")
    if gfr < 30:
        notes.append("Severe renal impairment — consider alternative drug or major dose reduction")

    if profile.oat_substrate:
        notes.append("OAT substrate — probenecid will block secretion and INCREASE drug levels")
    if profile.oct_substrate:
        notes.append("OCT2 substrate — cimetidine will block secretion and INCREASE drug levels")

    if reabsorption_frac > 0.8:
        notes.append("Highly reabsorbed — urine pH manipulation may significantly affect elimination")
        if profile.acid_base == "acid":
            notes.append("Weak acid — alkalinizing urine (sodium bicarbonate) will INCREASE elimination")
        elif profile.acid_base == "base":
            notes.append("Weak base — acidifying urine (ammonium chloride) will INCREASE elimination")

    return notes


def get_renal_profile(drug_name: str) -> RenalDrugProfile | None:
    """Look up a drug's renal profile. Returns None if not found."""
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    return RENAL_PROFILES.get(key)
