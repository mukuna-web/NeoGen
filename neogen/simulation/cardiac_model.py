"""
Multi-Channel Cardiac Ion Channel Model (CiPA-like)
=====================================================
Goes beyond hERG-only assessment to model multiple cardiac ion channels
that together determine the cardiac action potential shape.

The CiPA (Comprehensive in vitro Proarrhythmia Assay) initiative uses:
1. IKr (hERG) — rapid delayed rectifier K+ current (repolarization)
2. IKs — slow delayed rectifier K+ current (repolarization reserve)
3. INa (Nav1.5) — fast sodium current (depolarization)
4. INaL — late sodium current (plateau phase)
5. ICaL (Cav1.2) — L-type calcium current (plateau, contraction)
6. Ito — transient outward K+ current (early repolarization)
7. IK1 — inward rectifier K+ current (resting potential)

Drug effects on these channels are combined to predict:
- QT prolongation (dangerous)
- Action potential duration (APD90)
- Triangulation (proarrhythmic marker)
- Torsade de Pointes (TdP) risk score
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChannelBlock:
    """Drug block of a single cardiac ion channel."""
    channel: str            # IKr, IKs, INa, INaL, ICaL, Ito, IK1
    ic50_uM: float = 100.0  # Concentration for 50% block
    hill_n: float = 1.0     # Hill coefficient
    block_type: str = "state_independent"  # "state_independent", "use_dependent"


@dataclass
class CardiacChannelProfile:
    """Multi-channel cardiac profile for a drug."""
    drug_name: str
    channels: list = field(default_factory=list)
    clinical_qt_ms: float = 0.0      # Observed QT prolongation at therapeutic dose
    tdp_risk_category: str = "low"   # "low", "intermediate", "high"
    tdp_cases_reported: bool = False


# Multi-channel IC50 data (µM) from literature/CiPA
CARDIAC_CHANNEL_DATABASE = {
    "dofetilide": CardiacChannelProfile("dofetilide", [
        ChannelBlock("IKr", 0.005, 1.0),   # Extremely potent hERG blocker
        ChannelBlock("INaL", 100.0, 1.0),   # No INaL block
        ChannelBlock("ICaL", 100.0, 1.0),   # No ICaL block
    ], clinical_qt_ms=30, tdp_risk_category="high", tdp_cases_reported=True),

    "sotalol": CardiacChannelProfile("sotalol", [
        ChannelBlock("IKr", 100.0, 1.0),
        ChannelBlock("IKs", 200.0, 1.0),
    ], clinical_qt_ms=20, tdp_risk_category="high", tdp_cases_reported=True),

    "quinidine": CardiacChannelProfile("quinidine", [
        ChannelBlock("IKr", 0.3, 1.0),
        ChannelBlock("INa", 5.0, 1.0),
        ChannelBlock("ICaL", 15.0, 1.0),
        ChannelBlock("INaL", 3.0, 1.0),
    ], clinical_qt_ms=15, tdp_risk_category="high", tdp_cases_reported=True),

    "verapamil": CardiacChannelProfile("verapamil", [
        ChannelBlock("IKr", 0.1, 1.0),    # Blocks hERG
        ChannelBlock("ICaL", 0.2, 1.0),   # BUT also blocks ICaL → protective!
    ], clinical_qt_ms=0, tdp_risk_category="low", tdp_cases_reported=False),
    # Verapamil is the classic "false positive" — blocks hERG but ICaL block protects

    "ranolazine": CardiacChannelProfile("ranolazine", [
        ChannelBlock("IKr", 10.0, 1.0),
        ChannelBlock("INaL", 6.0, 1.0),   # INaL block is therapeutic
    ], clinical_qt_ms=5, tdp_risk_category="low", tdp_cases_reported=False),

    "ondansetron": CardiacChannelProfile("ondansetron", [
        ChannelBlock("IKr", 1.5, 1.0),
        ChannelBlock("INa", 30.0, 1.0),
    ], clinical_qt_ms=5, tdp_risk_category="intermediate"),

    "haloperidol": CardiacChannelProfile("haloperidol", [
        ChannelBlock("IKr", 0.007, 1.0),  # Very potent hERG blocker
        ChannelBlock("ICaL", 3.0, 1.0),   # Some ICaL block
    ], clinical_qt_ms=10, tdp_risk_category="high", tdp_cases_reported=True),

    "amiodarone": CardiacChannelProfile("amiodarone", [
        ChannelBlock("IKr", 0.2, 1.0),
        ChannelBlock("IKs", 5.0, 1.0),
        ChannelBlock("INa", 3.0, 1.0),
        ChannelBlock("INaL", 2.0, 1.0),
        ChannelBlock("ICaL", 1.0, 1.0),
    ], clinical_qt_ms=20, tdp_risk_category="intermediate"),
    # Amiodarone blocks everything → paradoxically less TdP than expected

    "moxifloxacin": CardiacChannelProfile("moxifloxacin", [
        ChannelBlock("IKr", 30.0, 1.0),
    ], clinical_qt_ms=8, tdp_risk_category="intermediate"),
    # CiPA positive control

    "aspirin": CardiacChannelProfile("aspirin", [
    ], clinical_qt_ms=0, tdp_risk_category="low"),

    "caffeine": CardiacChannelProfile("caffeine", [
        ChannelBlock("IKr", 500.0, 1.0),  # Very weak hERG
    ], clinical_qt_ms=0, tdp_risk_category="low"),

    "ibuprofen": CardiacChannelProfile("ibuprofen", [
        ChannelBlock("IKr", 200.0, 1.0),
    ], clinical_qt_ms=0, tdp_risk_category="low"),
}


def calculate_channel_block(block: ChannelBlock, conc_uM: float) -> float:
    """Calculate fractional block using Hill equation."""
    if conc_uM <= 0 or block.ic50_uM <= 0:
        return 0.0
    return (conc_uM ** block.hill_n) / (block.ic50_uM ** block.hill_n + conc_uM ** block.hill_n)


def predict_qt_effect(
    channel_blocks: dict,  # {channel: fractional_block}
) -> dict:
    """
    Predict net effect on cardiac action potential from multi-channel block.

    Uses simplified CiPA-like scoring:
    - IKr block → prolongs APD (bad)
    - INaL block → shortens APD (can offset IKr)
    - ICaL block → shortens APD (protective, like verapamil)
    - INa block → slows conduction (can be pro or anti-arrhythmic)
    """
    # Channel contributions to APD change (ms per % block)
    apd_sensitivity = {
        "IKr": +3.0,      # IKr block prolongs APD strongly
        "IKs": +1.5,      # IKs block prolongs (repolarization reserve)
        "INa": -0.5,      # INa block shortens slightly
        "INaL": -2.0,     # INaL block shortens APD (therapeutic for ranolazine)
        "ICaL": -2.5,     # ICaL block shortens APD (protective)
        "Ito": +0.5,      # Ito block has minor effect
        "IK1": +1.0,      # IK1 block prolongs
    }

    # Calculate net APD change
    delta_apd = 0.0
    for channel, frac_block in channel_blocks.items():
        sensitivity = apd_sensitivity.get(channel, 0)
        delta_apd += sensitivity * frac_block * 100  # Convert fraction to %

    # QT prolongation (roughly 1:1 with APD at therapeutic concentrations)
    delta_qt = delta_apd

    # Triangulation index (proarrhythmic marker)
    # High IKr block without ICaL compensation → triangulation → TdP risk
    ikr_block = channel_blocks.get("IKr", 0)
    ical_block = channel_blocks.get("ICaL", 0)
    inal_block = channel_blocks.get("INaL", 0)
    triangulation = max(0, ikr_block - ical_block - inal_block * 0.5)

    # TdP risk score (simplified CiPA)
    # Pure IKr blockers → high risk
    # Multi-channel blockers → variable (verapamil pattern = low risk)
    tdp_score = ikr_block * 3 - ical_block * 2 - inal_block * 1.5
    tdp_score = max(0, tdp_score)

    if tdp_score > 0.5:
        risk = "high"
    elif tdp_score > 0.2:
        risk = "intermediate"
    else:
        risk = "low"

    return {
        "delta_apd_ms": round(delta_apd, 1),
        "delta_qt_ms": round(delta_qt, 1),
        "triangulation_index": round(triangulation, 3),
        "tdp_risk_score": round(tdp_score, 3),
        "tdp_risk_category": risk,
        "channel_blocks": {k: round(v, 3) for k, v in channel_blocks.items()},
    }


def assess_cardiac_safety(
    smiles: str = "",
    drug_name: str = "",
    plasma_cmax_uM: float = 1.0,
) -> dict:
    """
    Full multi-channel cardiac safety assessment.

    Returns comprehensive cardiac risk profile including:
    - Per-channel block at therapeutic concentration
    - Net APD/QT effect
    - TdP risk score
    - Safety margins per channel
    """
    name_lower = drug_name.lower().strip()

    profile = None
    for known_name, p in CARDIAC_CHANNEL_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            profile = p
            break

    if not profile:
        return {
            "drug": drug_name or "Unknown",
            "assessed": False,
            "reason": "No multi-channel data available",
            "tdp_risk_category": "unknown",
        }

    # Calculate block at therapeutic Cmax
    channel_blocks = {}
    safety_margins = {}
    for block in profile.channels:
        frac = calculate_channel_block(block, plasma_cmax_uM)
        channel_blocks[block.channel] = frac
        safety_margins[block.channel] = {
            "ic50_uM": block.ic50_uM,
            "cmax_uM": round(plasma_cmax_uM, 3),
            "safety_margin": round(block.ic50_uM / max(plasma_cmax_uM, 0.001), 1),
            "fractional_block": round(frac, 3),
            "block_pct": round(frac * 100, 1),
        }

    # Predict net cardiac effect
    qt_effect = predict_qt_effect(channel_blocks)

    # Multi-channel compensation analysis
    ikr = channel_blocks.get("IKr", 0)
    ical = channel_blocks.get("ICaL", 0)
    if ikr > 0.1 and ical > 0.1:
        compensation = "ICaL block partially compensates IKr block (verapamil-like pattern)"
    elif ikr > 0.1:
        compensation = "IKr block without compensation — higher proarrhythmic risk"
    else:
        compensation = "Minimal cardiac ion channel interaction"

    return {
        "drug": profile.drug_name,
        "assessed": True,
        "channel_safety_margins": safety_margins,
        "qt_effect": qt_effect,
        "clinical_qt_prolongation_ms": profile.clinical_qt_ms,
        "tdp_risk_category": qt_effect["tdp_risk_category"],
        "tdp_cases_reported": profile.tdp_cases_reported,
        "multi_channel_compensation": compensation,
        "n_channels_blocked": len([b for b in channel_blocks.values() if b > 0.01]),
        "recommendation": _get_cardiac_recommendation(qt_effect, profile),
    }


def _get_cardiac_recommendation(qt_effect: dict, profile: CardiacChannelProfile) -> str:
    risk = qt_effect["tdp_risk_category"]
    if risk == "high":
        return "ECG monitoring recommended. Avoid in patients with QT prolongation, hypokalemia, or concomitant QT-prolonging drugs."
    elif risk == "intermediate":
        return "Baseline ECG advised. Monitor electrolytes. Caution with other QT-prolonging drugs."
    else:
        return "No significant cardiac risk at therapeutic concentrations."
