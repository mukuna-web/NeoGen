"""
Cardiac Action Potential Model (Simplified O'Hara-Rudy)
========================================================
Models the ventricular myocyte action potential with major ion currents.

The heart beats because cardiac cells generate electrical signals called
action potentials (APs). Drugs that alter these currents can cause
fatal arrhythmias — this is the #1 safety concern in drug development.

Phases of the cardiac AP:
  Phase 0: Rapid depolarization (Na+ inflow via Nav1.5)
  Phase 1: Early repolarization (transient K+ outflow via Ito)
  Phase 2: Plateau (Ca2+ inflow via ICaL balanced by K+ outflow via IKr)
  Phase 3: Repolarization (K+ outflow via IKr, IKs, IK1)
  Phase 4: Resting potential (IK1 maintains -85mV)

Key drug targets:
  - IKr (hERG/Kv11.1): MOST important for drug safety. Block → QT prolongation → TdP
  - INa (Nav1.5): Local anesthetics, class I antiarrhythmics
  - ICaL (Cav1.2): Calcium channel blockers (verapamil, diltiazem)
  - IKs (KCNQ1): Minor drug target, important in congenital LQTS
  - INaL (late sodium): Ranolazine target
  - Ito (Kv4.3): Rarely targeted by drugs

CiPA (FDA) approach: Multi-channel assessment, not just hERG.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IonChannelState:
    """State of a single ion channel type."""
    name: str
    gene: str
    current_name: str
    conductance: float = 100.0  # % of normal (100 = baseline)
    drug_block: float = 0.0     # % blocked by drug (0-100)
    ap_phase: str = ""          # Which AP phase this channel dominates
    direction: str = ""         # "inward" (depolarizing) or "outward" (repolarizing)


@dataclass
class CardiacAPState:
    """Complete cardiac action potential state."""
    channels: dict = field(default_factory=dict)
    ap_duration_ms: float = 300     # Action potential duration at 90% repolarization (APD90)
    qtc_ms: float = 420             # Corrected QT interval
    resting_potential_mV: float = -85
    peak_potential_mV: float = 40
    heart_rate_bpm: float = 72

    def __post_init__(self):
        if not self.channels:
            self.channels = {
                "INa": IonChannelState("Sodium (fast)", "SCN5A", "INa",
                    ap_phase="Phase 0", direction="inward"),
                "INaL": IonChannelState("Sodium (late)", "SCN5A", "INaL",
                    ap_phase="Phase 2", direction="inward"),
                "Ito": IonChannelState("Transient outward K+", "KCND3/KCNA4", "Ito",
                    ap_phase="Phase 1", direction="outward"),
                "ICaL": IonChannelState("L-type Calcium", "CACNA1C", "ICaL",
                    ap_phase="Phase 2", direction="inward"),
                "IKr": IonChannelState("Rapid delayed rectifier K+", "KCNH2 (hERG)", "IKr",
                    ap_phase="Phase 3", direction="outward"),
                "IKs": IonChannelState("Slow delayed rectifier K+", "KCNQ1", "IKs",
                    ap_phase="Phase 3", direction="outward"),
                "IK1": IonChannelState("Inward rectifier K+", "KCNJ2", "IK1",
                    ap_phase="Phase 4", direction="outward"),
            }


@dataclass
class CardiacDrugEffect:
    """Multi-channel drug effect on cardiac AP."""
    name: str
    drug_class: str
    # IC50 values (µM) for each channel — concentration causing 50% block
    ina_ic50: float = 1000       # INa block
    inal_ic50: float = 1000      # Late INa block
    ito_ic50: float = 1000       # Ito block
    ical_ic50: float = 1000      # ICaL block
    ikr_ic50: float = 1000       # IKr (hERG) block
    iks_ic50: float = 1000       # IKs block
    ik1_ic50: float = 1000       # IK1 block
    # Therapeutic concentration range
    cmax_therapeutic_um: float = 1.0  # Therapeutic Cmax in µM
    # Clinical data
    known_qt_effect: str = ""
    vaughan_williams_class: str = ""


# ════════════════════════════════════════════════════════════════
# CARDIAC DRUG DATABASE
# ════════════════════════════════════════════════════════════════

CARDIAC_DRUGS = {
    "verapamil": CardiacDrugEffect(
        name="Verapamil", drug_class="Calcium channel blocker (Class IV)",
        ina_ic50=100, ical_ic50=0.2, ikr_ic50=0.8, iks_ic50=50,
        cmax_therapeutic_um=0.4,
        known_qt_effect="Minimal QT effect despite hERG block — ICaL block is protective (multichannel effect)",
        vaughan_williams_class="IV",
    ),
    "dofetilide": CardiacDrugEffect(
        name="Dofetilide", drug_class="Pure IKr blocker (Class III)",
        ikr_ic50=0.005, iks_ic50=50, ical_ic50=100,
        cmax_therapeutic_um=0.002,
        known_qt_effect="STRONG QT prolongation — pure hERG blocker. High TdP risk.",
        vaughan_williams_class="III",
    ),
    "quinidine": CardiacDrugEffect(
        name="Quinidine", drug_class="Class IA antiarrhythmic",
        ina_ic50=5.0, inal_ic50=3.0, ikr_ic50=0.3, iks_ic50=10, ical_ic50=15,
        cmax_therapeutic_um=6.0,
        known_qt_effect="Moderate QT prolongation — INa + IKr block",
        vaughan_williams_class="IA",
    ),
    "amiodarone": CardiacDrugEffect(
        name="Amiodarone", drug_class="Multi-channel blocker (Class III)",
        ina_ic50=10, inal_ic50=5, ical_ic50=3, ikr_ic50=0.5, iks_ic50=5, ito_ic50=20,
        cmax_therapeutic_um=2.0,
        known_qt_effect="QT prolongation but low TdP risk — multi-channel blocking is protective",
        vaughan_williams_class="III (also I, II, IV properties)",
    ),
    "ranolazine": CardiacDrugEffect(
        name="Ranolazine", drug_class="Late INa blocker (antianginal)",
        inal_ic50=6.0, ikr_ic50=12.0,
        cmax_therapeutic_um=5.0,
        known_qt_effect="Mild QT prolongation — INaL block is actually anti-arrhythmic",
    ),
    "cisapride": CardiacDrugEffect(
        name="Cisapride", drug_class="5-HT4 agonist (WITHDRAWN from market)",
        ikr_ic50=0.006, ical_ic50=200,
        cmax_therapeutic_um=0.05,
        known_qt_effect="SEVERE QT prolongation — withdrawn due to fatal TdP arrhythmias",
    ),
    "haloperidol": CardiacDrugEffect(
        name="Haloperidol", drug_class="Typical antipsychotic",
        ikr_ic50=0.025, ina_ic50=5, ical_ic50=10,
        cmax_therapeutic_um=0.03,
        known_qt_effect="Moderate-high QT prolongation risk. ECG monitoring recommended.",
    ),
    "methadone": CardiacDrugEffect(
        name="Methadone", drug_class="Opioid agonist",
        ikr_ic50=2.0, ina_ic50=50,
        cmax_therapeutic_um=1.0,
        known_qt_effect="Moderate QT prolongation — dose-dependent. ECG before initiation.",
    ),
    "ondansetron": CardiacDrugEffect(
        name="Ondansetron", drug_class="5-HT3 antagonist (antiemetic)",
        ikr_ic50=1.5, ina_ic50=100,
        cmax_therapeutic_um=0.3,
        known_qt_effect="Mild QT prolongation at high doses. 32mg IV dose withdrawn.",
    ),
    "caffeine": CardiacDrugEffect(
        name="Caffeine", drug_class="Adenosine antagonist",
        ikr_ic50=200, ical_ic50=500, ina_ic50=1000,
        cmax_therapeutic_um=30,
        known_qt_effect="No significant QT effect at normal doses",
    ),
    "aspirin": CardiacDrugEffect(
        name="Aspirin", drug_class="COX inhibitor",
        ikr_ic50=10000, ical_ic50=10000,
        cmax_therapeutic_um=200,
        known_qt_effect="No QT effect",
    ),
}


# ════════════════════════════════════════════════════════════════
# AP CALCULATIONS
# ════════════════════════════════════════════════════════════════

def calculate_channel_block(ic50: float, conc: float) -> float:
    """Hill equation: % block = 100 × C^n / (IC50^n + C^n). n=1 for most channels."""
    if ic50 <= 0 or conc <= 0:
        return 0.0
    return 100.0 * conc / (ic50 + conc)


def simulate_cardiac_ap(
    drug_name: str,
    plasma_conc_um: float = 1.0,
    baseline_qtc: float = 420,
    heart_rate: float = 72,
) -> dict:
    """
    Simulate drug effects on cardiac action potential.

    Uses the CiPA multi-channel approach:
    1. Calculate % block for each channel at given concentration
    2. Estimate APD change from channel blocks
    3. Calculate QTc change
    4. Assess TdP risk using multi-channel score

    Returns comprehensive cardiac safety assessment.
    """
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    drug = CARDIAC_DRUGS.get(key)

    if not drug:
        return {
            "drug": drug_name,
            "has_cardiac_effect": False,
            "qtc_change_ms": 0,
            "tdp_risk": "Not assessed — no cardiac profile available",
        }

    state = CardiacAPState(heart_rate_bpm=heart_rate, qtc_ms=baseline_qtc)

    # Calculate block for each channel
    blocks = {
        "INa": calculate_channel_block(drug.ina_ic50, plasma_conc_um),
        "INaL": calculate_channel_block(drug.inal_ic50, plasma_conc_um),
        "Ito": calculate_channel_block(drug.ito_ic50, plasma_conc_um),
        "ICaL": calculate_channel_block(drug.ical_ic50, plasma_conc_um),
        "IKr": calculate_channel_block(drug.ikr_ic50, plasma_conc_um),
        "IKs": calculate_channel_block(drug.iks_ic50, plasma_conc_um),
        "IK1": calculate_channel_block(drug.ik1_ic50, plasma_conc_um),
    }

    # Apply blocks to state
    for ch_name, block_pct in blocks.items():
        if ch_name in state.channels:
            state.channels[ch_name].drug_block = block_pct
            state.channels[ch_name].conductance = 100 - block_pct

    # Estimate APD change
    # Repolarizing block (IKr, IKs, IK1) → prolongation
    # Depolarizing block (INa, ICaL) → shortening
    repol_block = blocks["IKr"] * 1.5 + blocks["IKs"] * 0.5 + blocks["IK1"] * 0.3
    depol_block = blocks["ICaL"] * 1.0 + blocks["INaL"] * 0.3

    # Net APD change (ms) — rough estimate
    apd_change = (repol_block - depol_block) * 1.5  # ~1.5 ms per % net block
    apd_change = max(-100, min(200, apd_change))

    new_apd = 300 + apd_change
    new_qtc = baseline_qtc + apd_change

    state.ap_duration_ms = new_apd
    state.qtc_ms = new_qtc

    # Safety margin (Cmax_therapeutic / IC50_hERG)
    safety_margin = drug.ikr_ic50 / drug.cmax_therapeutic_um if drug.cmax_therapeutic_um > 0 else 999
    safety_margin_at_conc = drug.ikr_ic50 / plasma_conc_um if plasma_conc_um > 0 else 999

    # CiPA-inspired TdP risk score
    # High IKr block without ICaL compensation = dangerous
    # High IKr block WITH ICaL block = safer (multi-channel effect)
    net_repol_reserve = 100 - blocks["IKr"] - blocks["IKs"] * 0.3 + blocks["ICaL"] * 0.5
    cipa_score = blocks["IKr"] - blocks["ICaL"] * 0.7 - blocks["INaL"] * 0.2

    if cipa_score > 50:
        tdp_risk = "HIGH — Significant TdP risk. ECG monitoring mandatory."
        tdp_category = 3
    elif cipa_score > 20:
        tdp_risk = "INTERMEDIATE — Possible TdP risk. Consider ECG monitoring."
        tdp_category = 2
    elif cipa_score > 5:
        tdp_risk = "LOW — Minimal TdP risk at this concentration."
        tdp_category = 1
    else:
        tdp_risk = "NEGLIGIBLE — No significant proarrhythmic risk."
        tdp_category = 0

    # Channel block details
    channel_details = []
    for ch_name, ch in state.channels.items():
        if ch.drug_block > 1:
            channel_details.append({
                "channel": ch_name,
                "gene": ch.gene,
                "current": ch.current_name,
                "block_percent": round(ch.drug_block, 1),
                "conductance_remaining": round(ch.conductance, 1),
                "ap_phase": ch.ap_phase,
                "direction": ch.direction,
                "clinical_impact": (
                    "SIGNIFICANT" if ch.drug_block > 30
                    else "MODERATE" if ch.drug_block > 10
                    else "MILD"
                ),
            })

    channel_details.sort(key=lambda x: x["block_percent"], reverse=True)

    return {
        "drug": drug.name,
        "drug_class": drug.drug_class,
        "has_cardiac_effect": True,
        "plasma_conc_um": plasma_conc_um,
        "channel_blocks": channel_details,
        "ap_parameters": {
            "apd90_ms": round(new_apd, 1),
            "apd90_change_ms": round(apd_change, 1),
            "qtc_ms": round(new_qtc, 1),
            "qtc_change_ms": round(apd_change, 1),
            "resting_potential_mV": state.resting_potential_mV,
            "peak_potential_mV": state.peak_potential_mV,
        },
        "safety": {
            "herg_safety_margin": round(safety_margin, 1),
            "safety_margin_at_conc": round(safety_margin_at_conc, 1),
            "cipa_score": round(cipa_score, 1),
            "tdp_risk_category": tdp_category,
            "tdp_risk": tdp_risk,
            "net_repolarization_reserve": round(net_repol_reserve, 1),
        },
        "clinical": {
            "known_qt_effect": drug.known_qt_effect,
            "vaughan_williams_class": drug.vaughan_williams_class,
            "ecg_recommendation": (
                "Baseline ECG + continuous telemetry" if tdp_category >= 3
                else "Baseline ECG + repeat after dose change" if tdp_category >= 2
                else "Baseline ECG recommended" if tdp_category >= 1
                else "No routine ECG monitoring needed"
            ),
        },
    }
