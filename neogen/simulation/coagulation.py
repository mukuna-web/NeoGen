"""
Blood Coagulation Cascade Model
=================================
Models the clotting system and how drugs affect bleeding/clotting risk.

The cascade has two initiation pathways:
  Intrinsic (contact activation): XII → XIa → IX → IXa → X → Xa → II → IIa → Fibrin
  Extrinsic (tissue factor):     TF + VII → VIIa → X → Xa → II → IIa → Fibrin
  Common pathway:                X → Xa → II (prothrombin) → IIa (thrombin) → Fibrinogen → Fibrin

Key factors:
  - Factor II (Prothrombin) → Factor IIa (Thrombin) — the central enzyme
  - Factor VII — extrinsic pathway, shortest half-life (~6h) — first to fall with warfarin
  - Factor IX — intrinsic pathway (hemophilia B if deficient)
  - Factor X — common pathway convergence point
  - Factor XIII — cross-links fibrin (stabilizes clot)

Drug targets:
  - Warfarin: blocks vitamin K recycling → reduces II, VII, IX, X synthesis
  - Heparin: potentiates antithrombin III → inactivates IIa and Xa
  - DOACs: direct inhibitors (rivaroxaban → Xa, dabigatran → IIa)
  - Aspirin: blocks TXA2 → inhibits platelet aggregation (separate from cascade)
  - Clopidogrel: blocks ADP receptor (P2Y12) on platelets

Clinical tests:
  - PT/INR: measures extrinsic + common pathway (warfarin monitoring)
  - aPTT: measures intrinsic + common pathway (heparin monitoring)
  - Platelet count + bleeding time: platelet function
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CoagulationFactor:
    """Represents a single clotting factor."""
    name: str
    number: str              # Roman numeral (e.g., "II", "VII")
    half_life_h: float       # Plasma half-life
    activity: float = 100.0  # % of normal (100 = fully active)
    synthesis_rate: float = 100.0  # % of normal synthesis
    vitamin_k_dependent: bool = False


@dataclass
class PlateletState:
    """Represents platelet function."""
    count: float = 250000      # per µL (normal 150,000-400,000)
    aggregation: float = 100.0  # % of normal aggregation
    txa2_activity: float = 100.0  # Thromboxane A2 (aspirin target)
    adp_receptor: float = 100.0   # P2Y12 receptor (clopidogrel target)
    gpiib_iiia: float = 100.0     # GP IIb/IIIa (abciximab target)


@dataclass
class CoagulationState:
    """Complete coagulation system state."""
    factors: dict = field(default_factory=dict)
    platelets: PlateletState = field(default_factory=PlateletState)
    antithrombin_activity: float = 100.0  # % (heparin enhances this)
    fibrinolysis_rate: float = 100.0      # % (tPA enhances this)

    def __post_init__(self):
        if not self.factors:
            self.factors = {
                "I": CoagulationFactor("Fibrinogen", "I", half_life_h=96),
                "II": CoagulationFactor("Prothrombin", "II", half_life_h=60, vitamin_k_dependent=True),
                "V": CoagulationFactor("Proaccelerin", "V", half_life_h=36),
                "VII": CoagulationFactor("Proconvertin", "VII", half_life_h=6, vitamin_k_dependent=True),
                "VIII": CoagulationFactor("Antihemophilic A", "VIII", half_life_h=12),
                "IX": CoagulationFactor("Christmas Factor", "IX", half_life_h=24, vitamin_k_dependent=True),
                "X": CoagulationFactor("Stuart-Prower", "X", half_life_h=40, vitamin_k_dependent=True),
                "XI": CoagulationFactor("Thromboplastin antecedent", "XI", half_life_h=52),
                "XII": CoagulationFactor("Hageman Factor", "XII", half_life_h=52),
                "XIII": CoagulationFactor("Fibrin-stabilizing", "XIII", half_life_h=120),
            }


# ════════════════════════════════════════════════════════════════
# DRUG EFFECTS ON COAGULATION
# ════════════════════════════════════════════════════════════════

@dataclass
class CoagDrugEffect:
    """How a drug affects coagulation."""
    name: str
    drug_class: str
    mechanism: str

    # Factor effects (% reduction in activity)
    factor_ii_reduction: float = 0      # Prothrombin
    factor_vii_reduction: float = 0     # Proconvertin (shortest t½)
    factor_ix_reduction: float = 0      # Christmas
    factor_x_reduction: float = 0       # Stuart-Prower
    direct_xa_inhibition: float = 0     # Direct Factor Xa block (rivaroxaban)
    direct_iia_inhibition: float = 0    # Direct thrombin block (dabigatran)

    # Platelet effects
    txa2_inhibition: float = 0          # Aspirin
    adp_receptor_block: float = 0       # Clopidogrel
    gpiib_iiia_block: float = 0         # Abciximab

    # Antithrombin enhancement
    at_enhancement: float = 0           # Heparin (UFH)
    at_xa_selective: float = 0          # LMWH/Fondaparinux (Xa-selective)

    # Fibrinolysis
    fibrinolysis_enhancement: float = 0  # tPA, streptokinase

    # Time course
    onset_h: float = 1.0
    peak_effect_h: float = 4.0
    duration_h: float = 24.0
    time_to_full_effect_days: float = 0  # For warfarin (takes days)

    # Monitoring
    affects_pt_inr: bool = False
    affects_aptt: bool = False
    target_inr: tuple = (0, 0)          # Target INR range if applicable


COAG_DRUG_EFFECTS = {
    "warfarin": CoagDrugEffect(
        name="Warfarin", drug_class="Vitamin K antagonist",
        mechanism="Blocks VKORC1 → prevents vitamin K recycling → reduces synthesis of factors II, VII, IX, X",
        factor_ii_reduction=70, factor_vii_reduction=80,
        factor_ix_reduction=70, factor_x_reduction=75,
        onset_h=12, peak_effect_h=72, duration_h=120,
        time_to_full_effect_days=5,
        affects_pt_inr=True, target_inr=(2.0, 3.0),
    ),

    "aspirin": CoagDrugEffect(
        name="Aspirin", drug_class="COX-1 inhibitor / Antiplatelet",
        mechanism="Irreversibly acetylates COX-1 → blocks TXA2 synthesis → inhibits platelet aggregation for platelet lifetime (7-10 days)",
        txa2_inhibition=95,
        onset_h=0.5, peak_effect_h=1.0, duration_h=168,  # 7 days (platelet lifespan)
    ),

    "clopidogrel": CoagDrugEffect(
        name="Clopidogrel (Plavix)", drug_class="P2Y12 inhibitor / Antiplatelet",
        mechanism="Active metabolite irreversibly blocks ADP receptor (P2Y12) on platelets → inhibits aggregation",
        adp_receptor_block=80,
        onset_h=2.0, peak_effect_h=6.0, duration_h=168,
    ),

    "rivaroxaban": CoagDrugEffect(
        name="Rivaroxaban (Xarelto)", drug_class="Direct oral anticoagulant (DOAC)",
        mechanism="Directly and reversibly inhibits Factor Xa → blocks conversion of prothrombin to thrombin",
        direct_xa_inhibition=90,
        onset_h=1.0, peak_effect_h=3.0, duration_h=12,
        affects_pt_inr=True, affects_aptt=True,
    ),

    "dabigatran": CoagDrugEffect(
        name="Dabigatran (Pradaxa)", drug_class="Direct thrombin inhibitor (DOAC)",
        mechanism="Directly and reversibly inhibits Factor IIa (thrombin) → prevents fibrin formation",
        direct_iia_inhibition=85,
        onset_h=1.0, peak_effect_h=2.0, duration_h=12,
        affects_aptt=True,
    ),

    "heparin": CoagDrugEffect(
        name="Heparin (UFH)", drug_class="Indirect thrombin + Xa inhibitor",
        mechanism="Potentiates antithrombin III activity 1000-fold → inactivates thrombin (IIa) and Xa equally",
        at_enhancement=1000,
        onset_h=0.05, peak_effect_h=0.25, duration_h=4,
        affects_aptt=True,
    ),

    "enoxaparin": CoagDrugEffect(
        name="Enoxaparin (LMWH)", drug_class="Low molecular weight heparin",
        mechanism="Potentiates antithrombin III → preferentially inhibits Factor Xa (3:1 ratio vs IIa)",
        at_xa_selective=500,
        onset_h=0.5, peak_effect_h=3.0, duration_h=12,
    ),

    "ibuprofen": CoagDrugEffect(
        name="Ibuprofen", drug_class="NSAID / Reversible COX-1 inhibitor",
        mechanism="Reversibly inhibits COX-1 → temporarily reduces TXA2 → mild antiplatelet effect (duration of drug exposure only)",
        txa2_inhibition=50,
        onset_h=0.5, peak_effect_h=2.0, duration_h=8,
    ),
}


# ════════════════════════════════════════════════════════════════
# COAGULATION CALCULATIONS
# ════════════════════════════════════════════════════════════════

def calculate_inr(state: CoagulationState) -> float:
    """
    Calculate INR from coagulation factor activities.

    INR is primarily determined by the extrinsic pathway (Factor VII has
    the shortest half-life, so it's the first to fall with warfarin).

    INR = (PT_patient / PT_normal)^ISI
    Simplified: INR ≈ f(Factor II, VII, X activity)
    """
    f2 = state.factors["II"].activity / 100
    f7 = state.factors["VII"].activity / 100
    f10 = state.factors["X"].activity / 100

    # Weighted combination (VII has most impact on PT)
    combined = 0.3 * f2 + 0.4 * f7 + 0.3 * f10
    combined = max(combined, 0.05)

    # INR is approximately inverse of factor activity
    inr = 1.0 / combined
    return round(min(inr, 10.0), 1)  # Cap at 10


def calculate_aptt(state: CoagulationState) -> float:
    """
    Calculate aPTT from intrinsic pathway factor activities.

    Normal aPTT: 25-35 seconds.
    Prolonged by: heparin, hemophilia (VIII/IX deficiency), factor XI/XII deficiency.
    """
    f8 = state.factors["VIII"].activity / 100
    f9 = state.factors["IX"].activity / 100
    f11 = state.factors["XI"].activity / 100
    f12 = state.factors["XII"].activity / 100

    # Antithrombin effect (heparin)
    at_factor = 1.0
    if state.antithrombin_activity > 100:
        at_factor = state.antithrombin_activity / 100

    combined = (f8 * f9 * f11 * f12) ** 0.25
    combined = max(combined, 0.05)

    # Normal aPTT ~30s, inversely related to factor activity
    aptt = 30.0 / combined * at_factor
    return round(min(aptt, 200), 1)


def calculate_bleeding_risk(state: CoagulationState) -> dict:
    """Calculate overall bleeding risk score."""
    inr = calculate_inr(state)
    aptt = calculate_aptt(state)

    # Platelet contribution to bleeding risk
    platelet_function = (
        state.platelets.aggregation *
        (state.platelets.txa2_activity / 100) *
        (state.platelets.adp_receptor / 100) *
        (state.platelets.gpiib_iiia / 100)
    ) / 100

    platelet_count_risk = 0
    if state.platelets.count < 50000:
        platelet_count_risk = 0.8
    elif state.platelets.count < 100000:
        platelet_count_risk = 0.4
    elif state.platelets.count < 150000:
        platelet_count_risk = 0.2

    # Combined risk score (0-1)
    coag_risk = min(1.0, (inr - 1.0) / 4.0) if inr > 1.0 else 0
    aptt_risk = min(1.0, (aptt - 35) / 100) if aptt > 35 else 0
    platelet_risk = (1 - platelet_function / 100) + platelet_count_risk

    overall_risk = min(1.0, coag_risk * 0.35 + aptt_risk * 0.25 + platelet_risk * 0.40)

    # Risk category
    if overall_risk > 0.7:
        category = "HIGH — Active bleeding risk. Hold procedures, consider reversal."
    elif overall_risk > 0.4:
        category = "MODERATE — Increased bleeding tendency. Monitor closely."
    elif overall_risk > 0.2:
        category = "MILD — Slight increase in bleeding time. Usually clinically insignificant."
    else:
        category = "LOW — Normal hemostasis."

    return {
        "inr": inr,
        "aptt_seconds": aptt,
        "platelet_count": state.platelets.count,
        "platelet_function": round(platelet_function, 1),
        "overall_risk_score": round(overall_risk, 3),
        "risk_category": category,
        "coag_cascade_risk": round(coag_risk, 3),
        "platelet_risk": round(platelet_risk, 3),
    }


def simulate_coagulation_effects(
    drug_name: str,
    plasma_concentration: float = 1.0,
    ec50: float = 5.0,
    time_h: float = 1.0,
    days_on_drug: float = 0,
    baseline_state: CoagulationState | None = None,
) -> dict:
    """
    Simulate drug effects on the coagulation system.

    Args:
        drug_name: Drug name
        plasma_concentration: Current plasma concentration
        ec50: Concentration for 50% effect
        time_h: Time since last dose
        days_on_drug: For drugs with delayed onset (warfarin)
        baseline_state: Pre-existing coagulation state

    Returns:
        Complete coagulation assessment
    """
    state = baseline_state or CoagulationState()
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    drug_effect = COAG_DRUG_EFFECTS.get(key)

    if not drug_effect:
        # No known coagulation effect
        risk = calculate_bleeding_risk(state)
        return {
            "drug": drug_name,
            "has_coag_effect": False,
            "state": _state_to_dict(state),
            "bleeding_risk": risk,
            "clinical_summary": "No significant coagulation effects for this drug",
        }

    # Concentration-effect
    effect_frac = plasma_concentration / (ec50 + plasma_concentration)

    # Time-effect
    if time_h < drug_effect.onset_h:
        time_factor = 0
    elif time_h < drug_effect.peak_effect_h:
        time_factor = (time_h - drug_effect.onset_h) / (drug_effect.peak_effect_h - drug_effect.onset_h)
    else:
        time_factor = 1.0

    # Warfarin special: delayed onset based on factor half-lives
    if drug_effect.time_to_full_effect_days > 0 and days_on_drug >= 0:
        # Each factor depletes based on its own half-life
        for fnum, factor in state.factors.items():
            if factor.vitamin_k_dependent:
                reduction_attr = f"factor_{fnum.lower()}_reduction"
                max_reduction = getattr(drug_effect, reduction_attr, 0)
                if max_reduction > 0:
                    # Factor decays with its own half-life
                    n_halflives = (days_on_drug * 24) / factor.half_life_h
                    depletion = max_reduction * effect_frac * (1 - 0.5 ** n_halflives)
                    factor.activity = max(5, 100 - depletion)
    else:
        # Immediate effect drugs
        scaled = effect_frac * min(time_factor, 1.0)

        # Apply factor reductions
        if drug_effect.factor_ii_reduction:
            state.factors["II"].activity = max(5, 100 - drug_effect.factor_ii_reduction * scaled)
        if drug_effect.factor_vii_reduction:
            state.factors["VII"].activity = max(5, 100 - drug_effect.factor_vii_reduction * scaled)
        if drug_effect.factor_ix_reduction:
            state.factors["IX"].activity = max(5, 100 - drug_effect.factor_ix_reduction * scaled)
        if drug_effect.factor_x_reduction:
            state.factors["X"].activity = max(5, 100 - drug_effect.factor_x_reduction * scaled)

    # Direct inhibitors
    if drug_effect.direct_xa_inhibition:
        # Reduce effective Xa activity
        xa_block = drug_effect.direct_xa_inhibition * effect_frac * min(time_factor, 1.0)
        state.factors["X"].activity = max(5, 100 - xa_block)

    if drug_effect.direct_iia_inhibition:
        iia_block = drug_effect.direct_iia_inhibition * effect_frac * min(time_factor, 1.0)
        state.factors["II"].activity = max(5, 100 - iia_block)

    # Antithrombin enhancement (heparin)
    if drug_effect.at_enhancement:
        state.antithrombin_activity = 100 + drug_effect.at_enhancement * effect_frac * min(time_factor, 1.0)
    if drug_effect.at_xa_selective:
        state.antithrombin_activity = 100 + drug_effect.at_xa_selective * effect_frac * min(time_factor, 1.0)

    # Platelet effects
    scaled_platelet = effect_frac * min(time_factor, 1.0)
    if drug_effect.txa2_inhibition:
        state.platelets.txa2_activity = max(5, 100 - drug_effect.txa2_inhibition * scaled_platelet)
    if drug_effect.adp_receptor_block:
        state.platelets.adp_receptor = max(5, 100 - drug_effect.adp_receptor_block * scaled_platelet)
    if drug_effect.gpiib_iiia_block:
        state.platelets.gpiib_iiia = max(5, 100 - drug_effect.gpiib_iiia_block * scaled_platelet)

    # Calculate results
    risk = calculate_bleeding_risk(state)

    # Factor activity summary
    factor_summary = []
    for fnum, factor in state.factors.items():
        if factor.activity < 95:
            factor_summary.append({
                "factor": f"Factor {fnum}",
                "name": factor.name,
                "activity": round(factor.activity, 1),
                "half_life_h": factor.half_life_h,
                "status": (
                    "CRITICAL" if factor.activity < 20
                    else "LOW" if factor.activity < 50
                    else "REDUCED" if factor.activity < 80
                    else "NORMAL"
                ),
            })

    return {
        "drug": drug_effect.name,
        "drug_class": drug_effect.drug_class,
        "mechanism": drug_effect.mechanism,
        "has_coag_effect": True,
        "state": _state_to_dict(state),
        "affected_factors": factor_summary,
        "bleeding_risk": risk,
        "monitoring": {
            "test": "PT/INR" if drug_effect.affects_pt_inr else "aPTT" if drug_effect.affects_aptt else "Platelet function",
            "target_range": f"INR {drug_effect.target_inr[0]}-{drug_effect.target_inr[1]}" if drug_effect.target_inr[0] > 0 else "N/A",
        },
        "clinical_summary": _generate_coag_summary(drug_effect, risk, state),
        "reversal_agents": _get_reversal_agents(key),
    }


def _state_to_dict(state: CoagulationState) -> dict:
    """Convert state to serializable dict."""
    return {
        "factors": {
            k: {"name": v.name, "activity": round(v.activity, 1), "half_life_h": v.half_life_h}
            for k, v in state.factors.items()
        },
        "platelets": {
            "count": state.platelets.count,
            "aggregation": round(state.platelets.aggregation, 1),
            "txa2": round(state.platelets.txa2_activity, 1),
            "adp_receptor": round(state.platelets.adp_receptor, 1),
        },
        "antithrombin": round(state.antithrombin_activity, 1),
    }


def _generate_coag_summary(drug_effect: CoagDrugEffect, risk: dict, state: CoagulationState) -> str:
    """Generate clinical summary of coagulation effects."""
    parts = [drug_effect.mechanism + "."]

    if risk["inr"] > 1.5:
        parts.append(f"INR elevated to {risk['inr']} — extrinsic pathway prolonged.")
    if risk["aptt_seconds"] > 40:
        parts.append(f"aPTT prolonged to {risk['aptt_seconds']}s — intrinsic pathway affected.")
    if state.platelets.txa2_activity < 50:
        parts.append("Platelet TXA2 suppressed — anti-aggregatory effect active.")
    if state.platelets.adp_receptor < 50:
        parts.append("ADP receptor blocked — platelet aggregation impaired.")

    parts.append(f"Overall bleeding risk: {risk['risk_category']}")

    return " ".join(parts)


def _get_reversal_agents(drug_key: str) -> list[dict]:
    """Get appropriate reversal agents for a drug."""
    agents = {
        "warfarin": [
            {"agent": "Vitamin K (phytonadione)", "dose": "1-10 mg IV/PO", "onset": "6-24 hours", "notes": "Full reversal takes 24-48h"},
            {"agent": "Fresh Frozen Plasma (FFP)", "dose": "10-15 mL/kg", "onset": "Immediate", "notes": "For active bleeding"},
            {"agent": "4-Factor PCC (Kcentra)", "dose": "25-50 IU/kg", "onset": "10-30 min", "notes": "Preferred for life-threatening bleeding"},
        ],
        "heparin": [
            {"agent": "Protamine sulfate", "dose": "1 mg per 100 units heparin", "onset": "5 minutes", "notes": "Max single dose 50mg"},
        ],
        "rivaroxaban": [
            {"agent": "Andexanet alfa (Andexxa)", "dose": "400-800 mg IV", "onset": "2-5 min", "notes": "Specific Factor Xa inhibitor reversal"},
            {"agent": "4-Factor PCC", "dose": "50 IU/kg", "onset": "10-30 min", "notes": "Off-label but used in emergencies"},
        ],
        "dabigatran": [
            {"agent": "Idarucizumab (Praxbind)", "dose": "5g IV", "onset": "Minutes", "notes": "Specific dabigatran reversal — binds with 350x affinity"},
        ],
        "aspirin": [
            {"agent": "Platelet transfusion", "dose": "1 unit", "onset": "Immediate", "notes": "For life-threatening bleeding only"},
            {"agent": "Desmopressin (DDAVP)", "dose": "0.3 µg/kg IV", "onset": "30 min", "notes": "Enhances residual platelet function"},
        ],
    }
    return agents.get(drug_key, [{"agent": "Supportive care", "dose": "N/A", "onset": "N/A", "notes": "No specific reversal agent"}])
