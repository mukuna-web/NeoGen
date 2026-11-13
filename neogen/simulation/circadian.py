"""
Circadian Rhythm Model
========================
Models time-of-day variation in drug pharmacokinetics.

The human body runs on a ~24-hour internal clock (circadian rhythm) that
affects nearly every aspect of drug handling:

1. ABSORPTION: Gastric emptying is faster in the morning. Gastric pH varies.
2. DISTRIBUTION: Plasma protein levels fluctuate ~10% over 24h.
3. METABOLISM: CYP3A4 expression peaks at night (2-3x variation).
   CYP2E1 peaks in early morning. CYP1A2 relatively constant.
4. ELIMINATION: GFR is ~20% higher during the day than at night.
   Urine pH varies (more acidic in morning).
5. PHARMACODYNAMICS: Drug targets have circadian expression patterns.

Clinical importance:
- Statins: taken at bedtime because cholesterol synthesis peaks at night
- Corticosteroids: given in morning to mimic natural cortisol peak (8am)
- Aspirin: evening dosing may be more effective for morning heart attacks
- Antihypertensives: BP has a morning surge → timing matters
- Chemotherapy: chronotherapy shows 2-5x differences in toxicity by timing

Reference: Dallmann et al., Annu Rev Pharmacol Toxicol 2014
"""
from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)


def get_circadian_modifiers(hour_of_day: float) -> dict:
    """
    Get circadian modification factors for PK parameters at a given time of day.

    Args:
        hour_of_day: 0-24 (0=midnight, 6=6am, 12=noon, 18=6pm)

    Returns:
        Dict of modification factors (1.0 = no change from average)
    """
    h = hour_of_day % 24
    # Convert to radians for sinusoidal modeling
    rad = 2 * math.pi * h / 24

    # ── ABSORPTION ──
    # Gastric emptying is fastest in morning (8-10am), slowest at night
    # Peak at 9am (h=9), trough at 9pm (h=21)
    gastric_emptying = 1.0 + 0.20 * math.cos(rad - 2 * math.pi * 9 / 24)
    # ka (absorption rate) scales with gastric emptying
    ka_modifier = gastric_emptying

    # Gastric pH: lowest (most acidic) in early morning, highest in evening
    # pH ~1.5 at 6am, ~2.5 at 6pm
    gastric_ph = 2.0 - 0.5 * math.cos(rad - 2 * math.pi * 6 / 24)

    # ── METABOLISM ──
    # CYP3A4: peaks at night (~2am), trough in afternoon (~2pm)
    # Amplitude: ±30% of mean
    cyp3a4_modifier = 1.0 + 0.30 * math.cos(rad - 2 * math.pi * 2 / 24)

    # CYP2E1: peaks in early morning (~6am)
    cyp2e1_modifier = 1.0 + 0.20 * math.cos(rad - 2 * math.pi * 6 / 24)

    # CYP1A2: relatively constant (±10%)
    cyp1a2_modifier = 1.0 + 0.10 * math.cos(rad - 2 * math.pi * 8 / 24)

    # CYP2D6: minimal circadian variation
    cyp2d6_modifier = 1.0 + 0.05 * math.cos(rad)

    # Overall hepatic clearance (weighted by CYP3A4 dominance)
    hepatic_cl_modifier = 0.6 * cyp3a4_modifier + 0.2 * cyp2e1_modifier + 0.2 * cyp1a2_modifier

    # ── ELIMINATION ──
    # GFR: ~20% higher during daytime (peak at 2pm), lowest at 4am
    gfr_modifier = 1.0 + 0.10 * math.cos(rad - 2 * math.pi * 14 / 24)

    # Urine pH: more acidic in morning (pH ~5.5), more alkaline in evening (~7.0)
    urine_ph = 6.25 + 0.75 * math.cos(rad - 2 * math.pi * 18 / 24)

    # ── DISTRIBUTION ──
    # Plasma albumin: slight variation (~5%)
    albumin_modifier = 1.0 + 0.05 * math.cos(rad - 2 * math.pi * 16 / 24)

    # Blood flow: cardiac output slightly higher during day
    cardiac_output_modifier = 1.0 + 0.08 * math.cos(rad - 2 * math.pi * 12 / 24)

    # ── PHARMACODYNAMICS ──
    # Blood pressure: morning surge (peak 8-10am), dip at 3am
    bp_modifier = 1.0 + 0.12 * math.cos(rad - 2 * math.pi * 9 / 24)

    # Heart rate: higher during day
    hr_modifier = 1.0 + 0.10 * math.cos(rad - 2 * math.pi * 14 / 24)

    # Cortisol: peaks at 8am, lowest at midnight
    cortisol_modifier = 1.0 + 0.80 * math.cos(rad - 2 * math.pi * 8 / 24)

    # Melatonin: peaks at 2am, suppressed during day
    melatonin_modifier = 1.0 + 0.90 * math.cos(rad - 2 * math.pi * 2 / 24)

    # Pain sensitivity: highest in early morning (lower pain threshold)
    pain_sensitivity = 1.0 + 0.15 * math.cos(rad - 2 * math.pi * 6 / 24)

    return {
        "hour": h,
        "time_label": _hour_to_label(h),

        "absorption": {
            "ka_modifier": round(ka_modifier, 3),
            "gastric_emptying": round(gastric_emptying, 3),
            "gastric_ph": round(gastric_ph, 2),
            "note": "Faster absorption in morning, slower at night",
        },
        "metabolism": {
            "hepatic_cl_modifier": round(hepatic_cl_modifier, 3),
            "cyp3a4": round(cyp3a4_modifier, 3),
            "cyp2e1": round(cyp2e1_modifier, 3),
            "cyp1a2": round(cyp1a2_modifier, 3),
            "cyp2d6": round(cyp2d6_modifier, 3),
            "note": "CYP3A4 peaks at ~2am (30% higher than afternoon)",
        },
        "elimination": {
            "gfr_modifier": round(gfr_modifier, 3),
            "urine_ph": round(urine_ph, 2),
            "note": "GFR 20% higher during daytime",
        },
        "distribution": {
            "albumin_modifier": round(albumin_modifier, 3),
            "cardiac_output_modifier": round(cardiac_output_modifier, 3),
        },
        "physiology": {
            "bp_modifier": round(bp_modifier, 3),
            "hr_modifier": round(hr_modifier, 3),
            "cortisol": round(cortisol_modifier, 3),
            "melatonin": round(melatonin_modifier, 3),
            "pain_sensitivity": round(pain_sensitivity, 3),
        },

        # Combined PK impact
        "pk_impact": {
            "cmax_modifier": round(ka_modifier * 0.7 + 0.3, 3),
            "auc_modifier": round(1.0 / hepatic_cl_modifier, 3),
            "half_life_modifier": round(1.0 / hepatic_cl_modifier, 3),
            "clearance_modifier": round(hepatic_cl_modifier * 0.7 + gfr_modifier * 0.3, 3),
        },

        # Clinical recommendations
        "recommendations": _get_timing_recommendations(h),
    }


def _hour_to_label(h: float) -> str:
    """Convert hour to human-readable label."""
    labels = {
        0: "Midnight", 3: "3 AM (deep sleep)", 6: "6 AM (waking)",
        8: "8 AM (morning)", 9: "9 AM (cortisol peak)",
        12: "Noon", 14: "2 PM (afternoon)", 18: "6 PM (evening)",
        21: "9 PM (pre-sleep)", 23: "11 PM (melatonin rising)",
    }
    closest = min(labels.keys(), key=lambda x: abs(x - h))
    if abs(closest - h) < 1.5:
        return labels[closest]
    h_int = int(h)
    ampm = "AM" if h_int < 12 else "PM"
    h12 = h_int % 12 or 12
    return f"{h12} {ampm}"


def _get_timing_recommendations(h: float) -> list[str]:
    """Get drug timing recommendations for the given hour."""
    recs = []
    if 6 <= h <= 10:
        recs.append("Good time for: corticosteroids (mimic cortisol peak), antihypertensives (morning BP surge)")
        recs.append("Caution: pain sensitivity highest — analgesic dosing may need adjustment")
    elif 20 <= h <= 24 or 0 <= h <= 2:
        recs.append("Good time for: statins (cholesterol synthesis peaks at night), H2-blockers")
        recs.append("CYP3A4 activity elevated — drugs metabolized by CYP3A4 cleared faster")
    elif 12 <= h <= 16:
        recs.append("GFR at peak — renally cleared drugs eliminated faster")
        recs.append("CYP3A4 at trough — CYP3A4 substrates have higher levels")
    return recs


# ════════════════════════════════════════════════════════════════
# DRUG-SPECIFIC CHRONOTHERAPY
# ════════════════════════════════════════════════════════════════

CHRONOTHERAPY_DRUGS = {
    "simvastatin": {
        "best_time": "Evening (9-10 PM)",
        "reason": "HMG-CoA reductase (cholesterol synthesis) peaks at midnight. Evening dosing gives 30% more cholesterol reduction than morning dosing.",
        "evidence": "Strong — multiple clinical trials confirm",
    },
    "atorvastatin": {
        "best_time": "Any time (long half-life)",
        "reason": "t½=14h — long enough that timing doesn't matter as much. But evening still slightly better.",
        "evidence": "Moderate — some studies show no difference",
    },
    "aspirin": {
        "best_time": "Evening (before bed)",
        "reason": "Morning platelet reactivity is highest. Evening aspirin provides better antiplatelet coverage during the morning surge when most MIs occur.",
        "evidence": "Emerging — ASPREE timing substudy",
    },
    "omeprazole": {
        "best_time": "Morning (30 min before breakfast)",
        "reason": "PPIs only inhibit ACTIVE proton pumps. Breakfast stimulates acid secretion → activates pumps → PPI works.",
        "evidence": "Strong — pharmacological mechanism",
    },
    "prednisolone": {
        "best_time": "Morning (8 AM)",
        "reason": "Mimics natural cortisol peak → less HPA axis suppression than evening dosing.",
        "evidence": "Strong — standard clinical practice",
    },
    "amlodipine": {
        "best_time": "Evening",
        "reason": "BP dips at night (normal dipping). Evening dosing restores dipping pattern in non-dippers → reduced cardiovascular risk.",
        "evidence": "Strong — MAPEC trial showed 61% CV risk reduction with bedtime dosing",
    },
    "metformin": {
        "best_time": "With meals (morning and evening)",
        "reason": "Reduces postprandial glucose. GI side effects less with food. BID dosing preferred.",
        "evidence": "Strong — standard practice",
    },
    "levothyroxine": {
        "best_time": "Morning (30-60 min before breakfast)",
        "reason": "Empty stomach needed for absorption. Food reduces absorption by 40%. Coffee reduces by 30%.",
        "evidence": "Strong — FDA label recommendation",
    },
}


def get_chronotherapy(drug_name: str) -> dict | None:
    """Get chronotherapy recommendation for a drug."""
    key = drug_name.lower().replace(" ", "_")
    return CHRONOTHERAPY_DRUGS.get(key)
