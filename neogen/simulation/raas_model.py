"""
Renin-Angiotensin-Aldosterone System (RAAS) + Baroreceptor Reflex
==================================================================
Models the body's primary blood pressure regulation system.

RAAS Cascade:
  Renin (kidney) → cleaves Angiotensinogen → Angiotensin I
  → ACE (lung) → Angiotensin II
  → AT1 receptor → vasoconstriction + aldosterone release
  → Aldosterone → Na+ retention → ↑ blood volume → ↑ BP

Drug targets:
  - ACE inhibitors (enalapril, lisinopril): block ACE → ↓ Ang II
  - ARBs (losartan, valsartan): block AT1 receptor → ↓ vasoconstriction
  - Direct renin inhibitors (aliskiren): block renin → ↓ Ang I
  - Aldosterone antagonists (spironolactone): block aldosterone receptor
  - Beta-blockers: ↓ renin release from kidney

Baroreceptor Reflex:
  BP drops → baroreceptors detect → sympathetic activation →
  → ↑ heart rate, ↑ contractility, ↑ vasoconstriction
  → compensatory BP increase

  This is why vasodilators (nifedipine) cause reflex tachycardia
  but slower-acting ones (amlodipine) don't.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HemodynamicState:
    """Complete hemodynamic state."""
    systolic_bp: float = 120     # mmHg
    diastolic_bp: float = 80     # mmHg
    mean_arterial_pressure: float = 93  # MAP = DBP + 1/3(SBP-DBP)
    heart_rate: float = 72       # bpm
    cardiac_output: float = 5.0  # L/min
    total_peripheral_resistance: float = 18.6  # mmHg·min/L (MAP/CO)
    stroke_volume: float = 70    # mL

    # RAAS components (% of normal)
    renin_activity: float = 100
    angiotensin_I: float = 100
    angiotensin_II: float = 100
    aldosterone: float = 100
    ace_activity: float = 100

    # Volumes
    plasma_volume: float = 100   # % of normal
    sodium_retention: float = 100  # %

    # Sympathetic
    sympathetic_tone: float = 100  # %
    parasympathetic_tone: float = 100  # %


@dataclass
class RAASdrug:
    """RAAS-modifying drug profile."""
    name: str
    drug_class: str
    mechanism: str
    # Effects on RAAS components (% reduction/increase)
    ace_inhibition: float = 0        # % ACE blocked
    at1_blockade: float = 0          # % AT1 receptor blocked
    renin_inhibition: float = 0      # % renin blocked
    aldosterone_blockade: float = 0  # % aldosterone receptor blocked
    beta_blockade: float = 0         # % beta-adrenergic blocked
    alpha_blockade: float = 0        # % alpha-1 blocked (vasodilation)
    ccb_effect: float = 0            # % calcium channel block (vasodilation)
    diuretic_effect: float = 0       # % increase in Na+ excretion
    # Hemodynamic effects
    sbp_reduction: float = 0         # mmHg reduction expected
    dbp_reduction: float = 0
    hr_change: float = 0             # bpm change (negative = decrease)
    # Baroreceptor
    triggers_reflex_tachycardia: bool = False
    onset_h: float = 1.0
    peak_h: float = 4.0


RAAS_DRUGS = {
    "lisinopril": RAASdrug(
        name="Lisinopril", drug_class="ACE Inhibitor",
        mechanism="Competitively inhibits ACE → blocks conversion of Ang I → Ang II → reduces vasoconstriction and aldosterone",
        ace_inhibition=85, sbp_reduction=15, dbp_reduction=10, hr_change=0,
        onset_h=1.0, peak_h=6.0,
    ),
    "enalapril": RAASdrug(
        name="Enalapril", drug_class="ACE Inhibitor (prodrug)",
        mechanism="Prodrug → enalaprilat (active). Inhibits ACE → ↓ Ang II",
        ace_inhibition=80, sbp_reduction=14, dbp_reduction=9, hr_change=0,
        onset_h=2.0, peak_h=6.0,
    ),
    "losartan": RAASdrug(
        name="Losartan", drug_class="ARB (Angiotensin II Receptor Blocker)",
        mechanism="Selectively blocks AT1 receptor → Ang II cannot cause vasoconstriction. Does NOT affect bradykinin (less cough than ACEI).",
        at1_blockade=85, sbp_reduction=13, dbp_reduction=8, hr_change=0,
        onset_h=1.0, peak_h=3.0,
    ),
    "valsartan": RAASdrug(
        name="Valsartan", drug_class="ARB",
        mechanism="Blocks AT1 receptor → vasodilation + reduced aldosterone",
        at1_blockade=90, sbp_reduction=14, dbp_reduction=9, hr_change=0,
        onset_h=2.0, peak_h=4.0,
    ),
    "aliskiren": RAASdrug(
        name="Aliskiren", drug_class="Direct Renin Inhibitor",
        mechanism="Directly inhibits renin → blocks very first step of RAAS cascade",
        renin_inhibition=90, sbp_reduction=12, dbp_reduction=7, hr_change=0,
        onset_h=1.0, peak_h=3.0,
    ),
    "spironolactone": RAASdrug(
        name="Spironolactone", drug_class="Aldosterone Antagonist (K+-sparing diuretic)",
        mechanism="Blocks aldosterone receptor in collecting duct → ↓ Na+ retention, ↓ K+ excretion. Also anti-androgenic.",
        aldosterone_blockade=80, diuretic_effect=20, sbp_reduction=10, dbp_reduction=6, hr_change=0,
        onset_h=24.0, peak_h=72.0,
    ),
    "metoprolol": RAASdrug(
        name="Metoprolol", drug_class="Beta-1 Selective Blocker",
        mechanism="Blocks β1-adrenergic receptors → ↓ heart rate, ↓ contractility, ↓ renin release",
        beta_blockade=80, sbp_reduction=12, dbp_reduction=8, hr_change=-15,
        onset_h=1.0, peak_h=2.0,
    ),
    "propranolol": RAASdrug(
        name="Propranolol", drug_class="Non-selective Beta Blocker",
        mechanism="Blocks β1 + β2 receptors → ↓ HR, ↓ renin, bronchospasm risk (β2 in lungs)",
        beta_blockade=90, sbp_reduction=14, dbp_reduction=10, hr_change=-18,
        onset_h=1.0, peak_h=2.0,
    ),
    "amlodipine": RAASdrug(
        name="Amlodipine", drug_class="Dihydropyridine CCB",
        mechanism="Blocks L-type Ca2+ channels in vascular smooth muscle → vasodilation. Slow onset → minimal reflex tachycardia.",
        ccb_effect=80, sbp_reduction=15, dbp_reduction=10, hr_change=2,
        onset_h=6.0, peak_h=8.0, triggers_reflex_tachycardia=False,
    ),
    "nifedipine": RAASdrug(
        name="Nifedipine (IR)", drug_class="Dihydropyridine CCB (immediate release)",
        mechanism="Rapid vasodilation → baroreceptor reflex → REFLEX TACHYCARDIA. This is why IR nifedipine is dangerous.",
        ccb_effect=85, sbp_reduction=20, dbp_reduction=12, hr_change=15,
        onset_h=0.25, peak_h=0.5, triggers_reflex_tachycardia=True,
    ),
    "prazosin": RAASdrug(
        name="Prazosin", drug_class="Alpha-1 Blocker",
        mechanism="Blocks α1-adrenergic receptors in vascular smooth muscle → vasodilation. First-dose hypotension risk.",
        alpha_blockade=80, sbp_reduction=12, dbp_reduction=8, hr_change=5,
        onset_h=1.0, peak_h=3.0, triggers_reflex_tachycardia=True,
    ),
    "hydrochlorothiazide": RAASdrug(
        name="Hydrochlorothiazide", drug_class="Thiazide Diuretic",
        mechanism="Inhibits NCC in distal tubule → ↑ Na+/water excretion → ↓ plasma volume → ↓ BP",
        diuretic_effect=50, sbp_reduction=12, dbp_reduction=6, hr_change=0,
        onset_h=2.0, peak_h=4.0,
    ),
    "furosemide": RAASdrug(
        name="Furosemide", drug_class="Loop Diuretic",
        mechanism="Inhibits NKCC2 in thick ascending limb → massive Na+/K+/Cl-/water excretion. Potent but short-acting.",
        diuretic_effect=90, sbp_reduction=15, dbp_reduction=8, hr_change=5,
        onset_h=0.5, peak_h=1.0,
    ),
}


def simulate_raas(
    drug_name: str,
    baseline_sbp: float = 140,
    baseline_dbp: float = 90,
    baseline_hr: float = 78,
    time_h: float = 4.0,
    plasma_conc_fraction: float = 1.0,
) -> dict:
    """Simulate RAAS effects and hemodynamic changes."""
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    drug = RAAS_DRUGS.get(key)

    if not drug:
        return {"drug": drug_name, "has_raas_effect": False,
                "message": "No RAAS/hemodynamic profile available"}

    # Time-effect relationship
    if time_h < drug.onset_h:
        time_factor = 0
    elif time_h < drug.peak_h:
        time_factor = (time_h - drug.onset_h) / (drug.peak_h - drug.onset_h)
    else:
        time_factor = 1.0

    effect = plasma_conc_fraction * time_factor

    # Calculate hemodynamic changes
    state = HemodynamicState(
        systolic_bp=baseline_sbp,
        diastolic_bp=baseline_dbp,
        heart_rate=baseline_hr,
    )

    sbp_change = -drug.sbp_reduction * effect
    dbp_change = -drug.dbp_reduction * effect
    hr_change = drug.hr_change * effect

    # Baroreceptor reflex
    reflex_hr_increase = 0
    if drug.triggers_reflex_tachycardia and sbp_change < -10:
        reflex_hr_increase = abs(sbp_change) * 0.5  # ~0.5 bpm per mmHg drop
        hr_change += reflex_hr_increase

    new_sbp = max(70, baseline_sbp + sbp_change)
    new_dbp = max(40, baseline_dbp + dbp_change)
    new_hr = max(40, min(150, baseline_hr + hr_change))
    new_map = new_dbp + (new_sbp - new_dbp) / 3

    # RAAS component changes
    raas_changes = {}
    if drug.ace_inhibition > 0:
        raas_changes["ACE activity"] = round(100 - drug.ace_inhibition * effect)
        raas_changes["Angiotensin II"] = round(100 - drug.ace_inhibition * 0.9 * effect)
        raas_changes["Aldosterone"] = round(100 - drug.ace_inhibition * 0.6 * effect)
        raas_changes["Bradykinin"] = round(100 + drug.ace_inhibition * 0.8 * effect)  # Elevated (cough!)
    if drug.at1_blockade > 0:
        raas_changes["AT1 receptor"] = round(100 - drug.at1_blockade * effect)
        raas_changes["Renin (compensatory)"] = round(100 + drug.at1_blockade * 0.5 * effect)
    if drug.renin_inhibition > 0:
        raas_changes["Renin activity"] = round(100 - drug.renin_inhibition * effect)
        raas_changes["Angiotensin I"] = round(100 - drug.renin_inhibition * 0.8 * effect)
    if drug.beta_blockade > 0:
        raas_changes["Sympathetic tone"] = round(100 - drug.beta_blockade * 0.5 * effect)
        raas_changes["Renin release"] = round(100 - drug.beta_blockade * 0.6 * effect)

    # Warnings
    warnings = []
    if new_sbp < 90:
        warnings.append("HYPOTENSION — SBP < 90 mmHg. Risk of dizziness, syncope, organ hypoperfusion.")
    if new_hr < 50:
        warnings.append("BRADYCARDIA — HR < 50 bpm. Monitor for symptomatic bradycardia.")
    if new_hr > 100:
        warnings.append(f"TACHYCARDIA — HR {new_hr:.0f} bpm. Reflex response to vasodilation.")
    if drug.ace_inhibition > 0:
        warnings.append("ACE inhibitor — monitor for cough (bradykinin accumulation) and angioedema (rare but serious).")
        warnings.append("Contraindicated in pregnancy (teratogenic). Check potassium (hyperkalemia risk).")
    if drug.aldosterone_blockade > 0:
        warnings.append("Aldosterone antagonist — MONITOR POTASSIUM closely. Hyperkalemia risk, especially with ACEI/ARB.")

    return {
        "drug": drug.name,
        "drug_class": drug.drug_class,
        "mechanism": drug.mechanism,
        "has_raas_effect": True,
        "hemodynamics": {
            "sbp": {"before": baseline_sbp, "after": round(new_sbp, 1), "change": round(sbp_change, 1)},
            "dbp": {"before": baseline_dbp, "after": round(new_dbp, 1), "change": round(dbp_change, 1)},
            "map": {"before": round(baseline_dbp + (baseline_sbp - baseline_dbp)/3, 1), "after": round(new_map, 1)},
            "hr": {"before": baseline_hr, "after": round(new_hr, 1), "change": round(hr_change, 1)},
            "reflex_tachycardia": reflex_hr_increase > 2,
            "reflex_hr_increase": round(reflex_hr_increase, 1),
        },
        "raas_changes": raas_changes,
        "effect_fraction": round(effect, 3),
        "warnings": warnings,
    }
