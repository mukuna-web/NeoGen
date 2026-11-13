"""
Pharmacodynamic (PD) Modeling Engine
=====================================
Models the TIME COURSE of drug effects — not just "this drug affects the brain"
but "pain relief starts at 20min, peaks at 1.5h (67% reduction), fades by 6h."

Key concepts:
1. Emax Model: Effect = (Emax × C^n) / (EC50^n + C^n)
   - Emax = maximum possible effect (e.g., 100% pain relief)
   - EC50 = concentration at 50% of max effect
   - n = Hill coefficient (steepness of dose-response)

2. Effect Compartment: Drug concentration at the receptor site lags behind
   plasma concentration. ke0 = equilibration rate constant.
   Ce(t) follows plasma with delay → dCe/dt = ke0 × (Cp - Ce)

3. Multi-effect: One drug can have multiple effects simultaneously
   (aspirin: pain relief peaks at 1h, antiplatelet effect is immediate,
    GI damage accumulates over days)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PDEffect:
    """A single pharmacodynamic effect of a drug."""
    name: str                         # "Pain relief", "Heart rate reduction"
    organ: str                        # Which organ this effect occurs in
    effect_type: str                  # "therapeutic", "side_effect", "toxic"

    # Emax model parameters
    emax: float = 100.0               # Maximum effect (% or specific unit)
    ec50: float = 5.0                 # Concentration for 50% effect (mg/L)
    hill_n: float = 1.5               # Hill coefficient (steepness)
    baseline: float = 0.0             # Baseline value (e.g., pain=100, HR=72)
    unit: str = "%"                   # Effect unit

    # Effect compartment
    ke0: float = 1.0                  # Equilibration rate (1/h). Higher = faster onset
    # ke0 ~0.5 = slow onset (6h to equilibrate)
    # ke0 ~2.0 = moderate (2h)
    # ke0 ~5.0 = fast (30min)
    # ke0 ~20.0 = near-instant

    # Concentration source
    conc_source: str = "plasma"       # "plasma", "brain", "liver", "kidney", etc.

    # Threshold effects
    onset_threshold: float = 0.0      # Min concentration for any effect
    toxic_threshold: float = 1000.0   # Concentration where toxicity begins

    # Display
    color: str = "#00ccff"            # Chart color
    description: str = ""             # What this effect means clinically


@dataclass
class DrugPDProfile:
    """Complete PD profile for a drug — all effects over time."""
    drug_name: str
    effects: list = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
# DRUG PD PROFILES DATABASE
# ════════════════════════════════════════════════════════════════

PD_PROFILES = {
    "aspirin": DrugPDProfile("aspirin", [
        PDEffect("Pain Relief", "brain", "therapeutic",
                 emax=75, ec50=3.0, hill_n=1.5, baseline=100,
                 ke0=3.0, unit="% pain", conc_source="plasma",
                 color="#00cc88", description="COX-2 inhibition reduces prostaglandin-mediated pain"),
        PDEffect("Anti-inflammatory", "muscle", "therapeutic",
                 emax=60, ec50=5.0, hill_n=1.2, baseline=100,
                 ke0=1.5, unit="% inflammation", conc_source="plasma",
                 color="#44ddff", description="Reduces swelling and inflammation at injury sites"),
        PDEffect("Antiplatelet Effect", "heart", "therapeutic",
                 emax=95, ec50=0.5, hill_n=3.0, baseline=100,
                 ke0=10.0, unit="% aggregation", conc_source="plasma",
                 color="#00ff88", description="Irreversible COX-1 inhibition in platelets — very fast onset"),
        PDEffect("GI Mucosal Damage", "stomach", "side_effect",
                 emax=50, ec50=8.0, hill_n=1.0, baseline=0,
                 ke0=0.5, unit="% damage risk", conc_source="plasma",
                 color="#ff8800", description="Slow-onset GI erosion from reduced protective prostaglandins"),
        PDEffect("Antipyretic (Fever)", "brain", "therapeutic",
                 emax=100, ec50=4.0, hill_n=1.8, baseline=39.0,
                 ke0=2.0, unit="°C", conc_source="brain",
                 color="#ffaa44", description="Reduces hypothalamic set point back toward 37°C"),
    ]),

    "caffeine": DrugPDProfile("caffeine", [
        PDEffect("Alertness", "brain", "therapeutic",
                 emax=80, ec50=2.0, hill_n=1.5, baseline=50,
                 ke0=5.0, unit="% alertness", conc_source="brain",
                 color="#ffff00", description="Adenosine A2A antagonism → increased wakefulness"),
        PDEffect("Heart Rate Increase", "heart", "side_effect",
                 emax=25, ec50=5.0, hill_n=2.0, baseline=72,
                 ke0=3.0, unit="bpm", conc_source="plasma",
                 color="#ff6644", description="Sympathetic activation → tachycardia at high doses"),
        PDEffect("Anxiety", "brain", "side_effect",
                 emax=80, ec50=8.0, hill_n=2.5, baseline=10,
                 ke0=4.0, unit="% anxiety", conc_source="brain",
                 color="#ff4444", description="High doses overstimulate → anxiety, jitteriness"),
        PDEffect("Diuresis", "kidney", "side_effect",
                 emax=60, ec50=3.0, hill_n=1.2, baseline=1.0,
                 ke0=2.0, unit="mL/h urine", conc_source="kidney",
                 color="#44aaff", description="Increased GFR and reduced Na+ reabsorption"),
        PDEffect("Bronchodilation", "lung", "therapeutic",
                 emax=30, ec50=4.0, hill_n=1.0, baseline=100,
                 ke0=2.0, unit="% airway resistance", conc_source="plasma",
                 color="#00ddaa", description="Phosphodiesterase inhibition → smooth muscle relaxation"),
    ]),

    "paracetamol": DrugPDProfile("paracetamol", [
        PDEffect("Pain Relief", "brain", "therapeutic",
                 emax=70, ec50=4.0, hill_n=1.5, baseline=100,
                 ke0=2.5, unit="% pain", conc_source="brain",
                 color="#00cc88", description="Central COX-2 + endocannabinoid pathway → analgesia"),
        PDEffect("Fever Reduction", "brain", "therapeutic",
                 emax=100, ec50=3.0, hill_n=2.0, baseline=39.0,
                 ke0=2.0, unit="°C", conc_source="brain",
                 color="#ffaa44", description="Hypothalamic PGE2 reduction → antipyresis"),
        PDEffect("Liver GSH Depletion", "liver", "toxic",
                 emax=100, ec50=20.0, hill_n=3.0, baseline=100,
                 ke0=0.8, unit="% GSH", conc_source="liver",
                 color="#ff3333", description="NAPQI formation depletes glutathione → at overdose → hepatocyte death"),
        PDEffect("Hepatocyte Damage", "liver", "toxic",
                 emax=90, ec50=40.0, hill_n=4.0, baseline=0,
                 ke0=0.3, unit="% damage", conc_source="liver",
                 color="#ff0000", description="Once GSH <30%, NAPQI forms protein adducts → ALT/AST rise → liver failure"),
    ]),

    "ibuprofen": DrugPDProfile("ibuprofen", [
        PDEffect("Pain Relief", "brain", "therapeutic",
                 emax=65, ec50=5.0, hill_n=1.3, baseline=100,
                 ke0=2.0, unit="% pain", conc_source="plasma",
                 color="#00cc88", description="COX-2 inhibition → anti-inflammatory analgesia"),
        PDEffect("Anti-inflammatory", "muscle", "therapeutic",
                 emax=55, ec50=8.0, hill_n=1.2, baseline=100,
                 ke0=1.0, unit="% inflammation", conc_source="plasma",
                 color="#44ddff", description="Peripheral COX-2 inhibition reduces inflammation"),
        PDEffect("GI Risk", "stomach", "side_effect",
                 emax=35, ec50=12.0, hill_n=1.0, baseline=0,
                 ke0=0.3, unit="% risk", conc_source="plasma",
                 color="#ff8800", description="Reversible COX-1 inhibition (less GI risk than aspirin)"),
        PDEffect("Renal Perfusion", "kidney", "side_effect",
                 emax=30, ec50=15.0, hill_n=1.5, baseline=100,
                 ke0=1.0, unit="% renal flow", conc_source="kidney",
                 color="#ff6644", description="Prostaglandin-dependent renal flow reduced"),
    ]),

    "metformin": DrugPDProfile("metformin", [
        PDEffect("Glucose Lowering", "liver", "therapeutic",
                 emax=35, ec50=1.5, hill_n=1.2, baseline=180,
                 ke0=0.5, unit="mg/dL glucose", conc_source="liver",
                 color="#00cc88", description="AMPK activation suppresses hepatic gluconeogenesis"),
        PDEffect("Insulin Sensitivity", "muscle", "therapeutic",
                 emax=40, ec50=2.0, hill_n=1.0, baseline=50,
                 ke0=0.3, unit="% sensitivity", conc_source="plasma",
                 color="#44ddff", description="Improved GLUT4 translocation in muscle"),
        PDEffect("GI Tolerance", "stomach", "side_effect",
                 emax=60, ec50=3.0, hill_n=2.0, baseline=0,
                 ke0=1.0, unit="% GI symptoms", conc_source="plasma",
                 color="#ff8800", description="Serotonin release in GI mucosa → nausea/diarrhea"),
        PDEffect("Lactic Acid", "muscle", "toxic",
                 emax=50, ec50=10.0, hill_n=3.0, baseline=1.0,
                 ke0=0.2, unit="mmol/L lactate", conc_source="plasma",
                 color="#ff3333", description="Rare lactic acidosis (1/30,000) — mainly in renal impairment"),
    ]),

    "atorvastatin": DrugPDProfile("atorvastatin", [
        PDEffect("LDL Reduction", "liver", "therapeutic",
                 emax=55, ec50=0.005, hill_n=1.5, baseline=130,
                 ke0=0.02, unit="mg/dL LDL", conc_source="liver",
                 color="#00cc88", description="HMG-CoA reductase inhibition → LDL receptor upregulation (takes 2-4 weeks)"),
        PDEffect("HDL Increase", "liver", "therapeutic",
                 emax=10, ec50=0.01, hill_n=1.0, baseline=45,
                 ke0=0.01, unit="mg/dL HDL", conc_source="liver",
                 color="#44ddff", description="Modest HDL increase (5-10%)"),
        PDEffect("Myalgia Risk", "muscle", "side_effect",
                 emax=30, ec50=0.05, hill_n=2.0, baseline=0,
                 ke0=0.1, unit="% risk", conc_source="muscle",
                 color="#ff8800", description="CoQ10 depletion → mitochondrial dysfunction → muscle pain"),
    ]),
}

# SMILES to drug name mapping
SMILES_TO_NAME = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": "aspirin",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "caffeine",
    "CC(=O)NC1=CC=C(O)C=C1": "paracetamol",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": "ibuprofen",
    "CCO": "ethanol",
}


# ════════════════════════════════════════════════════════════════
# EMAX MODEL CALCULATION
# ════════════════════════════════════════════════════════════════

def emax_effect(conc: float, emax: float, ec50: float, hill_n: float = 1.0) -> float:
    """
    Calculate drug effect using Emax (Hill) equation.

    E = (Emax × C^n) / (EC50^n + C^n)

    Returns effect as fraction of Emax (0 to 1).
    """
    if conc <= 0 or ec50 <= 0:
        return 0.0
    return (emax * (conc ** hill_n)) / ((ec50 ** hill_n) + (conc ** hill_n))


def effect_compartment(plasma_conc: np.ndarray, time_h: np.ndarray, ke0: float) -> np.ndarray:
    """
    Calculate effect compartment concentration using convolution.

    Ce(t) lags behind Cp(t) with rate constant ke0.
    dCe/dt = ke0 × (Cp - Ce)

    Uses simple Euler integration.
    """
    ce = np.zeros_like(plasma_conc)
    for i in range(1, len(time_h)):
        dt = time_h[i] - time_h[i - 1]
        ce[i] = ce[i - 1] + ke0 * (plasma_conc[i] - ce[i - 1]) * dt
    return np.clip(ce, 0, None)


# ════════════════════════════════════════════════════════════════
# MAIN PD SIMULATION
# ════════════════════════════════════════════════════════════════

def simulate_pd_effects(
    smiles: str = "",
    drug_name: str = "",
    time_course: list = None,
    dose_mg: float = 100.0,
) -> dict:
    """
    Simulate pharmacodynamic effects over time.

    Takes the PBPK time course (concentrations in each organ over time)
    and calculates the time course of each pharmacodynamic effect.

    Returns dict with:
    - effects: list of effect profiles with time-course data
    - summary: peak effects and timing
    """
    if not drug_name or len(drug_name) < 3:
        drug_name = SMILES_TO_NAME.get(smiles, "")

    name_lower = drug_name.lower().strip()

    # Find drug PD profile
    pd_profile = None
    for known_name, profile in PD_PROFILES.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            pd_profile = profile
            break

    if not pd_profile or not time_course:
        return {"effects": [], "summary": {}}

    # Extract time and concentration arrays
    time_h = np.array([tp["time_h"] for tp in time_course])
    conc_sources = {
        "plasma": np.array([tp.get("plasma_conc", 0) for tp in time_course]),
        "brain": np.array([tp.get("brain_conc", 0) for tp in time_course]),
        "liver": np.array([tp.get("liver_conc", 0) for tp in time_course]),
        "kidney": np.array([tp.get("kidney_conc", 0) for tp in time_course]),
        "muscle": np.array([tp.get("muscle_conc", 0) for tp in time_course]),
        "gut": np.array([tp.get("gut_conc", 0) for tp in time_course]),
    }

    results = []
    summary = {}

    for eff in pd_profile.effects:
        # Get concentration at the relevant site
        conc_array = conc_sources.get(eff.conc_source, conc_sources["plasma"])

        # Calculate effect compartment concentration (delayed response)
        ce = effect_compartment(conc_array, time_h, eff.ke0)

        # Calculate effect at each time point using Emax model
        effect_values = np.array([emax_effect(c, eff.emax, eff.ec50, eff.hill_n) for c in ce])

        # Apply to baseline
        if eff.effect_type == "therapeutic" and eff.baseline > 0:
            # Therapeutic: reduce from baseline (e.g., pain 100 → 30)
            response = eff.baseline - effect_values
        elif eff.effect_type == "toxic" and "GSH" in eff.name:
            # GSH depletion: decrease from baseline
            response = eff.baseline - effect_values
        else:
            # Side effect / toxic: increase from baseline
            response = eff.baseline + effect_values

        # Find peak effect
        if eff.effect_type == "therapeutic":
            peak_idx = np.argmin(response) if eff.baseline > 0 else np.argmax(response)
        else:
            peak_idx = np.argmax(response) if eff.baseline == 0 else np.argmin(response)

        peak_effect = float(effect_values[peak_idx])
        peak_time = float(time_h[peak_idx])
        peak_response = float(response[peak_idx])

        # Duration of meaningful effect (>20% of peak)
        threshold = peak_effect * 0.2
        above_threshold = effect_values > threshold
        if np.any(above_threshold):
            onset_idx = np.argmax(above_threshold)
            offset_idx = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1
            onset_time = float(time_h[onset_idx])
            offset_time = float(time_h[offset_idx])
            duration = offset_time - onset_time
        else:
            onset_time = 0
            offset_time = 0
            duration = 0

        effect_result = {
            "name": eff.name,
            "organ": eff.organ,
            "effect_type": eff.effect_type,
            "unit": eff.unit,
            "color": eff.color,
            "description": eff.description,
            "baseline": eff.baseline,
            "ec50": eff.ec50,
            "emax": eff.emax,

            # Time course (sampled — every 5th point for API efficiency)
            "time_points": [float(x) for x in time_h[::5]],
            "effect_curve": [float(x) for x in effect_values[::5]],
            "response_curve": [float(x) for x in response[::5]],
            "effect_site_conc": [float(x) for x in ce[::5]],

            # Summary stats
            "peak_effect": round(peak_effect, 1),
            "peak_effect_pct": round(peak_effect / eff.emax * 100, 0) if eff.emax > 0 else 0,
            "peak_time_h": round(peak_time, 2),
            "peak_response": round(peak_response, 1),
            "onset_time_h": round(onset_time, 2),
            "offset_time_h": round(offset_time, 2),
            "duration_h": round(duration, 1),
        }

        results.append(effect_result)

        summary[eff.name] = {
            "peak": round(peak_effect, 1),
            "peak_time": round(peak_time, 2),
            "duration": round(duration, 1),
            "type": eff.effect_type,
        }

    return {
        "drug_name": pd_profile.drug_name,
        "n_effects": len(results),
        "effects": results,
        "summary": summary,
    }


# ════════════════════════════════════════════════════════════════
# MULTI-ORGAN CROSSTALK
# ════════════════════════════════════════════════════════════════

@dataclass
class CrosstalkEffect:
    """How dysfunction in one organ affects another."""
    source_organ: str           # The organ that's dysfunctional
    target_organ: str           # The organ that's affected
    source_condition: str       # What triggers this (e.g., "GFR < 30")
    effect_on_target: str       # What changes (e.g., "CYP3A4 activity")
    modifier_key: str           # Which PBPK parameter to modify
    modifier_value: float       # Multiplier to apply
    mechanism: str              # Explanation
    visual: str                 # What to show on 3D body


CROSSTALK_RULES = [
    # ── Kidney → Liver ──
    CrosstalkEffect("kidney", "liver", "eGFR < 30",
                    "CYP3A4 reduced 30-50%", "hepatic_cl_scale", 0.6,
                    "Uremic toxins (indoxyl sulfate, p-cresol) inhibit hepatic CYP3A4 and CYP2C9",
                    "Dark particles flow from kidney to liver, liver dims slightly"),
    CrosstalkEffect("kidney", "liver", "eGFR < 15",
                    "CYP3A4 reduced 60%", "hepatic_cl_scale", 0.4,
                    "Severe uremia → major CYP inhibition → drug accumulation risk",
                    "Heavy dark particle flow kidney → liver, liver turns orange"),

    # ── Liver → Kidney ──
    CrosstalkEffect("liver", "kidney", "ALT > 200",
                    "Hepatorenal syndrome risk", "renal_cl_scale", 0.7,
                    "Severe liver disease → splanchnic vasodilation → renal vasoconstriction → reduced GFR",
                    "Liver glows red, kidney dims, connecting vessel constricts"),

    # ── Liver → Blood (albumin) ──
    CrosstalkEffect("liver", "blood", "Albumin < 2.5",
                    "Free drug fraction doubles", "fu_modifier", 2.0,
                    "Cirrhotic liver can't produce albumin → less protein binding → more free (active) drug",
                    "Blood particles increase in brightness (more free drug)"),

    # ── Heart → All Organs ──
    CrosstalkEffect("heart", "liver", "NYHA III-IV",
                    "Hepatic blood flow reduced 40%", "Q_liver_scale", 0.6,
                    "Low cardiac output → reduced hepatic perfusion → drug accumulates in liver",
                    "Heart beats slowly, flow to liver diminishes"),
    CrosstalkEffect("heart", "kidney", "NYHA III-IV",
                    "Renal blood flow reduced 30%", "Q_kidney_scale", 0.7,
                    "Cardiorenal syndrome: low CO → renal hypoperfusion → AKI",
                    "Heart beats slowly, kidney flow diminishes"),
    CrosstalkEffect("heart", "brain", "NYHA III-IV",
                    "Cerebral perfusion reduced 20%", "Q_brain_scale", 0.8,
                    "Low CO → reduced cerebral blood flow → confusion/drowsiness",
                    "Heart slow, brain dims slightly"),

    # ── Inflammation → Liver ──
    CrosstalkEffect("immune", "liver", "IL-6 elevated",
                    "CYP1A2/2C19/3A4 reduced 30-70%", "hepatic_cl_scale", 0.5,
                    "IL-6 and TNF-α suppress CYP transcription via NF-κB pathway",
                    "Inflammatory particles around liver, liver enzyme activity drops"),

    # ── Gut → Liver (portal hypertension) ──
    CrosstalkEffect("gut", "liver", "Portal HTN",
                    "First-pass metabolism reduced", "first_pass_scale", 0.5,
                    "Portal hypertension → portosystemic shunting → drug bypasses liver → increased bioavailability",
                    "Drug particles bypass liver, go directly to systemic circulation"),

    # ── Thyroid → Liver ──
    CrosstalkEffect("thyroid", "liver", "Hypothyroid",
                    "CYP metabolism reduced 20%", "hepatic_cl_scale", 0.8,
                    "Low T3/T4 → reduced CYP expression → slower drug metabolism",
                    "Liver dims, metabolism slows"),
    CrosstalkEffect("thyroid", "liver", "Hyperthyroid",
                    "CYP metabolism increased 30%", "hepatic_cl_scale", 1.3,
                    "High T3/T4 → increased CYP expression → faster drug metabolism",
                    "Liver brightens, metabolism accelerates"),

    # ── Kidney → Brain ──
    CrosstalkEffect("kidney", "brain", "eGFR < 15",
                    "Uremic encephalopathy → BBB disruption", "Kp_brain_modifier", 1.5,
                    "Uremic toxins damage BBB tight junctions → drugs enter brain more easily → CNS toxicity",
                    "Dark particles cross BBB, brain shows warning glow"),

    # ── Liver → Brain ──
    CrosstalkEffect("liver", "brain", "Ammonia elevated",
                    "Hepatic encephalopathy → BBB disruption", "Kp_brain_modifier", 1.8,
                    "Liver can't clear ammonia → crosses BBB → astrocyte swelling → cerebral edema",
                    "Toxic particles flow liver → brain, brain swells and glows"),

    # ── Obesity → Multiple ──
    CrosstalkEffect("fat", "liver", "BMI > 35",
                    "CYP2E1 induced, CYP3A4 variable", "cyp2e1_scale", 1.5,
                    "Fatty liver induces CYP2E1 → more NAPQI from paracetamol → increased liver toxicity risk",
                    "Fat compartment large, liver shows increased activity"),
]


def evaluate_crosstalk(
    diseases: list = None,
    clinical_labs: dict = None,
    patient_age: int = 30,
    patient_weight: float = 70,
) -> dict:
    """
    Evaluate multi-organ crosstalk based on active diseases and clinical labs.

    Returns:
    - active_crosstalks: list of triggered crosstalk effects
    - modifiers: dict of PBPK parameter modifications
    - visualizations: list of visual effects for 3D body
    """
    active = []
    modifiers = {}
    visuals = []

    if not diseases and not clinical_labs:
        return {"active_crosstalks": [], "modifiers": {}, "visualizations": []}

    labs = clinical_labs or {}
    disease_set = set(d.lower() for d in (diseases or []))

    for rule in CROSSTALK_RULES:
        triggered = False

        # Check condition
        cond = rule.source_condition.lower()

        if "egfr" in cond:
            egfr = labs.get("egfr", 100)
            threshold = float(''.join(c for c in cond if c.isdigit() or c == '.'))
            if "<" in cond and egfr < threshold:
                triggered = True
        elif "alt" in cond:
            alt = labs.get("alt", 25)
            threshold = float(''.join(c for c in cond if c.isdigit() or c == '.'))
            if ">" in cond and alt > threshold:
                triggered = True
        elif "albumin" in cond:
            alb = labs.get("albumin", 4.0)
            threshold = float(''.join(c for c in cond if c.isdigit() or c == '.'))
            if "<" in cond and alb < threshold:
                triggered = True
        elif "nyha" in cond:
            if any("heart_iii" in d or "heart_iv" in d for d in disease_set):
                triggered = True
        elif "il-6" in cond or "il6" in cond:
            if any("sepsis" in d or "ra_flare" in d or "severe_inflammation" in d for d in disease_set):
                triggered = True
        elif "portal" in cond:
            if any("liver_c" in d or "liver_b" in d for d in disease_set):
                triggered = True
        elif "hypothyroid" in cond:
            if any("hypothyroid" in d for d in disease_set):
                triggered = True
        elif "hyperthyroid" in cond:
            if any("hyperthyroid" in d for d in disease_set):
                triggered = True
        elif "ammonia" in cond:
            if any("liver_c" in d for d in disease_set):
                triggered = True
        elif "bmi" in cond:
            bmi = patient_weight / ((170 / 100) ** 2)  # Rough estimate
            threshold = float(''.join(c for c in cond if c.isdigit() or c == '.'))
            if ">" in cond and bmi > threshold:
                triggered = True

        if triggered:
            active.append({
                "source": rule.source_organ,
                "target": rule.target_organ,
                "condition": rule.source_condition,
                "effect": rule.effect_on_target,
                "mechanism": rule.mechanism,
            })
            modifiers[rule.modifier_key] = modifiers.get(rule.modifier_key, 1.0) * rule.modifier_value
            visuals.append({
                "source": rule.source_organ,
                "target": rule.target_organ,
                "description": rule.visual,
                "modifier_key": rule.modifier_key,
                "modifier_value": rule.modifier_value,
            })

    return {
        "active_crosstalks": active,
        "modifiers": modifiers,
        "visualizations": visuals,
        "n_active": len(active),
    }
