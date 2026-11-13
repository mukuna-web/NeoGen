"""
Neurotransmitter Systems Model
================================
Models the 5 major neurotransmitter systems in the brain and how drugs affect them:

1. DOPAMINE (DA) — reward, motivation, motor control
   - Pathways: mesolimbic (reward), nigrostriatal (motor), mesocortical (cognition), tuberoinfundibular (prolactin)
   - Drugs: L-DOPA (↑DA), antipsychotics (block D2), cocaine (block reuptake), amphetamine (reverse transport)

2. SEROTONIN (5-HT) — mood, sleep, appetite, pain
   - Pathways: raphe nuclei → widespread cortical projection
   - Drugs: SSRIs (block reuptake), triptans (5-HT1B/1D agonist), ondansetron (block 5-HT3)

3. NOREPINEPHRINE (NE) — alertness, fight-or-flight, attention
   - Pathways: locus coeruleus → cortex, hippocampus
   - Drugs: SNRIs (block reuptake), clonidine (α2 agonist), prazosin (α1 blocker)

4. GABA (γ-aminobutyric acid) — primary INHIBITORY neurotransmitter
   - Everywhere in CNS, provides "brakes" on neural activity
   - Drugs: benzodiazepines (enhance GABA-A), barbiturates, gabapentin, ethanol

5. GLUTAMATE (Glu) — primary EXCITATORY neurotransmitter
   - NMDA, AMPA, kainate receptors
   - Drugs: ketamine (NMDA blocker), memantine (Alzheimer's), lamotrigine (reduces release)

Each system has:
- Baseline firing rate
- Drug effects modeled as % change in synaptic neurotransmitter concentration
- Time-dependent effects (onset, peak, duration)
- Clinical manifestations (what the patient experiences)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NeurotransmitterState:
    """Current state of a neurotransmitter system."""
    name: str
    abbreviation: str
    baseline_level: float = 100.0  # % of normal (100 = baseline)
    current_level: float = 100.0   # Current % (modified by drugs)
    receptor_sensitivity: float = 100.0  # % of normal receptor response
    reuptake_rate: float = 100.0   # % of normal reuptake
    synthesis_rate: float = 100.0  # % of normal synthesis
    degradation_rate: float = 100.0  # MAO/COMT activity


@dataclass
class NTDrugEffect:
    """How a drug affects a specific neurotransmitter system."""
    nt_system: str                    # "DA", "5HT", "NE", "GABA", "GLU"
    mechanism: str                    # Description of mechanism
    reuptake_inhibition: float = 0    # % inhibition of reuptake transporter (0-100)
    receptor_agonism: float = 0       # % receptor activation (0-100)
    receptor_antagonism: float = 0    # % receptor blockade (0-100)
    release_enhancement: float = 0    # % increase in release (0-100)
    release_inhibition: float = 0     # % decrease in release (0-100)
    enzyme_inhibition: float = 0      # % MAO/COMT inhibition (0-100)
    onset_h: float = 0.5             # Hours to onset of CNS effect
    peak_effect_h: float = 2.0       # Hours to peak effect
    duration_h: float = 6.0          # Duration of effect


@dataclass
class NTDrugProfile:
    """Complete neurotransmitter profile for a drug."""
    name: str
    drug_class: str
    effects: list[NTDrugEffect] = field(default_factory=list)
    clinical_onset: str = ""
    clinical_peak: str = ""
    clinical_notes: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
# DRUG NEUROTRANSMITTER DATABASE
# ════════════════════════════════════════════════════════════════

NT_DRUG_PROFILES = {
    "caffeine": NTDrugProfile(
        name="Caffeine",
        drug_class="Adenosine antagonist / Indirect catecholamine enhancer",
        effects=[
            NTDrugEffect("DA", "Blocks adenosine A2A → disinhibits dopamine release",
                         release_enhancement=20, onset_h=0.25, peak_effect_h=1.0, duration_h=5),
            NTDrugEffect("NE", "Blocks adenosine → increases norepinephrine release",
                         release_enhancement=30, onset_h=0.25, peak_effect_h=0.75, duration_h=4),
            NTDrugEffect("5HT", "Minimal direct serotonin effect",
                         release_enhancement=5, onset_h=0.5, peak_effect_h=1.5, duration_h=4),
            NTDrugEffect("GABA", "Reduces GABA-ergic inhibition via adenosine blockade",
                         receptor_antagonism=10, onset_h=0.25, peak_effect_h=1.0, duration_h=4),
            NTDrugEffect("GLU", "Enhances glutamate transmission (excitatory)",
                         release_enhancement=15, onset_h=0.25, peak_effect_h=1.0, duration_h=4),
        ],
        clinical_onset="15-45 min",
        clinical_peak="30-90 min",
        clinical_notes=[
            "Alertness and focus increase (NE + DA)",
            "Anxiety possible at high doses (NE excess)",
            "Seizure threshold lowered at very high doses (GLU > GABA)",
            "Tolerance develops to alerting effects over 1-2 weeks (receptor downregulation)",
        ],
    ),

    "aspirin": NTDrugProfile(
        name="Aspirin",
        drug_class="COX inhibitor / Minimal CNS effects",
        effects=[
            NTDrugEffect("5HT", "COX inhibition reduces prostaglandin-mediated 5-HT sensitization",
                         release_inhibition=5, onset_h=1.0, peak_effect_h=3.0, duration_h=6),
        ],
        clinical_onset="30-60 min (peripheral analgesic)",
        clinical_peak="1-3 h",
        clinical_notes=[
            "Minimal direct CNS neurotransmitter effects",
            "Analgesic effect is primarily peripheral (COX-1/2 inhibition)",
            "Central COX-2 inhibition may contribute to antipyretic effect",
        ],
    ),

    "paracetamol": NTDrugProfile(
        name="Paracetamol",
        drug_class="COX-3 inhibitor / Serotonergic analgesic",
        effects=[
            NTDrugEffect("5HT", "Active metabolite AM404 enhances serotonergic descending pain inhibition",
                         release_enhancement=15, onset_h=0.5, peak_effect_h=1.5, duration_h=5),
            NTDrugEffect("DA", "Minimal dopaminergic effect",
                         release_enhancement=5, onset_h=1.0, peak_effect_h=2.0, duration_h=4),
        ],
        clinical_onset="15-30 min (analgesic)",
        clinical_peak="1-2 h",
        clinical_notes=[
            "Central analgesic mechanism involves serotonergic pathways",
            "AM404 metabolite activates TRPV1 and inhibits anandamide reuptake",
            "Weak anti-inflammatory effect (poor COX inhibition in inflamed tissue)",
        ],
    ),

    "ibuprofen": NTDrugProfile(
        name="Ibuprofen",
        drug_class="NSAID / COX-1/2 inhibitor",
        effects=[
            NTDrugEffect("5HT", "Reduces prostaglandin-mediated serotonin sensitization in CNS",
                         release_inhibition=8, onset_h=0.5, peak_effect_h=2.0, duration_h=6),
        ],
        clinical_onset="20-30 min",
        clinical_peak="1-2 h",
        clinical_notes=[
            "Primary analgesic effect is peripheral (COX-1/2 inhibition)",
            "CNS penetration moderate — contributes to antipyretic effect",
            "May reduce Alzheimer's risk long-term (anti-neuroinflammatory)",
        ],
    ),

    "diazepam": NTDrugProfile(
        name="Diazepam (Valium)",
        drug_class="Benzodiazepine / GABA-A positive allosteric modulator",
        effects=[
            NTDrugEffect("GABA", "Binds GABA-A receptor → increases Cl⁻ channel opening frequency → enhanced inhibition",
                         receptor_agonism=80, onset_h=0.25, peak_effect_h=1.0, duration_h=24),
            NTDrugEffect("DA", "Indirect: GABA inhibits GABA interneurons in VTA → disinhibits DA (euphoria)",
                         release_enhancement=15, onset_h=0.5, peak_effect_h=1.5, duration_h=12),
            NTDrugEffect("NE", "GABA-mediated reduction of locus coeruleus firing → reduced anxiety",
                         release_inhibition=40, onset_h=0.25, peak_effect_h=1.0, duration_h=20),
            NTDrugEffect("GLU", "GABA enhancement reduces glutamatergic excitation",
                         release_inhibition=30, onset_h=0.25, peak_effect_h=1.0, duration_h=18),
        ],
        clinical_onset="15-30 min (oral), 1-3 min (IV)",
        clinical_peak="30-90 min",
        clinical_notes=[
            "Anxiolytic: NE reduction + GABA enhancement",
            "Sedation: widespread cortical GABA enhancement",
            "Muscle relaxation: spinal GABA enhancement",
            "Anticonvulsant: raises seizure threshold (GLU/GABA balance)",
            "Respiratory depression risk at high doses (brainstem GABA)",
            "Additive with alcohol and opioids — DANGEROUS combination",
        ],
    ),

    "fluoxetine": NTDrugProfile(
        name="Fluoxetine (Prozac)",
        drug_class="SSRI — Selective Serotonin Reuptake Inhibitor",
        effects=[
            NTDrugEffect("5HT", "Blocks SERT (serotonin transporter) → 5-HT accumulates in synapse",
                         reuptake_inhibition=90, onset_h=2.0, peak_effect_h=168.0, duration_h=720),
            NTDrugEffect("NE", "Mild NE reuptake inhibition at high doses",
                         reuptake_inhibition=10, onset_h=4.0, peak_effect_h=168.0, duration_h=720),
            NTDrugEffect("DA", "Minimal dopamine effect (unlike bupropion)",
                         reuptake_inhibition=5, onset_h=4.0, peak_effect_h=168.0, duration_h=720),
        ],
        clinical_onset="2-4 weeks for antidepressant effect (receptor downregulation needed)",
        clinical_peak="4-6 weeks",
        clinical_notes=[
            "Acute: synaptic 5-HT rises → initial anxiety, nausea (5-HT3/4 activation)",
            "Week 1-2: 5-HT1A autoreceptors desensitize → more 5-HT release",
            "Week 2-4: postsynaptic 5-HT receptor downregulation → antidepressant effect",
            "This delayed mechanism is why SSRIs take weeks to work",
            "CYP2D6 potent inhibitor — major DDI risk",
            "Discontinuation syndrome if stopped abruptly (short t½ metabolite)",
        ],
    ),

    "ethanol": NTDrugProfile(
        name="Ethanol (Alcohol)",
        drug_class="CNS depressant / GABA enhancer / Glutamate inhibitor",
        effects=[
            NTDrugEffect("GABA", "Enhances GABA-A receptor function → sedation, anxiolysis",
                         receptor_agonism=60, onset_h=0.15, peak_effect_h=0.5, duration_h=4),
            NTDrugEffect("GLU", "Inhibits NMDA glutamate receptors → impaired memory, coordination",
                         receptor_antagonism=50, onset_h=0.15, peak_effect_h=0.5, duration_h=4),
            NTDrugEffect("DA", "Increases DA release in nucleus accumbens → euphoria, reward",
                         release_enhancement=40, onset_h=0.15, peak_effect_h=0.75, duration_h=3),
            NTDrugEffect("5HT", "Enhances 5-HT3 receptor → nausea at high doses",
                         release_enhancement=20, onset_h=0.25, peak_effect_h=1.0, duration_h=3),
            NTDrugEffect("NE", "Biphasic: initial NE release (stimulant phase) then suppression",
                         release_enhancement=25, release_inhibition=10, onset_h=0.15, peak_effect_h=0.5, duration_h=4),
        ],
        clinical_onset="5-15 min",
        clinical_peak="30-90 min",
        clinical_notes=[
            "Low dose (BAC 0.02-0.05): relaxation, disinhibition (GABA + DA)",
            "Moderate (0.05-0.10): impaired judgment, coordination (NMDA block)",
            "High (0.10-0.20): slurred speech, ataxia (profound GABA/GLU imbalance)",
            "Dangerous (0.25-0.40): respiratory depression, coma, death",
            "Chronic use: GABA receptor downregulation + NMDA upregulation → withdrawal seizures",
            "CYP2E1 inducer — increases paracetamol toxicity",
        ],
    ),

    "metformin": NTDrugProfile(
        name="Metformin",
        drug_class="Biguanide / No significant CNS effects",
        effects=[],
        clinical_onset="N/A — no CNS effect",
        clinical_peak="N/A",
        clinical_notes=[
            "Metformin has minimal BBB penetration",
            "No significant neurotransmitter effects at therapeutic doses",
            "Emerging research suggests possible neuroprotective effects via AMPK activation",
        ],
    ),
}


# ════════════════════════════════════════════════════════════════
# NEUROTRANSMITTER SIMULATION
# ════════════════════════════════════════════════════════════════

def simulate_neurotransmitter_effects(
    drug_name: str,
    brain_concentration: float = 1.0,  # mg/L
    ec50: float = 5.0,                 # mg/L at which 50% effect occurs
    time_h: float = 1.0,               # Current time post-dose
) -> dict:
    """
    Simulate drug effects on all 5 neurotransmitter systems.

    Returns per-NT-system changes with clinical correlations.
    """
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    profile = NT_DRUG_PROFILES.get(key)

    if not profile:
        return {
            "drug": drug_name,
            "nt_effects": {},
            "clinical_summary": "No neurotransmitter profile available for this drug",
            "overall_cns_effect": "unknown",
        }

    # Concentration-effect relationship (sigmoid Emax)
    effect_fraction = brain_concentration ** 1.5 / (ec50 ** 1.5 + brain_concentration ** 1.5)

    nt_effects = {}
    for effect in profile.effects:
        # Time-dependent effect (onset → peak → decline)
        if time_h < effect.onset_h:
            time_factor = 0
        elif time_h < effect.peak_effect_h:
            time_factor = (time_h - effect.onset_h) / (effect.peak_effect_h - effect.onset_h)
        elif time_h < effect.duration_h:
            time_factor = 1.0 - (time_h - effect.peak_effect_h) / (effect.duration_h - effect.peak_effect_h)
        else:
            time_factor = 0

        time_factor = max(0, min(1.0, time_factor))
        scaled_effect = effect_fraction * time_factor

        # Calculate net change in synaptic neurotransmitter level
        synaptic_change = 0
        if effect.reuptake_inhibition > 0:
            synaptic_change += effect.reuptake_inhibition * scaled_effect
        if effect.release_enhancement > 0:
            synaptic_change += effect.release_enhancement * scaled_effect
        if effect.release_inhibition > 0:
            synaptic_change -= effect.release_inhibition * scaled_effect
        if effect.enzyme_inhibition > 0:
            synaptic_change += effect.enzyme_inhibition * scaled_effect * 0.5

        # Receptor effects
        receptor_activation = 0
        if effect.receptor_agonism > 0:
            receptor_activation = effect.receptor_agonism * scaled_effect
        receptor_blockade = 0
        if effect.receptor_antagonism > 0:
            receptor_blockade = effect.receptor_antagonism * scaled_effect

        nt_effects[effect.nt_system] = {
            "system": _nt_full_name(effect.nt_system),
            "mechanism": effect.mechanism,
            "synaptic_level_change": round(synaptic_change, 1),  # % above/below baseline
            "receptor_activation": round(receptor_activation, 1),
            "receptor_blockade": round(receptor_blockade, 1),
            "net_effect": round(synaptic_change + receptor_activation - receptor_blockade, 1),
            "time_factor": round(time_factor, 2),
            "status": (
                "ENHANCED" if synaptic_change + receptor_activation > 10
                else "SUPPRESSED" if receptor_blockade > 10 or synaptic_change < -10
                else "MINIMAL"
            ),
        }

    # Determine overall CNS effect
    total_excitatory = sum(
        e.get("net_effect", 0) for k, e in nt_effects.items()
        if k in ("DA", "NE", "GLU") and e.get("net_effect", 0) > 0
    )
    total_inhibitory = sum(
        abs(e.get("net_effect", 0)) for k, e in nt_effects.items()
        if k == "GABA" and e.get("net_effect", 0) > 0
    ) + sum(
        abs(e.get("net_effect", 0)) for k, e in nt_effects.items()
        if k == "GLU" and e.get("net_effect", 0) < 0
    )

    if total_excitatory > total_inhibitory + 20:
        overall = "CNS STIMULATION"
    elif total_inhibitory > total_excitatory + 20:
        overall = "CNS DEPRESSION"
    elif total_excitatory > 5 and total_inhibitory > 5:
        overall = "MIXED CNS EFFECTS"
    else:
        overall = "MINIMAL CNS EFFECT"

    # Clinical manifestations
    manifestations = _determine_clinical_manifestations(nt_effects)

    return {
        "drug": profile.name,
        "drug_class": profile.drug_class,
        "nt_effects": nt_effects,
        "overall_cns_effect": overall,
        "clinical_onset": profile.clinical_onset,
        "clinical_peak": profile.clinical_peak,
        "clinical_manifestations": manifestations,
        "clinical_notes": profile.clinical_notes,
        "effect_fraction": round(effect_fraction, 3),
        "time_h": time_h,
    }


def _nt_full_name(abbrev: str) -> str:
    return {
        "DA": "Dopamine",
        "5HT": "Serotonin (5-HT)",
        "NE": "Norepinephrine",
        "GABA": "GABA",
        "GLU": "Glutamate",
    }.get(abbrev, abbrev)


def _determine_clinical_manifestations(nt_effects: dict) -> list[dict]:
    """Map neurotransmitter changes to clinical symptoms."""
    manifestations = []

    da = nt_effects.get("DA", {}).get("net_effect", 0)
    ht = nt_effects.get("5HT", {}).get("net_effect", 0)
    ne = nt_effects.get("NE", {}).get("net_effect", 0)
    gaba = nt_effects.get("GABA", {}).get("net_effect", 0)
    glu = nt_effects.get("GLU", {}).get("net_effect", 0)

    # Dopamine effects
    if da > 30:
        manifestations.append({"symptom": "Euphoria / Reward", "severity": "moderate", "system": "DA", "mechanism": "↑ DA in nucleus accumbens"})
    if da > 60:
        manifestations.append({"symptom": "Psychomotor agitation", "severity": "severe", "system": "DA", "mechanism": "Excessive DA in striatum"})
    if da < -30:
        manifestations.append({"symptom": "Anhedonia / Flat affect", "severity": "moderate", "system": "DA", "mechanism": "↓ DA in mesolimbic pathway"})

    # Serotonin effects
    if ht > 20:
        manifestations.append({"symptom": "Improved mood / Anxiolysis", "severity": "mild", "system": "5-HT", "mechanism": "↑ 5-HT in prefrontal cortex"})
    if ht > 50:
        manifestations.append({"symptom": "Nausea", "severity": "moderate", "system": "5-HT", "mechanism": "5-HT3 activation in CTZ"})
    if ht > 80:
        manifestations.append({"symptom": "Serotonin syndrome risk", "severity": "severe", "system": "5-HT", "mechanism": "Excessive synaptic 5-HT"})

    # Norepinephrine effects
    if ne > 20:
        manifestations.append({"symptom": "Increased alertness", "severity": "mild", "system": "NE", "mechanism": "↑ NE in locus coeruleus → cortex"})
    if ne > 40:
        manifestations.append({"symptom": "Anxiety / Tremor", "severity": "moderate", "system": "NE", "mechanism": "Excessive sympathetic activation"})
    if ne < -20:
        manifestations.append({"symptom": "Sedation / Drowsiness", "severity": "mild", "system": "NE", "mechanism": "↓ LC firing → reduced arousal"})

    # GABA effects
    if gaba > 30:
        manifestations.append({"symptom": "Sedation / Relaxation", "severity": "mild", "system": "GABA", "mechanism": "Enhanced cortical GABA-A"})
    if gaba > 60:
        manifestations.append({"symptom": "Ataxia / Slurred speech", "severity": "moderate", "system": "GABA", "mechanism": "Cerebellar GABA enhancement"})
    if gaba > 80:
        manifestations.append({"symptom": "Respiratory depression risk", "severity": "severe", "system": "GABA", "mechanism": "Brainstem GABA suppresses respiratory drive"})

    # Glutamate effects
    if glu < -20:
        manifestations.append({"symptom": "Impaired memory formation", "severity": "mild", "system": "GLU", "mechanism": "NMDA blockade → reduced LTP"})
    if glu < -40:
        manifestations.append({"symptom": "Dissociation / Confusion", "severity": "moderate", "system": "GLU", "mechanism": "Widespread NMDA antagonism"})
    if glu > 30:
        manifestations.append({"symptom": "Lowered seizure threshold", "severity": "moderate", "system": "GLU", "mechanism": "↑ excitatory glutamate transmission"})

    # Combined effects
    if ne > 20 and da > 10:
        manifestations.append({"symptom": "Psychostimulant effect", "severity": "moderate", "system": "DA+NE", "mechanism": "Combined catecholamine enhancement"})
    if gaba > 40 and glu < -20:
        manifestations.append({"symptom": "Profound CNS depression", "severity": "severe", "system": "GABA+GLU", "mechanism": "Enhanced inhibition + reduced excitation"})

    return manifestations


def get_nt_profile(drug_name: str) -> NTDrugProfile | None:
    """Look up a drug's neurotransmitter profile."""
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    return NT_DRUG_PROFILES.get(key)
