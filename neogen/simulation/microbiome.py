"""
Gut Microbiome Metabolism Model
================================
Models how gut bacteria metabolize drugs before/after absorption.

The human gut contains ~100 trillion bacteria with their own enzymes.
These bacterial enzymes can:
1. ACTIVATE prodrugs (sulfasalazine → 5-ASA + sulfapyridine)
2. INACTIVATE drugs (digoxin inactivated by Eggerthella lenta)
3. DECONJUGATE Phase II metabolites (β-glucuronidase → enterohepatic recycling)
4. PRODUCE toxic metabolites
5. Compete with host enzymes for drug metabolism

Key bacterial enzymes:
- Azoreductases: cleave azo bonds (sulfasalazine, prontosil)
- Nitroreductases: reduce nitro groups (metronidazole activation)
- β-glucuronidases: deconjugate glucuronides (reactivate drugs excreted in bile)
- β-glycosidases: cleave sugar moieties
- Esterases: hydrolyze esters
- Cytochrome-like reductases: various reductions

Factors affecting microbiome drug metabolism:
- Antibiotics: devastate microbiome → altered drug metabolism for weeks
- Diet: fiber → ↑ Firmicutes, Western diet → ↑ Bacteroidetes
- Age: neonates have different flora than adults
- Disease: IBD, cirrhosis alter microbiome
- Probiotics: can modestly modify microbiome
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MicrobiomeDrugEffect:
    """How gut microbiome affects a specific drug."""
    drug_name: str
    drug_class: str
    bacterial_action: str            # What bacteria do to the drug
    bacterial_enzyme: str            # Which enzyme
    key_species: list[str] = field(default_factory=list)
    effect_type: str = ""            # "activation", "inactivation", "reactivation", "toxification"
    magnitude: str = ""              # "major", "moderate", "minor"
    metabolite: str = ""             # Bacterial metabolite
    clinical_impact: str = ""
    affected_by_antibiotics: bool = True
    interpatient_variability: str = "moderate"


MICROBIOME_DRUG_DATA = {
    "digoxin": MicrobiomeDrugEffect(
        drug_name="Digoxin", drug_class="Cardiac glycoside",
        bacterial_action="Eggerthella lenta reduces the lactone ring → inactive dihydrodigoxin",
        bacterial_enzyme="Cardiac glycoside reductase (cgr operon)",
        key_species=["Eggerthella lenta"],
        effect_type="inactivation", magnitude="major",
        metabolite="Dihydrodigoxin (inactive)",
        clinical_impact="10-40% of patients have high E. lenta → need higher digoxin doses. Antibiotics (erythromycin) kill E. lenta → digoxin levels SPIKE → toxicity risk.",
        interpatient_variability="very high (10-fold)",
    ),
    "sulfasalazine": MicrobiomeDrugEffect(
        drug_name="Sulfasalazine", drug_class="DMARD / IBD drug",
        bacterial_action="Bacterial azoreductases cleave the azo bond → releases 5-ASA (active anti-inflammatory) + sulfapyridine",
        bacterial_enzyme="Azoreductase",
        key_species=["Bacteroides", "Clostridium", "E. coli"],
        effect_type="activation", magnitude="major",
        metabolite="5-aminosalicylic acid (5-ASA) — the active drug",
        clinical_impact="Drug is COMPLETELY dependent on bacterial activation. Without gut bacteria, sulfasalazine is inactive. Antibiotics will reduce efficacy.",
        interpatient_variability="moderate",
    ),
    "metronidazole": MicrobiomeDrugEffect(
        drug_name="Metronidazole", drug_class="Antibiotic / Antiprotozoal",
        bacterial_action="Bacterial nitroreductases reduce the nitro group → cytotoxic intermediates that damage bacterial DNA",
        bacterial_enzyme="Nitroreductase (in anaerobic bacteria)",
        key_species=["Anaerobic bacteria (target organisms)"],
        effect_type="activation", magnitude="major",
        metabolite="Nitro radical anion → DNA strand breaks",
        clinical_impact="Only works against ANAEROBIC bacteria because activation requires low-redox environment. Aerobic bacteria can't activate it → are resistant.",
        interpatient_variability="low",
    ),
    "levodopa": MicrobiomeDrugEffect(
        drug_name="Levodopa (L-DOPA)", drug_class="Dopamine precursor (Parkinson's)",
        bacterial_action="Gut bacterial tyrosine decarboxylases convert L-DOPA → dopamine in the gut BEFORE absorption → reduces bioavailability",
        bacterial_enzyme="Tyrosine decarboxylase (TyrDC)",
        key_species=["Enterococcus faecalis"],
        effect_type="inactivation", magnitude="major",
        metabolite="Dopamine (in gut — can't cross BBB from periphery)",
        clinical_impact="Explains why some Parkinson's patients need much higher L-DOPA doses. E. faecalis abundance correlates with L-DOPA dose requirements. Carbidopa partially compensates.",
        interpatient_variability="very high",
    ),
    "irinotecan": MicrobiomeDrugEffect(
        drug_name="Irinotecan (CPT-11)", drug_class="Chemotherapy (topoisomerase inhibitor)",
        bacterial_action="Bacterial β-glucuronidase deconjugates SN-38-glucuronide (inactive, excreted in bile) → SN-38 (active/toxic) → severe diarrhea",
        bacterial_enzyme="β-glucuronidase",
        key_species=["E. coli", "Clostridium", "Bacteroides"],
        effect_type="toxification", magnitude="major",
        metabolite="SN-38 (reactivated in colon → causes severe diarrhea in 60-80% of patients)",
        clinical_impact="MAJOR toxicity issue. Bacterial β-glucuronidase inhibitors (like ciprofloxacin) can reduce diarrhea. Targeted β-glucuronidase inhibitors in development.",
        interpatient_variability="high",
    ),
    "morphine": MicrobiomeDrugEffect(
        drug_name="Morphine", drug_class="Opioid analgesic",
        bacterial_action="Bacterial β-glucuronidase deconjugates morphine-3-glucuronide and morphine-6-glucuronide → reactivates morphine via enterohepatic recycling",
        bacterial_enzyme="β-glucuronidase",
        key_species=["E. coli", "Bacteroides"],
        effect_type="reactivation", magnitude="moderate",
        metabolite="Morphine (reactivated from glucuronide conjugates)",
        clinical_impact="Enterohepatic recycling prolongs morphine's effect. Antibiotics can reduce recycling → shorter morphine duration.",
        interpatient_variability="moderate",
    ),
    "lovastatin": MicrobiomeDrugEffect(
        drug_name="Lovastatin", drug_class="Statin (HMG-CoA reductase inhibitor)",
        bacterial_action="Gut bacteria hydrolyze the lactone prodrug → active hydroxy acid form. Also produce secondary bile acids that affect cholesterol absorption.",
        bacterial_enzyme="Esterases + bile salt hydrolases",
        key_species=["Lactobacillus", "Bifidobacterium", "Bacteroides"],
        effect_type="activation", magnitude="moderate",
        metabolite="Lovastatin acid (active form)",
        clinical_impact="Gut bacteria contribute to statin activation. Probiotics containing bile salt hydrolases may enhance statin efficacy.",
        interpatient_variability="moderate",
    ),
    "acetaminophen": MicrobiomeDrugEffect(
        drug_name="Acetaminophen (Paracetamol)", drug_class="Analgesic / Antipyretic",
        bacterial_action="Gut bacterial sulfotransferases produce p-cresol → competes with paracetamol for sulfation pathway in liver → more paracetamol shunted to CYP2E1 → more NAPQI",
        bacterial_enzyme="Arylsulfate sulfotransferase",
        key_species=["Clostridium difficile", "various anaerobes"],
        effect_type="toxification", magnitude="moderate",
        metabolite="p-cresol (bacterial) competes with paracetamol sulfation",
        clinical_impact="Higher p-cresol producers (specific microbiome composition) have MORE paracetamol hepatotoxicity at the same dose. Explains inter-individual toxicity variation.",
        interpatient_variability="high",
    ),
    "warfarin": MicrobiomeDrugEffect(
        drug_name="Warfarin", drug_class="Vitamin K antagonist",
        bacterial_action="Gut bacteria SYNTHESIZE vitamin K2 (menaquinone) → opposes warfarin's mechanism. Antibiotics kill vitamin K producers → INR rises.",
        bacterial_enzyme="Menaquinone biosynthesis pathway",
        key_species=["E. coli", "Bacteroides fragilis", "Klebsiella"],
        effect_type="inactivation", magnitude="moderate",
        metabolite="Vitamin K2 (menaquinone)",
        clinical_impact="Antibiotics (especially broad-spectrum) kill vitamin K-producing bacteria → LESS vitamin K → enhanced warfarin effect → INR spikes → bleeding risk. This is the most common drug-microbiome interaction in clinical practice.",
        interpatient_variability="high",
    ),
}


def simulate_microbiome_effects(
    drug_name: str,
    antibiotic_recent: bool = False,
    probiotic_use: bool = False,
    diet: str = "western",
) -> dict:
    """Simulate gut microbiome effects on drug metabolism."""
    key = drug_name.lower().replace(" ", "_").replace("-", "_")

    # Try alternate names
    if key == "paracetamol":
        key = "acetaminophen"

    drug = MICROBIOME_DRUG_DATA.get(key)

    if not drug:
        return {
            "drug": drug_name,
            "has_microbiome_effect": False,
            "message": "No significant microbiome interaction known",
            "general_note": "Most drugs have SOME microbiome interaction, but only clinically significant ones are modeled.",
        }

    # Antibiotic effect on microbiome-drug interaction
    antibiotic_modifier = 1.0
    antibiotic_warning = ""
    if antibiotic_recent and drug.affected_by_antibiotics:
        if drug.effect_type == "inactivation":
            antibiotic_warning = f"Recent antibiotics may have killed {', '.join(drug.key_species)} → drug levels may be HIGHER than expected"
            antibiotic_modifier = 0.3  # Less bacterial metabolism
        elif drug.effect_type == "activation":
            antibiotic_warning = f"Recent antibiotics may have killed activating bacteria → drug may be LESS effective"
            antibiotic_modifier = 0.3
        elif drug.effect_type == "toxification":
            antibiotic_warning = f"Recent antibiotics may reduce bacterial toxification → LOWER toxicity risk"
            antibiotic_modifier = 0.3
        elif drug.effect_type == "reactivation":
            antibiotic_warning = f"Recent antibiotics reduce enterohepatic recycling → SHORTER drug duration"
            antibiotic_modifier = 0.4

    return {
        "drug": drug.drug_name,
        "drug_class": drug.drug_class,
        "has_microbiome_effect": True,
        "bacterial_metabolism": {
            "action": drug.bacterial_action,
            "enzyme": drug.bacterial_enzyme,
            "key_species": drug.key_species,
            "effect_type": drug.effect_type,
            "magnitude": drug.magnitude,
            "metabolite": drug.metabolite,
        },
        "clinical_impact": drug.clinical_impact,
        "interpatient_variability": drug.interpatient_variability,
        "modifiers": {
            "recent_antibiotics": antibiotic_recent,
            "antibiotic_effect": antibiotic_warning if antibiotic_warning else "No effect",
            "bacterial_activity_modifier": round(antibiotic_modifier, 2),
            "probiotic_use": probiotic_use,
            "diet": diet,
        },
        "recommendations": _generate_microbiome_recommendations(drug, antibiotic_recent),
    }


def _generate_microbiome_recommendations(drug: MicrobiomeDrugEffect, on_antibiotics: bool) -> list[str]:
    """Generate clinical recommendations."""
    recs = []

    if drug.effect_type == "inactivation" and on_antibiotics:
        recs.append(f"Monitor for {drug.drug_name} toxicity — antibiotics may increase drug levels by killing metabolizing bacteria")
    if drug.effect_type == "activation" and on_antibiotics:
        recs.append(f"Monitor for reduced {drug.drug_name} efficacy — antibiotics may reduce bacterial activation")

    if drug.magnitude == "major":
        recs.append(f"This drug-microbiome interaction is clinically significant — consider in dosing decisions")

    if drug.interpatient_variability in ("high", "very high"):
        recs.append(f"High inter-patient variability expected — therapeutic drug monitoring recommended")

    return recs
