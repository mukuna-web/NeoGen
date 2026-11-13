"""
Pulmonary Drug Absorption Model
=================================
Models inhaled drug delivery and lung-specific pharmacology.

Regions of the respiratory tract:
  1. Oropharynx (throat): particles >10 µm impact here → swallowed
  2. Tracheobronchial (conducting airways): 5-10 µm particles deposit
  3. Alveolar (gas exchange): 1-5 µm particles reach here (therapeutic target)
  4. Exhaled: <0.5 µm particles stay airborne → breathed out

Key concepts:
  - MMAD (Mass Median Aerodynamic Diameter): determines where drug deposits
  - Fine Particle Fraction (FPF): % of dose in 1-5 µm range (reaches alveoli)
  - Lung bioavailability: 10-40% of metered dose actually reaches lungs
  - No first-pass metabolism: drug enters systemic circulation directly from lungs
  - Rapid onset: alveolar surface area 70 m² with thin epithelium (0.2 µm)

Pulmonary first-pass effect:
  - Lungs contain CYP enzymes and metabolize some drugs
  - Fentanyl: 75% retained in lungs on first pass (slow release)
  - Propranolol: significant pulmonary extraction
  - Insulin: can be absorbed via lungs (Afrezza™)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InhaledDrugProfile:
    """Profile for an inhaled drug formulation."""
    name: str
    drug_class: str
    device: str                      # MDI, DPI, nebulizer, soft-mist
    metered_dose_ug: float           # Dose leaving device (µg)
    lung_deposition_fraction: float  # Fraction reaching lungs (0.1-0.4 typically)
    mmad_um: float = 3.0            # Mass median aerodynamic diameter
    fpf_percent: float = 40         # Fine particle fraction (1-5 µm)

    # Regional deposition (fractions of lung dose)
    oropharyngeal_fraction: float = 0.5   # Swallowed
    tracheobronchial_fraction: float = 0.2
    alveolar_fraction: float = 0.3

    # PK from lungs
    absorption_half_life_min: float = 5   # How fast drug absorbs from lungs
    pulmonary_retention_fraction: float = 0  # Fraction retained in lung tissue
    local_duration_h: float = 4           # Duration of local effect in airways

    # Systemic effects
    systemic_bioavailability: float = 0.1  # Fraction reaching systemic circulation
    onset_min: float = 5                   # Time to clinical effect
    peak_effect_min: float = 30

    # Clinical
    therapeutic_target: str = ""
    mechanism: str = ""


INHALED_DRUGS = {
    "salbutamol": InhaledDrugProfile(
        name="Salbutamol (Albuterol)", drug_class="Short-acting β2-agonist (SABA)",
        device="MDI (metered dose inhaler)",
        metered_dose_ug=100, lung_deposition_fraction=0.15,
        mmad_um=3.5, fpf_percent=35,
        oropharyngeal_fraction=0.60, tracheobronchial_fraction=0.25, alveolar_fraction=0.15,
        absorption_half_life_min=3, local_duration_h=4,
        systemic_bioavailability=0.10, onset_min=3, peak_effect_min=15,
        therapeutic_target="Bronchial smooth muscle β2 receptors",
        mechanism="β2-agonist → cAMP ↑ → smooth muscle relaxation → bronchodilation",
    ),
    "budesonide": InhaledDrugProfile(
        name="Budesonide", drug_class="Inhaled corticosteroid (ICS)",
        device="DPI (dry powder inhaler)",
        metered_dose_ug=200, lung_deposition_fraction=0.25,
        mmad_um=2.5, fpf_percent=50,
        oropharyngeal_fraction=0.40, tracheobronchial_fraction=0.30, alveolar_fraction=0.30,
        absorption_half_life_min=30, pulmonary_retention_fraction=0.30,
        local_duration_h=12, systemic_bioavailability=0.06,
        onset_min=60, peak_effect_min=360,
        therapeutic_target="Airway epithelial glucocorticoid receptors",
        mechanism="Binds GR → suppresses inflammatory gene transcription → reduces airway inflammation over days-weeks",
    ),
    "ipratropium": InhaledDrugProfile(
        name="Ipratropium", drug_class="Short-acting anticholinergic (SAMA)",
        device="MDI or nebulizer",
        metered_dose_ug=20, lung_deposition_fraction=0.10,
        mmad_um=4.0, fpf_percent=30,
        oropharyngeal_fraction=0.65, tracheobronchial_fraction=0.25, alveolar_fraction=0.10,
        absorption_half_life_min=10, local_duration_h=6,
        systemic_bioavailability=0.02, onset_min=15, peak_effect_min=60,
        therapeutic_target="Airway smooth muscle M3 muscarinic receptors",
        mechanism="Blocks acetylcholine at M3 → reduces bronchoconstriction and mucus secretion",
    ),
    "tiotropium": InhaledDrugProfile(
        name="Tiotropium (Spiriva)", drug_class="Long-acting anticholinergic (LAMA)",
        device="HandiHaler DPI",
        metered_dose_ug=18, lung_deposition_fraction=0.20,
        mmad_um=2.0, fpf_percent=55,
        oropharyngeal_fraction=0.40, tracheobronchial_fraction=0.30, alveolar_fraction=0.30,
        absorption_half_life_min=60, pulmonary_retention_fraction=0.40,
        local_duration_h=24, systemic_bioavailability=0.03,
        onset_min=30, peak_effect_min=180,
        therapeutic_target="M3 muscarinic receptors (slowly dissociates → 24h duration)",
        mechanism="Kinetically selective M3 antagonist — binds M3 for 35h but M2 for only 4h",
    ),
    "formoterol": InhaledDrugProfile(
        name="Formoterol", drug_class="Long-acting β2-agonist (LABA)",
        device="DPI",
        metered_dose_ug=12, lung_deposition_fraction=0.25,
        mmad_um=2.5, fpf_percent=50,
        absorption_half_life_min=5, local_duration_h=12,
        systemic_bioavailability=0.08, onset_min=3, peak_effect_min=120,
        therapeutic_target="β2-adrenergic receptors in bronchial smooth muscle",
        mechanism="Full β2-agonist with fast onset AND long duration (lipophilic depot in membrane)",
    ),
    "insulin_inhaled": InhaledDrugProfile(
        name="Inhaled Insulin (Afrezza)", drug_class="Rapid-acting insulin",
        device="Technosphere inhaler",
        metered_dose_ug=4000, lung_deposition_fraction=0.40,
        mmad_um=2.5, fpf_percent=60,
        absorption_half_life_min=2, local_duration_h=0,  # No local effect — systemic only
        systemic_bioavailability=0.20, onset_min=12, peak_effect_min=35,
        therapeutic_target="Insulin receptors (systemic — muscle, fat, liver)",
        mechanism="Ultra-rapid absorption from alveoli → faster onset than subcutaneous insulin → mealtime glucose control",
    ),
}


def simulate_inhaled_drug(
    drug_name: str,
    puffs: int = 1,
    time_min: float = 15,
) -> dict:
    """Simulate inhaled drug delivery and effects."""
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    drug = INHALED_DRUGS.get(key)

    if not drug:
        return {"drug": drug_name, "has_pulmonary_data": False,
                "available": list(INHALED_DRUGS.keys())}

    total_metered = drug.metered_dose_ug * puffs
    lung_dose = total_metered * drug.lung_deposition_fraction
    oropharyngeal_dose = total_metered * drug.oropharyngeal_fraction
    tracheobronchial = lung_dose * drug.tracheobronchial_fraction / (drug.tracheobronchial_fraction + drug.alveolar_fraction)
    alveolar = lung_dose * drug.alveolar_fraction / (drug.tracheobronchial_fraction + drug.alveolar_fraction)

    # Absorption from lungs (first-order)
    import math
    k_abs = 0.693 / drug.absorption_half_life_min
    absorbed_fraction = 1 - math.exp(-k_abs * time_min)
    remaining_in_lung = lung_dose * (1 - absorbed_fraction) * (1 + drug.pulmonary_retention_fraction)
    systemic_amount = lung_dose * absorbed_fraction * drug.systemic_bioavailability

    # Effect timeline
    if time_min < drug.onset_min:
        effect_fraction = time_min / drug.onset_min * 0.3
    elif time_min < drug.peak_effect_min:
        effect_fraction = 0.3 + 0.7 * (time_min - drug.onset_min) / (drug.peak_effect_min - drug.onset_min)
    elif time_min < drug.local_duration_h * 60:
        effect_fraction = 1.0 - 0.5 * (time_min - drug.peak_effect_min) / (drug.local_duration_h * 60 - drug.peak_effect_min)
    else:
        effect_fraction = 0

    return {
        "drug": drug.name,
        "drug_class": drug.drug_class,
        "device": drug.device,
        "has_pulmonary_data": True,
        "dosing": {
            "puffs": puffs,
            "metered_dose_ug": total_metered,
            "lung_dose_ug": round(lung_dose, 1),
            "oropharyngeal_dose_ug": round(oropharyngeal_dose, 1),
            "tracheobronchial_ug": round(tracheobronchial, 1),
            "alveolar_ug": round(alveolar, 1),
            "lung_deposition_percent": round(drug.lung_deposition_fraction * 100, 1),
        },
        "absorption": {
            "absorbed_fraction": round(absorbed_fraction, 3),
            "remaining_in_lung_ug": round(remaining_in_lung, 1),
            "systemic_amount_ug": round(systemic_amount, 2),
            "absorption_half_life_min": drug.absorption_half_life_min,
        },
        "effect": {
            "effect_fraction": round(effect_fraction, 3),
            "onset_min": drug.onset_min,
            "peak_min": drug.peak_effect_min,
            "local_duration_h": drug.local_duration_h,
            "therapeutic_target": drug.therapeutic_target,
            "mechanism": drug.mechanism,
        },
        "clinical": {
            "swallowed_fraction_note": f"{drug.oropharyngeal_fraction*100:.0f}% of dose impacts oropharynx and is swallowed → undergoes first-pass metabolism → mostly inactive",
            "no_first_pass": "Drug absorbed from alveoli enters pulmonary veins → left heart → systemic circulation. NO first-pass metabolism.",
            "device_technique": f"Proper {drug.device} technique is critical — poor technique can reduce lung deposition by 50%+",
        },
    }
