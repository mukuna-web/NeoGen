"""Deterministic patient-specific scaling for Drug and PhysiologyModel.

This module is the bridge between :class:`PatientProfile` (age,
weight, sex, organ function, genotype) and the V1 engine. It does
not modify any engine file — it produces *scaled* copies of the
reference entities which are then passed unchanged to the
existing SimulationOrchestrator.

Two scaling paths:

    scale_physiology_to_patient(physiology, patient)
        Produces a PhysiologyModel with organ volumes and blood
        flows scaled by allometry, sex, age, and organ function.

    scale_drug_to_genotype(drug, patient)
        Produces a Drug with clint_microsomal / clint_hepatocyte
        scaled by the patient's CYP phenotype activity factors.

Convenience:

    scale_for_patient(drug, physiology, patient)
        Runs both and returns (scaled_drug, scaled_physiology).

The reference physiology used as input must be either
``build_reference_male()`` or ``build_reference_female()`` — these
are the 70 kg / 60 kg ICRP anchors. The scaling functions compute
the ratio to the reference and apply it.

Allometry
---------
Volumes: V_patient = V_ref × (weight_patient / weight_ref)
    Organ volumes scale approximately linearly with body weight
    in adults (ICRP 89). Non-linear scaling is pediatric territory
    (V1.1).

Flows: Q_patient = Q_ref × (weight_patient / weight_ref) ^ 0.75
    Blood flows scale allometrically with body weight^0.75 —
    Kleiber's law / allometric scaling theory.

Age:
    Elderly (>60 y) reductions:
      - Cardiac output: -1%/year from age 60
      - Liver blood flow: -0.8%/year from age 30 (West 1956)
      - GFR: -0.8 mL/min/year from age 30 (Rowe 1976)

Sex (already baked into reference_female but re-applied defensively):
    Females: +20% fat, -15% muscle, -15% CO vs male baseline
    (used only if the reference argument is male and patient is
    female or vice versa).

Organ function:
    hepatic ∈ [0, 1] scales liver blood flow and liver volume
    renal ∈ [0, 1] scales kidney blood flow and GFR (GFR applied
    downstream via the RenalClearanceEngine parameter)
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Tuple

from neogen.core.entities.drug import CYPMetabolismProfile, Drug
from neogen.core.entities.patient import (
    CYPPhenotype,
    GeneticProfile,
    OrganFunction,
    PatientProfile,
    Sex,
)
from neogen.core.entities.physiology import (
    OrganCompartment,
    PhysiologyModel,
)

logger = logging.getLogger(__name__)


REFERENCE_WEIGHT_MALE_KG = 70.0
REFERENCE_WEIGHT_FEMALE_KG = 60.0
REFERENCE_AGE_YEARS = 30.0

# Age coefficients (per year above the reference age).
LIVER_FLOW_AGE_DECLINE_PER_YEAR = 0.008  # 0.8 %/y after 30
CARDIAC_OUTPUT_AGE_DECLINE_PER_YEAR = 0.005  # 0.5 %/y after 30
GFR_AGE_DECLINE_PER_YEAR = 0.0067  # ~0.8 mL/min out of 120 = 0.67 %/y


# ════════════════════════════════════════════════════════════════
# PHYSIOLOGY SCALING
# ════════════════════════════════════════════════════════════════


def scale_physiology_to_patient(
    physiology: PhysiologyModel,
    patient: PatientProfile,
) -> PhysiologyModel:
    """Return a new PhysiologyModel scaled to the given patient.

    The input physiology is not mutated. A deep copy is produced,
    mutated in place, and returned — this matches the V1
    convention that orchestrator inputs flow through the engine
    without aliasing.
    """
    # Pick a reference weight by sex for allometry. The input
    # physiology's total volume gives us the "starting scale",
    # but we ground scaling on the patient's declared sex.
    ref_weight = (
        REFERENCE_WEIGHT_MALE_KG
        if patient.sex == Sex.MALE
        else REFERENCE_WEIGHT_FEMALE_KG
    )

    weight_ratio = patient.weight_kg / ref_weight
    # Kleiber allometry for flows
    flow_ratio = weight_ratio ** 0.75

    # Age adjustments relative to the reference age
    age_delta = max(0.0, patient.age_years - REFERENCE_AGE_YEARS)
    liver_age_factor = max(
        0.5, 1.0 - age_delta * LIVER_FLOW_AGE_DECLINE_PER_YEAR
    )
    co_age_factor = max(
        0.5, 1.0 - age_delta * CARDIAC_OUTPUT_AGE_DECLINE_PER_YEAR
    )

    # Organ function scaling
    hepatic_fn = patient.organ_function.hepatic
    renal_fn = patient.organ_function.renal
    cardiac_fn = patient.organ_function.cardiac

    scaled = copy.deepcopy(physiology)

    # Non-liver, non-kidney organs: simple weight scaling.
    # OrganCompartment is frozen, so each scaled compartment is
    # produced via dataclasses.replace and reassigned on the
    # (mutable) PhysiologyModel.
    scaled.plasma = _scale_organ(scaled.plasma, weight_ratio, flow_ratio)
    scaled.lung = _scale_organ(
        scaled.lung,
        weight_ratio,
        flow_ratio * co_age_factor * cardiac_fn,
    )
    scaled.liver = _scale_organ(
        scaled.liver,
        weight_ratio * hepatic_fn,
        flow_ratio * liver_age_factor * hepatic_fn * co_age_factor,
    )
    scaled.gut = _scale_organ(
        scaled.gut, weight_ratio, flow_ratio * co_age_factor
    )
    scaled.kidney = _scale_organ(
        scaled.kidney,
        weight_ratio,
        flow_ratio * co_age_factor * renal_fn,
    )
    scaled.brain = _scale_organ(
        scaled.brain, weight_ratio, flow_ratio * co_age_factor
    )
    scaled.heart = _scale_organ(
        scaled.heart, weight_ratio, flow_ratio * co_age_factor * cardiac_fn
    )
    scaled.muscle = _scale_organ(
        scaled.muscle, weight_ratio, flow_ratio * co_age_factor
    )
    scaled.fat = _scale_organ(
        scaled.fat, weight_ratio, flow_ratio * co_age_factor
    )
    scaled.skin = _scale_organ(
        scaled.skin, weight_ratio, flow_ratio * co_age_factor
    )
    scaled.bone = _scale_organ(
        scaled.bone, weight_ratio, flow_ratio * co_age_factor
    )
    scaled.spleen = _scale_organ(
        scaled.spleen, weight_ratio, flow_ratio * co_age_factor
    )
    scaled.pancreas = _scale_organ(
        scaled.pancreas, weight_ratio, flow_ratio * co_age_factor
    )
    scaled.rest = _scale_organ(
        scaled.rest, weight_ratio, flow_ratio * co_age_factor
    )

    # Sex-specific fat/muscle redistribution when patient sex differs
    # from the physiology baseline. A female patient fed the male
    # reference gets more fat and less muscle; vice versa.
    _apply_sex_composition(scaled, physiology, patient)

    logger.debug(
        "Scaled physiology: weight_ratio=%.3f flow_ratio=%.3f "
        "hepatic_fn=%.2f renal_fn=%.2f age_delta=%.0f",
        weight_ratio, flow_ratio, hepatic_fn, renal_fn, age_delta,
    )
    return scaled


def _scale_organ(
    organ: OrganCompartment,
    volume_ratio: float,
    flow_ratio: float,
) -> OrganCompartment:
    """Return a scaled copy of a frozen OrganCompartment."""
    return dataclasses.replace(
        organ,
        volume_L=max(1.0e-4, organ.volume_L * volume_ratio),
        blood_flow_L_per_h=max(
            0.0, organ.blood_flow_L_per_h * flow_ratio
        ),
    )


def _apply_sex_composition(
    scaled: PhysiologyModel,
    baseline: PhysiologyModel,
    patient: PatientProfile,
) -> None:
    """Adjust muscle/fat composition when patient sex contradicts baseline.

    Guessing the baseline sex from the incoming physiology is brittle,
    so V1 uses a simple rule: if the patient is female and the baseline
    has more muscle than fat (male shape), redistribute ~20% of muscle
    into fat. Vice versa for male patients built from a female baseline.

    OrganCompartment is frozen, so muscle/fat are re-created via
    :func:`dataclasses.replace` and reassigned on the mutable
    :class:`PhysiologyModel`.
    """
    baseline_muscle_ratio = (
        baseline.muscle.volume_L / max(baseline.fat.volume_L, 1.0e-3)
    )
    baseline_is_male_shape = baseline_muscle_ratio > 1.0
    patient_is_female = patient.sex == Sex.FEMALE

    if baseline_is_male_shape and patient_is_female:
        transfer = scaled.muscle.volume_L * 0.2
    elif (not baseline_is_male_shape) and (not patient_is_female):
        transfer = -scaled.fat.volume_L * 0.2  # negative = fat→muscle
    else:
        return

    new_muscle_V = max(1.0e-4, scaled.muscle.volume_L - transfer)
    new_fat_V = max(1.0e-4, scaled.fat.volume_L + transfer)
    scaled.muscle = dataclasses.replace(
        scaled.muscle, volume_L=new_muscle_V
    )
    scaled.fat = dataclasses.replace(
        scaled.fat, volume_L=new_fat_V
    )


def effective_gfr_ml_per_min(
    patient: PatientProfile, reference_gfr: float = 120.0
) -> float:
    """Return the GFR to pass into :class:`RenalClearanceEngine`.

    Applies the patient's renal function scalar and an age-related
    decline starting at 30. The result is bounded to [5, 150] mL/min.
    """
    age_delta = max(0.0, patient.age_years - REFERENCE_AGE_YEARS)
    age_factor = max(
        0.1, 1.0 - age_delta * GFR_AGE_DECLINE_PER_YEAR
    )
    gfr = reference_gfr * patient.organ_function.renal * age_factor
    return max(5.0, min(gfr, 150.0))


# ════════════════════════════════════════════════════════════════
# DRUG SCALING — GENOTYPE-DRIVEN CLint
# ════════════════════════════════════════════════════════════════


def genotype_clint_multiplier(
    drug_metabolism: CYPMetabolismProfile,
    genetics: GeneticProfile,
) -> float:
    """Composite multiplicative factor applied to CLint for this patient.

    Formula:

        factor = non_cyp_fraction
               + Σ_i  fm_i × activity_factor_i(genetics)

    Where ``i`` ranges over CYP isoforms that have both a non-zero
    fm on the drug and a phenotype on the patient. The non-CYP
    tail is unaffected by genotype. Activity factors come from
    :class:`CYPPhenotype.activity_factor` (POOR=0, IM=0.5,
    NM=1, UM=2).

    At the reference (all NM) this evaluates to 1.0 exactly for
    any drug, so the scaling is a no-op when no genotype
    information is present.
    """
    # Only the six phenotype-tracked CYPs in GeneticProfile
    tracked_cyps = (
        ("CYP3A4", drug_metabolism.cyp3a4),
        ("CYP2D6", drug_metabolism.cyp2d6),
        ("CYP2C9", drug_metabolism.cyp2c9),
        ("CYP2C19", drug_metabolism.cyp2c19),
        ("CYP1A2", drug_metabolism.cyp1a2),
        ("CYP2B6", drug_metabolism.cyp2b6),
    )
    # Enzymes not phenotype-tracked (CYP2C8, CYP2E1) contribute at NM.
    untracked_fm = drug_metabolism.cyp2c8 + drug_metabolism.cyp2e1

    tracked_activity = 0.0
    for enzyme_name, fm in tracked_cyps:
        if fm <= 0.0:
            continue
        activity = genetics.activity_factor_for(enzyme_name)
        tracked_activity += fm * activity

    return drug_metabolism.non_cyp_fraction + tracked_activity + untracked_fm


def scale_drug_to_genotype(
    drug: Drug,
    patient: PatientProfile,
) -> Drug:
    """Return a new Drug with CLint scaled by the patient's CYP genotype.

    The base Drug is never mutated. A frozen replica is produced
    via :func:`dataclasses.replace`. If the drug has no clint data
    at all, the input is returned unchanged.
    """
    factor = genotype_clint_multiplier(drug.metabolism, patient.genetics)

    if abs(factor - 1.0) < 1.0e-6:
        return drug  # no-op: reference phenotype

    updates: dict = {}
    if drug.clint_microsomal is not None:
        updates["clint_microsomal"] = max(
            0.0, drug.clint_microsomal * factor
        )
    if drug.clint_hepatocyte is not None:
        updates["clint_hepatocyte"] = max(
            0.0, drug.clint_hepatocyte * factor
        )

    if not updates:
        # No CLint to scale — return the original drug. The
        # genotype will have no effect on this run, which is a
        # valid outcome (e.g. drugs eliminated purely renally).
        return drug

    scaled = dataclasses.replace(drug, **updates)
    logger.debug(
        "Scaled drug %s CLint by factor %.3f (genotype=%s)",
        drug.name, factor, _genotype_signature(patient.genetics),
    )
    return scaled


def _genotype_signature(gen: GeneticProfile) -> str:
    """Compact string for logging a GeneticProfile."""
    parts = []
    for cyp in ("cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2", "cyp2b6"):
        pheno: CYPPhenotype = getattr(gen, cyp)
        if pheno != CYPPhenotype.NORMAL:
            parts.append(f"{cyp}={pheno.name}")
    return ",".join(parts) or "all_NM"


# ════════════════════════════════════════════════════════════════
# COMBINED HELPER
# ════════════════════════════════════════════════════════════════


def scale_for_patient(
    drug: Drug,
    physiology: PhysiologyModel,
    patient: PatientProfile,
) -> Tuple[Drug, PhysiologyModel]:
    """One-stop shop: scale both Drug and PhysiologyModel for a patient.

    Returns (scaled_drug, scaled_physiology). Either can be passed
    directly to the existing engine orchestrator without further
    changes.
    """
    scaled_drug = scale_drug_to_genotype(drug, patient)
    scaled_phys = scale_physiology_to_patient(physiology, patient)
    return scaled_drug, scaled_phys
