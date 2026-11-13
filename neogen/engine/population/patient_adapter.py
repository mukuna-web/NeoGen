"""Patient adapter — compose :class:`PatientProfile` into engine inputs.

This module is the **single entry point** by which the P5 population layer
modifies the P1 engine. It produces small, composable *scale factors* and
wraps them into drop-in replacements for the ``Drug`` and
``PhysiologyModel`` that the rest of the engine already accepts.

Design rules
------------

1. **Stateless, pure functions.** Every function takes frozen inputs and
   returns a frozen output. No side effects.
2. **Backward compatible.** Every public helper has a ``patient=None``
   short-circuit that returns the input unchanged, so the rest of the
   engine keeps working for legacy callers.
3. **No new fields on ``Drug``.** Patient effects are expressed by
   producing a *new* ``Drug`` via :func:`dataclasses.replace` — the same
   trick the P4 tests use for attaching a PD profile.
4. **Physics lives here, not in the engine.** The clearance engines stay
   oblivious to ``PatientProfile``. They receive a ``Drug`` whose
   ``clint_*`` and ``fu_plasma`` fields already carry the patient effect,
   and a ``PhysiologyModel`` whose liver/kidney flows already reflect
   organ function.

Scaling model
-------------

The total intrinsic-clearance scale applied to a victim drug is::

    scale = (Σ_i fm_i · phenotype_activity_i) + fm_non_cyp
            × hepatic_impairment_factor
            × smoker_cyp1a2_factor
            × age_maturation_factor

Where:

* ``phenotype_activity_i`` comes from the patient's ``GeneticProfile``
  (PM=0, IM=0.5, NM=1.0, UM=2.0 from ``CYPPhenotype.activity_factor``).
* ``fm_non_cyp`` keeps the non-CYP pathway unchanged — phenotype only
  scales what the CYPs actually metabolize.
* ``hepatic_impairment_factor`` maps ``OrganFunction.hepatic`` to a
  Child-Pugh-style CLint multiplier (A ≈ 0.7, B ≈ 0.4, C ≈ 0.2) via the
  same score. Score = 1.0 → factor = 1.0.
* ``smoker_cyp1a2_factor`` = 1 + 0.7 · fm_1a2 for smokers, else 1.0.
  Rationale: smoking roughly doubles CYP1A2 activity (theophylline,
  caffeine, clozapine); only the CYP1A2-metabolized fraction is affected.
* ``age_maturation_factor`` is a coarse pediatric / elderly envelope.

Renal
-----

GFR is the only renal knob P5 exposes. It's derived from
``OrganFunction.gfr_ml_per_min`` and passed explicitly to
``RenalClearanceEngine.compute`` at simulation time. We do **not** try
to rebuild the kidney compartment volumes.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from neogen.core.entities.drug import Drug
from neogen.core.entities.patient import (
    CYPPhenotype,
    GeneticProfile,
    OrganFunction,
    PatientProfile,
    Sex,
)
from neogen.core.entities.physiology import PhysiologyModel


# ════════════════════════════════════════════════════════════════
# CYP phenotype × metabolism — the headline P5 effect
# ════════════════════════════════════════════════════════════════


_CYP_NAMES = (
    "CYP3A4",
    "CYP2D6",
    "CYP2C9",
    "CYP2C19",
    "CYP1A2",
    "CYP2C8",
    "CYP2B6",
    "CYP2E1",
)


def effective_cyp_activity_factor(
    drug: Drug,
    patient: Optional[PatientProfile],
) -> float:
    """Weighted average CYP activity factor for a drug in a patient.

    This is the scalar that multiplies the hepatic ``CLint`` for the
    victim drug::

        factor = Σ_i (fm_i · activity_i) + fm_non_cyp

    The non-CYP fraction is preserved at 1.0 because phenotype only
    scales the metabolism that actually goes through the CYPs; renal,
    UGT, and biliary pathways are unaffected.

    ``patient=None`` returns 1.0 (reference individual, no change).
    """
    if patient is None:
        return 1.0

    genetics = patient.genetics
    metabolism = drug.metabolism

    cyp_contribution = 0.0
    for cyp in _CYP_NAMES:
        fm = metabolism.fm_for(cyp)
        if fm <= 0.0:
            continue
        cyp_contribution += fm * genetics.activity_factor_for(cyp)

    non_cyp = metabolism.non_cyp_fraction
    factor = cyp_contribution + non_cyp

    # Floor to avoid zero-CLint explosions for drugs that are entirely
    # metabolized by a CYP the patient lacks. The well-stirred liver
    # model handles zero CLint gracefully, but downstream code is
    # happier with a small positive number.
    return max(factor, 1e-4)


# ════════════════════════════════════════════════════════════════
# Organ impairment
# ════════════════════════════════════════════════════════════════


def hepatic_impairment_factor(patient: Optional[PatientProfile]) -> float:
    """Map ``OrganFunction.hepatic`` (0–1) to a CLint multiplier.

    The relationship is linear in score but capped at the upper end so
    that ``hepatic = 1.0`` → factor = 1.0 (reference, no impairment).
    A score of 0.7 (Child-Pugh A) → ~0.7; 0.5 (B) → ~0.5; 0.3 (C) → ~0.3.
    """
    if patient is None:
        return 1.0
    score = float(patient.organ_function.hepatic)
    if score >= 1.0:
        return 1.0
    # Linear scaling down to a physiological floor.
    return max(score, 0.05)


def smoker_cyp1a2_factor(
    drug: Drug,
    patient: Optional[PatientProfile],
) -> float:
    """Smoker CYP1A2 induction, weighted by the drug's fm_CYP1A2.

    Tobacco smoke induces CYP1A2 (PAH → AhR pathway). Observed AUC
    ratios are ~0.5 for caffeine and theophylline in smokers — i.e.
    ~2× the CLint. We express this as::

        factor = 1 + 0.7 · fm_1a2     # ≈ 1.7× when fm_1a2 = 1.0
    """
    if patient is None or not patient.is_smoker:
        return 1.0
    fm_1a2 = drug.metabolism.cyp1a2
    return 1.0 + 0.7 * fm_1a2


def age_cyp_maturation_factor(
    drug: Drug,
    patient: Optional[PatientProfile],
) -> float:
    """Coarse pediatric / geriatric CLint scaling.

    Not a full ontogeny model — V2 material. For V1 we apply a flat
    envelope:

    * age < 1 year  → 0.40 (immature CYPs, esp. 3A4 and 2C19)
    * age 1–5 years → 0.70
    * age 5–12      → 0.90
    * age 12–65     → 1.00 (reference)
    * age 65–75     → 0.85
    * age > 75      → 0.70 (sarcopenia, reduced liver mass & flow)

    The effect is applied uniformly across all CYPs, because V1 doesn't
    yet carry isoform-specific ontogeny curves.
    """
    if patient is None:
        return 1.0
    age = patient.age_years
    if age < 1.0:
        return 0.40
    if age < 5.0:
        return 0.70
    if age < 12.0:
        return 0.90
    if age < 65.0:
        return 1.00
    if age < 75.0:
        return 0.85
    return 0.70


# ════════════════════════════════════════════════════════════════
# Renal
# ════════════════════════════════════════════════════════════════


def gfr_for_patient(patient: Optional[PatientProfile]) -> float:
    """Return the patient's GFR in mL/min (default 120 if no patient)."""
    if patient is None:
        return 120.0
    return float(patient.organ_function.gfr_ml_per_min)


# ════════════════════════════════════════════════════════════════
# Composition — single scale applied to a drug
# ════════════════════════════════════════════════════════════════


def total_clint_scale(
    drug: Drug,
    patient: Optional[PatientProfile],
) -> float:
    """Product of every hepatic-CLint scale factor for this (drug, patient).

    This is the single number the patient adapter applies to
    ``drug.clint_microsomal`` / ``drug.clint_hepatocyte``. Keeping it as
    a product means each factor is independently testable and the
    composition rule is visible in one place.
    """
    if patient is None:
        return 1.0
    return (
        effective_cyp_activity_factor(drug, patient)
        * hepatic_impairment_factor(patient)
        * smoker_cyp1a2_factor(drug, patient)
        * age_cyp_maturation_factor(drug, patient)
    )


# ════════════════════════════════════════════════════════════════
# Drug / physiology rewriting
# ════════════════════════════════════════════════════════════════


def apply_patient_to_drug(
    drug: Drug,
    patient: Optional[PatientProfile],
) -> Drug:
    """Return a new ``Drug`` whose CLint fields carry the patient effect.

    The engine's :class:`HepaticClearanceEngine` consumes
    ``drug.clint_microsomal`` / ``drug.clint_hepatocyte`` directly, so
    pre-scaling these fields is the cleanest way to inject P5 effects
    without touching the engine. Returns the original ``Drug`` unchanged
    when ``patient`` is ``None``.

    The scale is applied to **whichever** CLint field is populated — the
    engine's ``_scale_clint`` helper prefers hepatocyte over microsomal,
    and we mirror that priority here.
    """
    if patient is None:
        return drug

    scale = total_clint_scale(drug, patient)
    if scale == 1.0:
        return drug

    updates: dict = {}
    if drug.clint_hepatocyte is not None:
        updates["clint_hepatocyte"] = drug.clint_hepatocyte * scale
    if drug.clint_microsomal is not None:
        updates["clint_microsomal"] = drug.clint_microsomal * scale
    # Michaelis-Menten: scale Vmax by the same factor so that PM/IM/UM
    # phenotypes and hepatic impairment still modulate the saturable
    # CLint path (CLint_eff = Vmax/(2*Km) × scale).
    if drug.vmax_pmol_per_min_per_mg is not None:
        updates["vmax_pmol_per_min_per_mg"] = (
            drug.vmax_pmol_per_min_per_mg * scale
        )

    if not updates:
        # Drug has no CLint data — nothing to scale. Return as-is.
        return drug

    return dataclasses.replace(drug, **updates)


def apply_patient_to_physiology(
    physiology: PhysiologyModel,
    patient: Optional[PatientProfile],
) -> PhysiologyModel:
    """Return the physiology model, unchanged in V1.

    Why this is a pass-through
    --------------------------

    The obvious move is to scale ``liver.blood_flow_L_per_h`` by
    ``organ_function.hepatic`` — but that breaks an invariant the rest
    of the engine depends on: ``hepatic_artery_flow = liver.blood_flow
    − portal_vein_flow``. With the default portal feeders (gut 60 +
    spleen 9 + pancreas 6 = 75 L/h) and a liver scale of 0.5, hepatic
    artery flow would go negative.

    For V1 we therefore keep liver/kidney blood flows at the reference
    values and express patient effects entirely through:

    * ``apply_patient_to_drug`` — scales CLint (hepatic impairment,
      phenotype, smoker, age). The well-stirred model is
      linear-homogeneous in (CLint, Qh), so for low-extraction drugs a
      CLint-only scaling captures the impairment correctly, and for
      high-extraction drugs the dominant metabolic effect still rides
      on CLint via ``fu·CLint / (Qh + fu·CLint)`` → ``fu·CLint/Qh`` as
      CLint falls.
    * ``gfr_for_patient`` — handed directly to ``RenalClearanceEngine``
      at simulation time, so renal impairment does not require a
      physiology rewrite at all.

    Properly shunt-aware hepatic blood flow scaling (portal
    hypertension, porto-systemic shunting) is P5 V2 material.
    """
    del patient  # unused in V1 — preserved in signature for forward compat
    return physiology
