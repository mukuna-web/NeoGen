"""Pediatric CYP Ontogeny — Phase 6.2 (Scaffold Only).

Models age-dependent CYP enzyme maturation for pediatric PBPK dose
scaling. CYP enzymes mature at different rates during postnatal
development, producing substantially different drug clearance in
neonates and infants compared to adults.

Scientific basis
----------------
CYP ontogeny follows a sigmoidal maturation pattern characterized by:

* **CYP3A7** dominant at birth, rapidly replaced by CYP3A4 over
  the first year of life.
* **CYP3A4** reaches adult levels by ~1 year.
* **CYP1A2** is essentially absent at birth, reaching adult levels
  by ~3-5 years. Caffeine clearance in neonates is ~10% of adult.
* **CYP2D6** matures gradually, reaching adult levels by ~5 years.
* **CYP2C9** reaches adult levels by ~1-2 years.
* **CYP2C19** reaches adult levels by ~2 years.

These maturation profiles have profound clinical consequences:
a neonate receiving an adult-scaled midazolam dose would experience
dramatically prolonged sedation because CYP3A4 is immature.

Limitations (scaffold)
----------------------
This is a scaffold implementation using linear maturation ramps.
Production-grade pediatric PBPK requires:

* Hill-equation maturation functions (Upreti 2016)
* Ontogeny of Phase II enzymes (UGTs mature differently from CYPs)
* Age-dependent organ volumes and blood flows (ICRP 89)
* Age-dependent protein binding (lower albumin in neonates)
* Maturation of renal function (GFR, tubular secretion)

References: Upreti 2016, Salem 2013, Hines 2008, Johnson 2006,
de Wildt 1999, Kearns 2003.
"""
from __future__ import annotations

from typing import Dict

from neogen.core.entities.drug import Drug


# ════════════════════════════════════════════════════════════════
# CYP MATURATION CURVES
# ════════════════════════════════════════════════════════════════

# Age (years) at which each CYP reaches adult activity.
# Linear ramp from 0 at birth to 1.0 at maturation age.
# This is a simplified scaffold — production code should use
# Hill-equation fits from Upreti 2016 / Salem 2013.
_MATURATION_AGE_YEARS: Dict[str, float] = {
    "CYP3A4": 1.0,    # rapid postnatal induction
    "CYP1A2": 4.0,    # very slow maturation
    "CYP2D6": 5.0,    # gradual
    "CYP2C9": 1.5,    # moderate
    "CYP2C19": 2.0,   # moderate
    "CYP2C8": 2.0,    # assumed similar to CYP2C9
    "CYP2B6": 2.0,    # limited data
    "CYP2E1": 1.0,    # rapid maturation (Johnsrud 2003)
}


def cyp_maturation_factor(age_years: float, cyp_name: str) -> float:
    """Return fraction of adult CYP activity at a given age.

    Parameters
    ----------
    age_years
        Patient age in years. Negative values are clamped to 0.
        Values above the maturation age return 1.0.
    cyp_name
        CYP isoform name (e.g. "CYP3A4", "CYP2D6"). Case-insensitive.

    Returns
    -------
    float
        Fraction of adult enzyme activity, range [0.0, 1.0].
        Unknown CYP names return 1.0 (assumed mature).

    Examples
    --------
    >>> cyp_maturation_factor(0.5, "CYP3A4")  # 6-month-old
    0.5
    >>> cyp_maturation_factor(2.0, "CYP3A4")  # 2-year-old
    1.0
    >>> cyp_maturation_factor(0.0, "CYP1A2")  # neonate
    0.0
    """
    age = max(0.0, age_years)
    cyp_key = cyp_name.upper()

    maturation_age = _MATURATION_AGE_YEARS.get(cyp_key)
    if maturation_age is None:
        return 1.0  # unknown CYP — assume mature

    if age >= maturation_age:
        return 1.0

    return age / maturation_age


def scale_clint_for_age(
    clint_adult: float,
    drug: Drug,
    age_years: float,
) -> float:
    """Scale adult CLint by weighted CYP maturation factors.

    Each CYP isoform contributes its fraction metabolized (fm) weighted
    by its maturation factor at the given age. Non-CYP pathways are
    assumed to be mature at all ages (conservative assumption — UGT
    ontogeny is actually slower than most CYPs).

    Parameters
    ----------
    clint_adult
        Adult intrinsic clearance (any units — scaling is fractional).
    drug
        Drug entity with CYP metabolism profile.
    age_years
        Patient age in years.

    Returns
    -------
    float
        Age-adjusted CLint. Always >= 0.

    Examples
    --------
    For a CYP3A4-dominant drug (fm_3A4=0.9) in a neonate (age=0):

    >>> scale_clint_for_age(100.0, drug_midazolam, 0.0)
    10.0  # only non-CYP fraction (0.1) is mature

    For the same drug in a 1-year-old:

    >>> scale_clint_for_age(100.0, drug_midazolam, 1.0)
    100.0  # CYP3A4 fully mature by 1 year
    """
    metabolism = drug.metabolism

    # CYP-mediated fraction, weighted by maturation
    total_scaling = 0.0
    for cyp_name in ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2",
                      "CYP2C8", "CYP2B6", "CYP2E1"]:
        attr_name = cyp_name.lower().replace("cyp", "cyp")
        fm = getattr(metabolism, attr_name, 0.0)
        mat = cyp_maturation_factor(age_years, cyp_name)
        total_scaling += fm * mat

    # Non-CYP pathways assumed mature at all ages
    non_cyp = metabolism.non_cyp_fraction
    total_scaling += non_cyp

    return clint_adult * total_scaling
