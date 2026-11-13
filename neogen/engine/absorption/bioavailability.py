"""Bioavailability decomposition — V1 P2 Step 4b.

Standalone module that computes the oral bioavailability decomposition:

    F = Fa × Fg × Fh

where:
    Fa = fraction absorbed (from AbsorptionProfile or BCS estimation)
    Fg = fraction escaping gut-wall metabolism (from AbsorptionProfile)
    Fh = fraction escaping hepatic first-pass (from HepaticClearanceResult)

Also provides BCS classification from molecular descriptors when the
Drug entity's ``bcs_class`` is not set.

Architecture reference: V1 Blueprint — Layer 2, Step 4.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from neogen.core.entities.drug import BCSClass, Drug
from neogen.engine.absorption.absorption_profile import AbsorptionProfile

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# FROZEN RESULT
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BioavailabilityReport:
    """Immutable decomposition of oral bioavailability.

    All fractions are dimensionless ∈ [0, 1].
    """

    drug_name: str
    bcs_class: BCSClass
    fa: float
    fg: float
    fh: float
    f_oral: float

    solubility: str
    permeability: str
    source: str  # "mechanistic" | "estimated"

    def __post_init__(self) -> None:
        for name, val in (
            ("fa", self.fa),
            ("fg", self.fg),
            ("fh", self.fh),
            ("f_oral", self.f_oral),
        ):
            if not (0.0 <= val <= 1.0 + 1e-6):
                raise ValueError(f"{name} must be in [0, 1], got {val}")


# ════════════════════════════════════════════════════════════════
# BCS CLASSIFICATION FROM DESCRIPTORS
# ════════════════════════════════════════════════════════════════


def classify_bcs(
    logp: float,
    tpsa: float,
) -> tuple[BCSClass, str, str]:
    """Classify BCS class from LogP (solubility proxy) and TPSA (permeability).

    Returns (bcs_class, solubility_label, permeability_label).

    References:
        Egan et al. J Med Chem 2000 — TPSA permeability model.
        Lipinski et al. Adv Drug Deliv Rev 2001 — solubility from LogP.
    """
    sol_high = logp < 2.0
    perm_high = tpsa < 90.0

    solubility = "high" if sol_high else "low"
    permeability = "high" if perm_high else "low"

    if sol_high and perm_high:
        bcs = BCSClass.I
    elif not sol_high and perm_high:
        bcs = BCSClass.II
    elif sol_high and not perm_high:
        bcs = BCSClass.III
    else:
        bcs = BCSClass.IV

    return bcs, solubility, permeability


# ════════════════════════════════════════════════════════════════
# FA ESTIMATION (WHEN P2 PROFILE IS NOT AVAILABLE)
# ════════════════════════════════════════════════════════════════


def estimate_fa(tpsa: float, logp: float) -> float:
    """Estimate fraction absorbed from TPSA and LogP.

    Empirical model based on Egan egg:
        TPSA < 75  → excellent absorption (>95%)
        TPSA 75-120 → linear decline
        TPSA > 120 → poor absorption

    Low solubility (high LogP) further penalises dissolution.
    """
    if tpsa < 75.0:
        fa = 0.95
    elif tpsa < 120.0:
        fa = 0.80 - 0.002 * (tpsa - 75.0)
    elif tpsa < 150.0:
        fa = 0.50 - 0.005 * (tpsa - 120.0)
    else:
        fa = max(0.10, 0.35 - 0.003 * (tpsa - 150.0))

    # Dissolution penalty for low solubility
    if logp >= 4.0:
        fa *= 0.75
    elif logp >= 2.0:
        fa *= 0.90

    return min(0.99, max(0.05, fa))


def estimate_fg(logp: float) -> float:
    """Estimate gut-wall escape fraction from LogP.

    More lipophilic → more gut CYP3A4 metabolism.
    """
    if logp > 4.0:
        return 0.60
    elif logp > 2.0:
        return 0.80
    return 0.95


def estimate_fh(logp: float, mw: float) -> float:
    """Estimate hepatic escape fraction from LogP and MW.

    High extraction ratio drugs (lipophilic, high MW) → low Fh.
    """
    if logp > 4.0 and mw > 300.0:
        return 0.40
    elif logp > 2.0:
        return 0.70
    return 0.90


# ════════════════════════════════════════════════════════════════
# ENGINE
# ════════════════════════════════════════════════════════════════


class BioavailabilityEngine:
    """Stateless engine that computes the Fa·Fg·Fh decomposition.

    Two modes of operation:

    1. **Mechanistic** — when a P2 ``AbsorptionProfile`` and a P1
       ``HepaticClearanceResult`` are available, the report is derived
       from actual simulation outputs.

    2. **Estimated** — when the above are not available (e.g. quick
       screening), the engine estimates Fa/Fg/Fh from molecular
       descriptors (LogP, TPSA, MW).
    """

    @staticmethod
    def from_profile(
        drug: Drug,
        absorption: AbsorptionProfile,
        fh: float,
    ) -> BioavailabilityReport:
        """Build a mechanistic report from P2 absorption + P1 hepatic Fh.

        Args:
            drug: V1 Drug entity.
            absorption: Frozen P2 AbsorptionProfile (provides Fa, Fg).
            fh: Hepatic escape fraction from HepaticClearanceResult
                (``1 - extraction_ratio``).
        """
        fa = absorption.fa
        fg = absorption.fg
        f_oral = fa * fg * fh

        sol, perm = _classify_drug_descriptors(drug)

        return BioavailabilityReport(
            drug_name=drug.name,
            bcs_class=drug.bcs_class,
            fa=round(fa, 4),
            fg=round(fg, 4),
            fh=round(fh, 4),
            f_oral=round(min(f_oral, 1.0), 4),
            solubility=sol,
            permeability=perm,
            source="mechanistic",
        )

    @staticmethod
    def estimate(
        drug: Drug,
        logp: Optional[float] = None,
        tpsa: Optional[float] = None,
        mw: Optional[float] = None,
    ) -> BioavailabilityReport:
        """Estimate F = Fa·Fg·Fh from molecular descriptors.

        Falls back to default descriptor values when the drug entity
        does not carry them and no overrides are given.
        """
        _logp = logp if logp is not None else getattr(drug, "logp", 2.0)
        _tpsa = tpsa if tpsa is not None else getattr(drug, "tpsa", 60.0)
        _mw = mw if mw is not None else getattr(drug, "mw", 300.0)

        bcs, sol_label, perm_label = classify_bcs(_logp, _tpsa)

        fa = estimate_fa(_tpsa, _logp)
        fg = estimate_fg(_logp)
        fh = estimate_fh(_logp, _mw)
        f_oral = fa * fg * fh

        return BioavailabilityReport(
            drug_name=drug.name,
            bcs_class=bcs,
            fa=round(fa, 4),
            fg=round(fg, 4),
            fh=round(fh, 4),
            f_oral=round(min(f_oral, 1.0), 4),
            solubility=sol_label,
            permeability=perm_label,
            source="estimated",
        )


def _classify_drug_descriptors(drug: Drug) -> tuple[str, str]:
    """Return (solubility, permeability) labels from the drug's BCS class."""
    bcs = drug.bcs_class
    sol = "high" if bcs in (BCSClass.I, BCSClass.III) else "low"
    perm = "high" if bcs in (BCSClass.I, BCSClass.II) else "low"
    return sol, perm
