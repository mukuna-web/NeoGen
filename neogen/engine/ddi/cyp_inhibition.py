"""
CYP Inhibition — V1 DDI Engine (Step 7)
========================================
Static, reversible, competitive CYP inhibition factor computation.

Implements the FDA basic DDI equation for each CYP isoform:

    R_cyp = 1 + [I]u / Ki

where
    R_cyp   = fold-increase in victim drug CLint loss on that CYP
    [I]u    = unbound inhibitor concentration at the enzyme site (µM)
    Ki      = dissociation constant of the inhibitor on that CYP (µM)

R is further translated into a whole-drug AUC fold change via the
standard FDA / Rowland static model, using the victim's fraction
metabolised (fm) per CYP:

    AUCR = 1 / ( Σ_i fm_i / R_i  +  (1 - Σ_i fm_i) )

This module deliberately covers ONLY static, reversible, competitive
inhibition. Three things that look similar but belong elsewhere:

  * Mechanism-based (time-dependent) inhibition kinetics — the
    kinact · [I] / (KI + [I]) inactivation ODE — is handled by
    `enzyme_pool_evolver.py` in Step 9, because it requires a full
    time-dependent integration against the PBPK solver.
  * CYP induction (steady-state Emax scaling) — Step 8.
  * Runtime PBPK coupling (CLint(t) provider) — Step 10.

Keeping Step 7 scope tight matches the plan and produces a small,
auditable, pure-function engine.

References:
    FDA. In Vitro Drug Interaction Studies — CYP Enzyme- and
    Transporter-Mediated Drug Interactions (2020).
    EMA. Guideline on the Investigation of Drug Interactions (2012).
    Rowland M, Matin SB. J Pharmacokinet Biopharm 1973; 1: 553-567.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from neogen.core.entities.drug import Drug

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# ENUMERATION
# ════════════════════════════════════════════════════════════════

#: Canonical list of CYP isoforms the V1 engine tracks. Order matches
#: the field layout on ``Drug.metabolism`` and ``Drug.inhibition``.
CYP_NAMES: Tuple[str, ...] = (
    "CYP3A4",
    "CYP2D6",
    "CYP2C9",
    "CYP2C19",
    "CYP1A2",
    "CYP2C8",
    "CYP2B6",
    "CYP2E1",
)


# ════════════════════════════════════════════════════════════════
# FROZEN RESULT DATACLASSES
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CYPInhibitionContribution:
    """Per-CYP breakdown of a single perpetrator's reversible inhibition.

    Attributes:
        cyp: Canonical CYP name (e.g. ``"CYP3A4"``).
        ki_um: Inhibitor Ki on this CYP (µM). None if the perpetrator is
            not a reported inhibitor of this CYP.
        inhibitor_conc_um: Effective unbound inhibitor concentration (µM)
            at the enzyme site used for this calculation.
        r: Competitive inhibition factor, ``1 + [I]u / Ki``. Always ≥ 1.0.
    """

    cyp: str
    ki_um: Optional[float]
    inhibitor_conc_um: float
    r: float


@dataclass(frozen=True)
class InhibitionState:
    """Immutable per-perpetrator snapshot of CYP inhibition factors.

    ``contributions`` is the canonical data — a tuple of per-CYP
    ``CYPInhibitionContribution`` records (one per tracked CYP). The
    helper methods provide O(8) lookups without allocating a dict.

    Attributes:
        perpetrator_name: Name of the inhibitor drug.
        inhibitor_conc_um: Nominal effective unbound inhibitor
            concentration used across all CYPs for this state.
        site: ``"liver"`` or ``"gut"`` — documentation only; the engine
            is site-agnostic and uses whatever concentration the caller
            supplies.
        contributions: Tuple of per-CYP records.
    """

    perpetrator_name: str
    inhibitor_conc_um: float
    site: str
    contributions: Tuple[CYPInhibitionContribution, ...]

    def r_for(self, cyp: str) -> float:
        """Return the R factor for ``cyp`` (case-insensitive). 1.0 if absent."""
        key = cyp.upper()
        for c in self.contributions:
            if c.cyp == key:
                return c.r
        return 1.0

    @property
    def max_r(self) -> float:
        """Largest R factor across all tracked CYPs (≥ 1.0)."""
        if not self.contributions:
            return 1.0
        return max(c.r for c in self.contributions)

    @property
    def strongest_cyp(self) -> Optional[str]:
        """The CYP with the largest R factor. None if no CYPs tracked."""
        if not self.contributions:
            return None
        return max(self.contributions, key=lambda c: c.r).cyp

    @property
    def is_meaningful(self) -> bool:
        """Whether any CYP has a non-trivial inhibition factor (R > 1.01)."""
        return self.max_r > 1.01

    def as_r_map(self) -> Dict[str, float]:
        """Return a flat ``{cyp: R}`` dict, including CYPs with R = 1.0."""
        return {c.cyp: c.r for c in self.contributions}


# ════════════════════════════════════════════════════════════════
# ENGINE
# ════════════════════════════════════════════════════════════════


class CYPInhibitionEngine:
    """Static, reversible, competitive CYP inhibition calculator.

    Usage:
        engine = CYPInhibitionEngine()
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        print(state.r_for("CYP3A4"))           # ~34
        auc_ratio = engine.victim_auc_ratio(state, midazolam)
    """

    # Maximum R value to guard against divide-by-zero when Ki is very small.
    # Physically, hepatic enzyme activity cannot drop below baseline
    # synthesis / degradation — clipping at 1e4 is defensive only.
    _R_CEILING: float = 1.0e4

    def compute(
        self,
        perpetrator: Drug,
        inhibitor_conc_um: float,
        *,
        site: str = "liver",
    ) -> InhibitionState:
        """Compute per-CYP reversible inhibition factors for a single perpetrator.

        Args:
            perpetrator: V1 ``Drug`` entity. Only ``perpetrator.inhibition``
                is consulted.
            inhibitor_conc_um: Unbound inhibitor concentration (µM) at the
                enzyme site. Caller decides how this is computed — usually
                the hepatic inlet unbound concentration for liver CYPs, or
                the enterocyte luminal concentration for gut CYP3A4.
            site: ``"liver"`` or ``"gut"``; stored in the result for
                provenance, not used in the calculation itself.

        Returns:
            Frozen ``InhibitionState`` containing one contribution per
            tracked CYP (R = 1.0 for CYPs with no reported Ki).
        """
        if inhibitor_conc_um < 0:
            raise ValueError(
                f"inhibitor_conc_um must be ≥ 0, got {inhibitor_conc_um}"
            )
        if site not in ("liver", "gut"):
            raise ValueError(f"site must be 'liver' or 'gut', got {site!r}")

        inhibition = perpetrator.inhibition
        contributions: list[CYPInhibitionContribution] = []

        for cyp in CYP_NAMES:
            ki = inhibition.ki_for(cyp)
            r = self._competitive_r(ki, inhibitor_conc_um)
            contributions.append(
                CYPInhibitionContribution(
                    cyp=cyp,
                    ki_um=ki,
                    inhibitor_conc_um=inhibitor_conc_um,
                    r=r,
                )
            )

        state = InhibitionState(
            perpetrator_name=perpetrator.name,
            inhibitor_conc_um=inhibitor_conc_um,
            site=site,
            contributions=tuple(contributions),
        )

        if state.is_meaningful:
            logger.info(
                "CYP inhibition: %s @ [I]u=%.4g µM (%s) → max R = %.2f on %s",
                perpetrator.name, inhibitor_conc_um, site,
                state.max_r, state.strongest_cyp,
            )

        return state

    def victim_auc_ratio(
        self,
        state: InhibitionState,
        victim: Drug,
    ) -> float:
        """Estimate the whole-drug AUC fold-change in a victim using the FDA
        static model.

        The formula assumes only the metabolic (CYP) clearance pathways are
        affected; renal and non-CYP clearance are unchanged::

            AUCR = 1 / ( Σ_i fm_i / R_i  +  (1 − Σ_i fm_i) )

        This is the fraction-metabolised corrected version of the basic
        FDA static equation and is the canonical hand-calculation used
        during DDI risk assessment.

        Args:
            state: Pre-computed ``InhibitionState`` for the perpetrator.
            victim: Victim drug whose fm values are read.

        Returns:
            AUC fold-change (≥ 1.0 for inhibitors, 1.0 if no overlap).
        """
        fm = victim.metabolism
        total_fm = 0.0
        inhibited_term = 0.0

        for contribution in state.contributions:
            fm_i = fm.fm_for(contribution.cyp)
            if fm_i <= 0.0:
                continue
            total_fm += fm_i
            r_i = max(contribution.r, 1.0)
            inhibited_term += fm_i / r_i

        # (1 - total_fm) represents the non-affected fraction (renal,
        # UGT, biliary, etc.) which is assumed unchanged by the DDI.
        remaining = max(1.0 - total_fm, 0.0)
        denom = inhibited_term + remaining

        if denom <= 0.0:
            # Degenerate: all fm routed through a CYP with R → ∞.
            return self._R_CEILING

        return 1.0 / denom

    # ── Internal helpers ──

    @classmethod
    def _competitive_r(
        cls,
        ki_um: Optional[float],
        inhibitor_conc_um: float,
    ) -> float:
        """Reversible competitive inhibition factor ``R = 1 + [I]u / Ki``.

        Guards:
          * ``Ki`` None or non-positive → R = 1.0 (not an inhibitor)
          * ``[I]u`` = 0 → R = 1.0 (no exposure)
          * R clipped at ``_R_CEILING`` to avoid numerical overflow.
        """
        if ki_um is None or ki_um <= 0.0:
            return 1.0
        if inhibitor_conc_um <= 0.0:
            return 1.0
        r = 1.0 + (inhibitor_conc_um / ki_um)
        return min(r, cls._R_CEILING)
