"""
CYP Induction — V1 DDI Engine (Step 8)
========================================
Static, steady-state CYP induction via the FDA basic Emax model.

For each inducible CYP isoform:

    fold_induction = 1 + Indmax · [I]u / (IndC50 + [I]u)

where
    Indmax  = maximum fold-increase in enzyme abundance (in vitro Emax)
    IndC50  = inducer concentration giving half-maximum induction (µM)
    [I]u    = unbound inducer concentration at the enzyme site (µM)

At steady state, clearance for CYP substrates scales linearly with
enzyme abundance::

    CLint_induced = CLint_baseline · fold_induction

Corresponding victim whole-drug AUC fold-change (FDA static model)::

    AUCR = 1 / ( Σ_i fm_i · fold_i  +  (1 − Σ_i fm_i) )

Since ``fold_i ≥ 1``, induction always produces ``AUCR ≤ 1`` — the
victim is cleared faster and its exposure is reduced. Classic example:
rifampicin + midazolam gives AUCR ≈ 0.05 (95 % reduction).

Scope of Step 8 — STEADY STATE ONLY:
    * No time course (t=0 to t=7 days). That lives in
      ``enzyme_pool_evolver.py`` at Step 9, which solves
      ``dE/dt = kdeg · (E_ss(t) − E)`` against the PBPK RHS.
    * No autoinduction. That is a V2+ feature.
    * CYP2D6 is not inducible and has no fields in ``InductionProfile``
      (Ingelman-Sundberg 2005).

References:
    Fahmi OA, Maurer TS et al. Drug Metab Dispos 2008; 36: 1698-1708.
    FDA. In Vitro Drug Interaction Studies — CYP Enzyme- and
    Transporter-Mediated Drug Interactions (2020).
    Almond LM et al. Drug Metab Dispos 2016; 44: 821-832.
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

#: The four inducible CYPs in the V1 ``InductionProfile``.
#: CYP2D6 is deliberately absent — it is not inducible in vivo.
INDUCIBLE_CYP_NAMES: Tuple[str, ...] = (
    "CYP3A4",
    "CYP2B6",
    "CYP1A2",
    "CYP2C9",
)


# ════════════════════════════════════════════════════════════════
# FROZEN RESULT DATACLASSES
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CYPInductionContribution:
    """Per-CYP breakdown of a single perpetrator's induction effect.

    Attributes:
        cyp: CYP isoform name (e.g. ``"CYP3A4"``).
        indmax: Maximum fold-increase (Emax) from the inducer's profile.
            None if the drug has no reported Indmax on this CYP.
        indc50_um: Inducer concentration giving half-max induction (µM).
        inducer_conc_um: Effective unbound inducer concentration (µM).
        fold: Steady-state fold-change ``1 + Indmax · C / (IndC50 + C)``.
            Always ≥ 1.0.
    """

    cyp: str
    indmax: Optional[float]
    indc50_um: Optional[float]
    inducer_conc_um: float
    fold: float


@dataclass(frozen=True)
class InductionState:
    """Immutable per-perpetrator snapshot of CYP induction factors.

    Attributes:
        perpetrator_name: Name of the inducing drug.
        inducer_conc_um: Nominal unbound inducer concentration used
            across all CYPs.
        site: ``"liver"`` or ``"gut"`` — documentation only.
        contributions: Tuple of per-CYP records. Covers only the CYPs
            in :data:`INDUCIBLE_CYP_NAMES`.
    """

    perpetrator_name: str
    inducer_conc_um: float
    site: str
    contributions: Tuple[CYPInductionContribution, ...]

    def fold_for(self, cyp: str) -> float:
        """Return the fold-change for ``cyp`` (case-insensitive).

        Returns 1.0 for non-inducible CYPs and CYPs with no Indmax data.
        """
        key = cyp.upper()
        for c in self.contributions:
            if c.cyp == key:
                return c.fold
        return 1.0

    @property
    def max_fold(self) -> float:
        """Largest fold across tracked CYPs (≥ 1.0)."""
        if not self.contributions:
            return 1.0
        return max(c.fold for c in self.contributions)

    @property
    def strongest_cyp(self) -> Optional[str]:
        """CYP with the largest induction fold. None if nothing tracked."""
        if not self.contributions:
            return None
        return max(self.contributions, key=lambda c: c.fold).cyp

    @property
    def is_meaningful(self) -> bool:
        """Whether any CYP has a non-trivial induction (fold > 1.1)."""
        return self.max_fold > 1.1

    def as_fold_map(self) -> Dict[str, float]:
        """Return a ``{cyp: fold}`` dict for the tracked CYPs."""
        return {c.cyp: c.fold for c in self.contributions}


# ════════════════════════════════════════════════════════════════
# ENGINE
# ════════════════════════════════════════════════════════════════


class CYPInductionEngine:
    """Steady-state CYP induction calculator using the FDA Emax model.

    Usage:
        engine = CYPInductionEngine()
        state = engine.compute(rifampicin, inducer_conc_um=1.0)
        print(state.fold_for("CYP3A4"))             # ≈ 10
        auc_ratio = engine.victim_auc_ratio(state, midazolam)
    """

    # Defensive ceiling on induction fold. In vivo maximum is ~15×
    # (rifampicin/CYP3A4). 50 leaves generous headroom while guarding
    # against malformed Indmax values.
    _FOLD_CEILING: float = 50.0

    def compute(
        self,
        perpetrator: Drug,
        inducer_conc_um: float,
        *,
        site: str = "liver",
    ) -> InductionState:
        """Compute per-CYP steady-state induction factors.

        Args:
            perpetrator: V1 ``Drug`` entity. Only ``perpetrator.induction``
                is consulted.
            inducer_conc_um: Unbound inducer concentration (µM) at the
                enzyme site. Caller decides how this is computed.
            site: ``"liver"`` or ``"gut"`` — documentation only; the
                engine does not use site in the math itself.

        Returns:
            Frozen ``InductionState`` with one contribution per inducible
            CYP. CYPs with no Indmax return ``fold = 1.0``.
        """
        if inducer_conc_um < 0:
            raise ValueError(
                f"inducer_conc_um must be ≥ 0, got {inducer_conc_um}"
            )
        if site not in ("liver", "gut"):
            raise ValueError(f"site must be 'liver' or 'gut', got {site!r}")

        induction = perpetrator.induction
        contributions: list[CYPInductionContribution] = []

        for cyp in INDUCIBLE_CYP_NAMES:
            indmax, indc50 = induction.params_for(cyp)
            fold = self._emax_fold(indmax, indc50, inducer_conc_um)
            contributions.append(
                CYPInductionContribution(
                    cyp=cyp,
                    indmax=indmax,
                    indc50_um=indc50,
                    inducer_conc_um=inducer_conc_um,
                    fold=fold,
                )
            )

        state = InductionState(
            perpetrator_name=perpetrator.name,
            inducer_conc_um=inducer_conc_um,
            site=site,
            contributions=tuple(contributions),
        )

        if state.is_meaningful:
            logger.info(
                "CYP induction: %s @ [I]u=%.4g µM (%s) → max fold = %.2f on %s",
                perpetrator.name, inducer_conc_um, site,
                state.max_fold, state.strongest_cyp,
            )

        return state

    def victim_auc_ratio(
        self,
        state: InductionState,
        victim: Drug,
    ) -> float:
        """Estimate whole-drug AUC fold-change in a victim using the
        FDA static induction model::

            AUCR = 1 / ( Σ_i fm_i · fold_i + (1 − Σ_i fm_i) )

        ``fold_i ≥ 1`` always, so the result satisfies ``AUCR ≤ 1``.
        A result of 1.0 means no meaningful induction overlap.

        Args:
            state: Pre-computed ``InductionState`` for the perpetrator.
            victim: Victim drug whose fm values are read.

        Returns:
            Victim AUC fold-change (0 < value ≤ 1.0 for inducers).
        """
        fm = victim.metabolism
        total_fm = 0.0
        induced_term = 0.0

        for contribution in state.contributions:
            fm_i = fm.fm_for(contribution.cyp)
            if fm_i <= 0.0:
                continue
            total_fm += fm_i
            induced_term += fm_i * max(contribution.fold, 1.0)

        remaining = max(1.0 - total_fm, 0.0)
        denom = induced_term + remaining

        if denom <= 0.0:
            return 1.0
        return 1.0 / denom

    # ── Internal helpers ──

    @classmethod
    def _emax_fold(
        cls,
        indmax: Optional[float],
        indc50_um: Optional[float],
        inducer_conc_um: float,
    ) -> float:
        """Compute the Emax fold-induction value.

        ``fold = 1 + Indmax · [I]u / (IndC50 + [I]u)``

        Guards:
          * ``Indmax`` None or ≤ 0 → fold = 1.0 (not an inducer).
          * ``IndC50`` None or ≤ 0 → fold = 1.0 (malformed profile).
          * ``[I]u`` = 0 → fold = 1.0 (no exposure).
          * Result clipped at ``_FOLD_CEILING``.
        """
        if indmax is None or indmax <= 0.0:
            return 1.0
        if indc50_um is None or indc50_um <= 0.0:
            return 1.0
        if inducer_conc_um <= 0.0:
            return 1.0

        saturation = inducer_conc_um / (indc50_um + inducer_conc_um)
        fold = 1.0 + indmax * saturation
        return min(fold, cls._FOLD_CEILING)
