"""Frozen DDI result dataclasses — V1 P3 Step 10.

Dedicated module for the immutable result contracts of the DDI layer.
These were originally defined inline in ``ddi_orchestrator.py`` and are
now separated per the V1 blueprint module layout (§3):

    ddi/
      ddi_result.py         ← this file (frozen result contracts)
      ddi_orchestrator.py   ← DDI assembly + CLintProvider builder
      cyp_inhibition.py     ← Step 7
      cyp_induction.py      ← Step 8
      enzyme_pool_evolver.py ← Step 9

For backwards compatibility, ``ddi_orchestrator.py`` re-exports
everything from this module so existing imports continue to work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from neogen.core.entities.drug import Drug
from neogen.engine.ddi.enzyme_pool_evolver import EnzymePoolTrace


# ════════════════════════════════════════════════════════════════
# PERPETRATOR EXPOSURE
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PerpetratorExposure:
    """Immutable record of one perpetrator's contribution to a DDI.

    Attributes:
        perpetrator: The V1 ``Drug`` causing the interaction.
        steady_state_conc_um: Unbound perpetrator concentration at the
            liver site (uM) during the on-window. Used for both the
            reversible R factor and (implicitly, via the trace) the Step 9
            enzyme pool ODE.
        duration_on_h: How long the perpetrator is actively dosed.
        duration_off_h: Washout window in which the concentration drops
            to zero. Enzymes continue evolving back to baseline.
        pretreatment_h: Hours of perpetrator dosing *before* the victim
            simulation starts. Used to shift the enzyme-pool trace lookup
            so that inducers (rifampicin ~7 day pretreatment) and MBIs
            have reached their steady-state plateau by the time the
            victim is dosed, matching clinical DDI protocols.
        enzyme_trace: Optional Step 9 trace of the enzyme pool over
            ``[0, pretreatment_h + duration_on_h + duration_off_h]``.
            None means the perpetrator has no induction / MBI — only
            reversible Step 7 inhibition applies.
    """

    perpetrator: Drug
    steady_state_conc_um: float
    duration_on_h: float
    duration_off_h: float = 0.0
    pretreatment_h: float = 0.0
    enzyme_trace: Optional[EnzymePoolTrace] = None

    @property
    def total_duration_h(self) -> float:
        return self.duration_on_h + self.duration_off_h

    def conc_at(self, t_h: float) -> float:
        """Instantaneous unbound perpetrator concentration (uM) at time t.

        Returns 0 outside the on-window. The V1 model is a step function;
        V2 will accept a proper PBPK time profile.
        """
        if t_h < 0.0:
            return 0.0
        if t_h > self.duration_on_h:
            return 0.0
        return self.steady_state_conc_um

    def trace_time(self, t_h: float) -> float:
        """Map victim simulation time -> enzyme-pool trace time.

        During the victim run, ``t_h`` starts at 0. The enzyme-pool
        trace was evolved starting at ``t = 0`` under zero perpetrator
        load, so the point that represents the steady-state enzyme
        level at the victim's dosing moment sits at
        ``t = pretreatment_h`` in the trace's clock.
        """
        return t_h + self.pretreatment_h


# ════════════════════════════════════════════════════════════════
# CLINT DIAGNOSTIC BREAKDOWN
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CLintBreakdown:
    """Diagnostic snapshot of the CLint decomposition at a single time.

    Attributes:
        time_h: Time point.
        clint_base_L_per_h: The victim's baseline whole-liver CLint.
        clint_eff_L_per_h: The effective CLint after DDI scaling.
        scaling: Dimensionless multiplier ``CLint_eff / CLint_base``.
            Equal to ``cyp_scaling x uptake_correction``.
        cyp_scaling: CYP-mediated portion of the multiplier,
            ``(1 - Σfm) + Σ fm · s_cyp(t)``.
        uptake_correction: OATP1B1 uptake-inhibition portion of the
            multiplier (1.0 when the victim is not an OATP1B1 substrate
            or no perpetrator inhibits OATP1B1).
        per_cyp_scaling: Per-CYP ``s_cyp = Π_k fold_k / R_k`` map, useful
            for auditing which mechanism dominates at this time point.
    """

    time_h: float
    clint_base_L_per_h: float
    clint_eff_L_per_h: float
    scaling: float
    cyp_scaling: float
    uptake_correction: float
    per_cyp_scaling: Tuple[Tuple[str, float], ...]


# ════════════════════════════════════════════════════════════════
# DDI SUMMARY (for SimulationResultV2)
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DDISummary:
    """Frozen summary of a DDI simulation for inclusion in SimulationResultV2.

    Attributes:
        victim_name: Name of the victim drug.
        perpetrator_names: Names of all perpetrator drugs.
        auc_fold_change: Predicted AUC ratio (DDI / baseline).
        mechanism: Primary mechanism description (e.g. "CYP3A4 inhibition").
        severity: Clinical severity tier ("minor", "moderate", "major").
        clint_scaling_at_dose_time: CLint multiplier at the victim's
            Tmax, useful for quick auditing.
    """

    victim_name: str
    perpetrator_names: Tuple[str, ...]
    auc_fold_change: float
    mechanism: str
    severity: str
    clint_scaling_at_dose_time: Optional[float] = None

    @property
    def n_perpetrators(self) -> int:
        return len(self.perpetrator_names)
