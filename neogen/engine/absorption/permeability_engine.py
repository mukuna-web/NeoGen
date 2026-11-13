"""Permeability-limited absorption engine — V1 P2 Step 4.

Couples the Step 3 Noyes-Whitney dissolution physics with permeation
across the gut wall and first-pass enterocyte metabolism, producing the
frozen ``AbsorptionProfile`` that Step 5 will feed to the PBPK solver as
an oral forcing function.

Physics pipeline
----------------
For each GI segment j and each instant t:

1. Dissolution source (reused from Step 3):
       dM_dissolved_j/dt_source = noyes_whitney_rate(U_j, D_j, seg, drug, form)

2. Permeation sink (this module):
       C_j = D_j / V_j                         (mg/mL)
       rate_perm_j = Peff · SA_j · C_j         (mg/s → mg/h)

3. Gut-wall first-pass (this module):
       fg_j = 1 − α · cyp3a4_activity_j        if substrate and segment
              1                                 otherwise
       rate_to_portal_j = fg_j · rate_perm_j
       rate_gut_loss_j  = (1 − fg_j) · rate_perm_j

4. Transit (Step 3 convention): first-order with k_j = 1/t_j.

State vector (length 2N + 3):

    y = [U_0..U_{N-1}, D_0..D_{N-1},
         cum_permeated,      # Fa numerator
         cum_portal,         # to portal vein after fg
         excreted]           # fell off the tail of the tract

Mass balance:

    Σ U + Σ D + cum_permeated + excreted ≡ dose_mg     ∀ t

(``cum_gut_metabolized = cum_permeated − cum_portal`` is an algebraic
derivation, not an independent state variable.)

Parameter sourcing from V1 entities
-----------------------------------
- ``drug.peff_jejunum`` is stored in ×10⁻⁴ cm/s units (legacy convention).
  If missing, a BCS-class fallback is applied.
- ``drug.metabolism.cyp3a4 > 0.3`` is treated as a gut-CYP3A4 substrate
  for the fg branch. V1 MVP — drug-specific gut extraction refinements
  are a V2+ concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from neogen.core.entities.drug import BCSClass, Drug
from neogen.core.entities.formulation import Formulation, FormulationType
from neogen.engine.absorption.absorption_profile import AbsorptionProfile
from neogen.engine.absorption.dissolution_engine import (
    noyes_whitney_rate_mg_per_h,
)
from neogen.engine.absorption.gut_transit import GISegment, GutTransit


# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════

# Gut extraction intensity — fraction of permeating drug lost to
# enterocyte CYP3A4 at peak activity (Jejunum_1). Per-segment loss is
# ``α · cyp3a4_activity_j``. The 0.3 default matches the legacy
# ``simulation/dissolution.py`` convention and is tunable per drug in V2.
_DEFAULT_GUT_EXTRACTION_ALPHA: float = 0.3

# Threshold above which a drug is treated as a gut CYP3A4 substrate.
_GUT_CYP3A4_FM_THRESHOLD: float = 0.3

# Fallback Peff values when ``drug.peff_jejunum`` is missing, keyed on BCS
# class (×10⁻⁴ cm/s).
_BCS_PEFF_FALLBACK: Dict[BCSClass, float] = {
    BCSClass.I: 4.0,     # High permeability reference
    BCSClass.II: 4.0,    # High permeability reference
    BCSClass.III: 0.3,   # Low permeability reference
    BCSClass.IV: 0.3,    # Low permeability reference
}

# Colonic Peff scaling by BCS class. BCS II and IV (low solubility)
# drugs have severely reduced colonic absorption because:
#
#   * Colonic fluid volume is ~13-150 mL in the fasted state (vs
#     500-1500 mL in the small intestine), which limits dissolution.
#   * Bile salts are largely absent in colon (concentration drops
#     from ~2 mM in jejunum to ~0.5 mM in ascending colon),
#     removing the micellar solubilisation that BCS II drugs depend
#     on for effective dissolution.
#   * The colonic mucin layer is thicker (~200 um vs ~30 um in
#     jejunum), creating a diffusion barrier for lipophilic drugs.
#   * Colonic epithelium has a smaller effective absorptive surface
#     because villi are absent.
#
# Published PBPK models (Simcyp ADAM, GastroPlus ACAT) implement this
# via drug-class-specific "absorption window" or "effective Peff
# scaling" parameters. V1.2 uses a simple BCS-class scaling that
# drops colonic Peff by 90 % for BCS II and IV, which matches
# Darwich 2010 and Tubic-Grozdanis 2010 observations that
# simvastatin, atorvastatin, itraconazole, and similar lipophilic
# BCS II drugs show near-zero colonic absorption.
#
# BCS I and III drugs pass through colon freely (caffeine, metformin)
# and use the full k_abs from their jejunal Peff.
_COLON_PEFF_FACTOR: Dict[BCSClass, float] = {
    BCSClass.I: 1.0,     # high solubility — unimpeded
    BCSClass.II: 0.1,    # low solubility, high permeability — severe colon penalty
    BCSClass.III: 1.0,   # high solubility — unimpeded (e.g. metformin)
    BCSClass.IV: 0.1,    # low solubility + low permeability — severe colon penalty
}


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def effective_peff_cm_per_s(drug: Drug) -> float:
    """Return the drug's effective jejunal Peff in cm/s.

    Falls back to a BCS-class reference if ``drug.peff_jejunum`` is None.
    The ``Drug`` entity stores this field in ×10⁻⁴ cm/s — we convert.
    """
    if drug.peff_jejunum is not None:
        return drug.peff_jejunum * 1.0e-4
    fallback = _BCS_PEFF_FALLBACK[drug.bcs_class]
    return fallback * 1.0e-4


def is_gut_cyp3a4_substrate(drug: Drug) -> bool:
    """Whether the drug is metabolised meaningfully by gut CYP3A4."""
    return drug.metabolism.cyp3a4 >= _GUT_CYP3A4_FM_THRESHOLD


def segment_fg(
    segment: GISegment, is_substrate: bool, alpha: float
) -> float:
    """Per-segment gut-wall escape fraction ``fg_j ∈ [0, 1]``."""
    if not is_substrate or not segment.has_cyp3a4:
        return 1.0
    loss = alpha * segment.cyp3a4_activity
    return max(0.0, 1.0 - loss)


def _fit_ka_and_tlag(
    t_h: np.ndarray, cum_portal_mg: np.ndarray
) -> Tuple[float, float]:
    """Approximate first-order ka and tlag from a cumulative curve.

    Strategy (simple and robust):
      - ``tlag`` = first time at which ``cum_portal_mg`` reaches 5% of
        its final value.
      - ``t63`` = first time at which ``cum_portal_mg`` reaches 63.2% of
        its final value.
      - ``ka = 1 / max(t63 − tlag, ε)``.

    Returns ``(0, 0)`` if the final portal mass is effectively zero.
    """
    final = float(cum_portal_mg[-1])
    if final <= 1.0e-9:
        return 0.0, 0.0

    tlag_target = 0.05 * final
    t63_target = 0.632 * final

    tlag_idx = int(np.searchsorted(cum_portal_mg, tlag_target))
    t63_idx = int(np.searchsorted(cum_portal_mg, t63_target))
    tlag_idx = min(tlag_idx, len(t_h) - 1)
    t63_idx = min(t63_idx, len(t_h) - 1)

    tlag = float(t_h[tlag_idx])
    t63 = float(t_h[t63_idx])

    delta = max(t63 - tlag, 1.0e-3)
    ka = 1.0 / delta
    return ka, tlag


# ════════════════════════════════════════════════════════════════
# ENGINE
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PermeabilityEngine:
    """Stateless coupled dissolution + absorption simulator.

    The engine itself carries only solver configuration; ``simulate`` is
    a pure function of its arguments.
    """

    solver_method: str = "RK45"
    rtol: float = 1.0e-6
    atol: float = 1.0e-9
    gut_extraction_alpha: float = _DEFAULT_GUT_EXTRACTION_ALPHA

    def simulate(
        self,
        drug: Drug,
        formulation: Formulation,
        dose_mg: float,
        gut_transit: Optional[GutTransit] = None,
        t_end_h: float = 24.0,
        n_points: int = 121,
        gut_wall_mbi_exposures: Optional[Sequence[object]] = None,
        gut_wall_induction_exposures: Optional[Sequence[object]] = None,
    ) -> AbsorptionProfile:
        """Run the coupled dissolution + absorption ODE for one oral dose.

        V1.2 adds the optional ``gut_wall_mbi_exposures`` kwarg: when
        supplied (and non-empty), the per-segment ``fg_j`` array is
        computed via the steady-state enterocyte CYP3A4 pool model in
        :mod:`neogen.engine.ddi.gut_wall_mbi` instead of the V1 static
        formula.

        V1.2 Track C adds ``gut_wall_induction_exposures``: when
        supplied, the enterocyte CYP3A4 pool is amplified above
        baseline via the Emax induction model, REDUCING fg and
        collapsing oral bioavailability for CYP3A4 substrates.

        Passing ``None`` (default) or empty sequences for both
        reproduces V1 behaviour byte-for-byte.
        """
        if dose_mg <= 0.0:
            raise ValueError(f"dose_mg must be > 0 (got {dose_mg})")
        if t_end_h <= 0.0:
            raise ValueError(f"t_end_h must be > 0 (got {t_end_h})")
        if n_points < 2:
            raise ValueError("n_points must be ≥ 2")

        # V1.2: Pre-absorptive degradation — reduce effective dose for
        # acid-labile drugs (e.g. omeprazole loses ~40% in gastric acid
        # before reaching the small intestine).
        effective_dose = dose_mg * (1.0 - drug.pre_absorptive_degradation_fraction)

        if gut_transit is None:
            from neogen.engine.absorption.gut_transit import (
                build_reference_gi_tract,
            )

            gut_transit = build_reference_gi_tract()

        segments: Tuple[GISegment, ...] = gut_transit.segments
        n_seg = len(segments)

        peff_cm_s = effective_peff_cm_per_s(drug)

        # V1.2 Fix 5: P-gp efflux reduces effective permeability for
        # P-gp substrates. This models the apical efflux that limits
        # net transcellular absorption (e.g. digoxin Fabs ≈ 0.70).
        if (
            getattr(drug.transporters, "is_pgp_substrate", False)
            and drug.transporters.pgp_absorption_factor is not None
        ):
            peff_cm_s *= drug.transporters.pgp_absorption_factor

        peff_cm_h = peff_cm_s * 3600.0  # cm/h

        substrate = is_gut_cyp3a4_substrate(drug)

        # V1.2: pick the effective gut extraction coefficient.
        # DEPRECATED: alpha is retained for backward compatibility
        # but is no longer used in the primary Qgut fg path.
        effective_alpha = (
            drug.gut_extraction_alpha
            if drug.gut_extraction_alpha is not None
            else self.gut_extraction_alpha
        )

        # Qgut model: compute whole-liver CLint for use in the
        # flow-limited gut-wall extraction formula (Yang 2007).
        # This replaces the calibrated alpha approach entirely.
        from neogen.engine.hepatic_clearance import HepaticClearanceEngine
        from neogen.engine.ddi.gut_wall_mbi import (
            compute_steady_state_fg_per_segment,
        )

        # Build a minimal physiology for CLint scaling. The
        # PermeabilityEngine doesn't normally have access to
        # physiology, so we use the reference male as default.
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        _physiology = build_reference_male()
        _hce = HepaticClearanceEngine()
        clint_liver_scaled = _hce._scale_clint(drug, _physiology)

        fg_per_segment = compute_steady_state_fg_per_segment(
            drug,
            gut_transit,
            exposures=gut_wall_mbi_exposures or (),
            induction_exposures=gut_wall_induction_exposures or (),
            alpha=effective_alpha,
            clint_liver_scaled_L_per_h=clint_liver_scaled,
        )

        # V1.2: CYP2D6 gut-wall extraction. CYP2D6 is expressed in
        # the gut wall at ~10% of CYP3A4 abundance. For drugs with
        # significant CYP2D6 metabolism, this adds an additional fg
        # reduction beyond the CYP3A4 model (propranolol, metoprolol).
        if drug.gut_cyp2d6_extraction_alpha is not None:
            fm_2d6 = drug.metabolism.cyp2d6
            if fm_2d6 >= 0.30:
                for j, seg in enumerate(segments):
                    fg2d6 = 1.0 - drug.gut_cyp2d6_extraction_alpha * fm_2d6
                    fg_per_segment[j] *= max(0.01, fg2d6)

        transit_k = np.array(
            [1.0 / max(s.transit_time_h, 1.0e-6) for s in segments]
        )

        # Precompute per-segment (Peff × SA / V) in 1/h so the permeation
        # rate is simply ``k_abs_j × D_j``. Units: cm/h × cm² / mL = 1/h
        # (since 1 mL = 1 cm³).
        #
        # V1.2: colonic permeation is scaled down for BCS II / IV drugs
        # via ``_COLON_PEFF_FACTOR``. See the constant's docstring for
        # the physiological rationale (absent bile salts, thicker mucin,
        # low fluid volume). BCS I and III drugs retain full colonic
        # permeability — caffeine and metformin show no regression.
        colon_factor = _COLON_PEFF_FACTOR.get(drug.bcs_class, 1.0)
        k_abs = np.array(
            [
                peff_cm_h
                * seg.surface_area_cm2
                / max(seg.volume_mL, 1.0e-6)
                * (colon_factor if seg.name.lower().startswith("colon") else 1.0)
                for seg in segments
            ]
        )

        # Disintegration delay gating.
        disintegration_h = formulation.disintegration_time_min / 60.0
        needs_disintegration = formulation.formulation_type in (
            FormulationType.IR_TABLET,
            FormulationType.IR_CAPSULE,
        )

        # ── Initial conditions ──────────────────────────────────
        # State = [U_0..U_{N-1}, D_0..D_{N-1}, cum_permeated, cum_portal,
        # excreted]
        y0 = np.zeros(2 * n_seg + 3)

        if formulation.formulation_type is FormulationType.SOLUTION:
            # Pre-dissolved in stomach.
            y0[n_seg + 0] = effective_dose
        elif formulation.formulation_type is FormulationType.ENTERIC_COATED:
            # Coating protects until first SI segment (duodenum in the
            # reference tract, index 1).
            entry_idx = 0
            for idx, seg in enumerate(segments):
                if seg.is_small_intestine:
                    entry_idx = idx
                    break
            y0[entry_idx] = effective_dose
        else:
            y0[0] = effective_dose

        # ── RHS closure ─────────────────────────────────────────

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            dy = np.zeros_like(y)
            undiss = y[:n_seg]
            diss = y[n_seg : 2 * n_seg]

            suppress_stomach = (
                needs_disintegration and t < disintegration_h
            )

            for j, seg in enumerate(segments):
                u = undiss[j]
                d = diss[j]

                if suppress_stomach and j == 0:
                    diss_rate = 0.0
                    perm_rate = 0.0
                    out_u = 0.0
                    out_d = 0.0
                else:
                    # Noyes-Whitney source (reuses Step 3 helper)
                    diss_rate = noyes_whitney_rate_mg_per_h(
                        u, d, seg, drug, formulation
                    )
                    # Permeation sink (linear in D_j via k_abs)
                    perm_rate = k_abs[j] * d
                    # Transit first-order flows
                    out_u = transit_k[j] * u
                    out_d = transit_k[j] * d

                # dU_j/dt = -dissolution - transit out (+ transit in below)
                dy[j] += -diss_rate - out_u
                # dD_j/dt = +dissolution - permeation - transit out (+ in)
                dy[n_seg + j] += diss_rate - perm_rate - out_d

                # Transit into next segment, or excretion if last
                if j < n_seg - 1:
                    dy[j + 1] += out_u
                    dy[n_seg + j + 1] += out_d
                else:
                    dy[-1] += out_u + out_d

                # Cumulative permeated (Fa numerator)
                dy[-3] += perm_rate
                # Cumulative portal (after gut-wall metabolism)
                dy[-2] += perm_rate * fg_per_segment[j]

            return dy

        # ── Integrate ──────────────────────────────────────────
        t_eval = np.linspace(0.0, t_end_h, n_points)
        sol = solve_ivp(
            rhs,
            (0.0, t_end_h),
            y0,
            method=self.solver_method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        y_out = np.clip(sol.y, a_min=0.0, a_max=None)

        undiss_mat = y_out[:n_seg, :].T
        diss_mat = y_out[n_seg : 2 * n_seg, :].T
        cum_permeated = y_out[-3, :]
        cum_portal = y_out[-2, :]
        excreted = y_out[-1, :]
        cum_gut_metabolized = np.maximum(cum_permeated - cum_portal, 0.0)

        # ── Summary metrics ────────────────────────────────────
        fa = float(cum_permeated[-1] / effective_dose)
        fa = min(max(fa, 0.0), 1.0)

        perm_final = float(cum_permeated[-1])
        if perm_final > 1.0e-9:
            fg = float(cum_portal[-1] / perm_final)
        else:
            fg = 1.0
        fg = min(max(fg, 0.0), 1.0)

        f_portal = fa * fg

        ka_eff, tlag = _fit_ka_and_tlag(t_eval, cum_portal)

        # Per-segment portal fraction — attribute each permeation event
        # to its segment by integrating k_abs × D_j × fg_j over time.
        per_seg_portal = np.zeros(n_seg)
        # Trapezoidal integration of the per-segment portal rate.
        for j in range(n_seg):
            rate_j = k_abs[j] * diss_mat[:, j] * fg_per_segment[j]
            per_seg_portal[j] = float(np.trapezoid(rate_j, t_eval))
        portal_total = per_seg_portal.sum()
        if portal_total > 1.0e-9:
            per_seg_fraction = {
                segments[j].name: float(per_seg_portal[j] / portal_total)
                for j in range(n_seg)
            }
        else:
            per_seg_fraction = {seg.name: 0.0 for seg in segments}

        return AbsorptionProfile(
            drug_name=drug.name,
            dose_mg=dose_mg,
            formulation_type=formulation.formulation_type,
            fa=fa,
            fg=fg,
            f_portal=f_portal,
            effective_ka_per_h=ka_eff,
            tlag_h=tlag,
            t_h=t_eval,
            cum_permeated_mg=cum_permeated,
            cum_portal_mg=cum_portal,
            cum_gut_metabolized_mg=cum_gut_metabolized,
            undissolved_mg=undiss_mat,
            dissolved_mg=diss_mat,
            excreted_mg=excreted,
            segment_names=tuple(seg.name for seg in segments),
            per_segment_portal_fraction=per_seg_fraction,
            solver_success=bool(sol.success),
            solver_method=self.solver_method,
        )
