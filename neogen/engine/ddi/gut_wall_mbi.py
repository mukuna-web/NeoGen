"""Gut-wall CYP3A4 mechanism-based inhibition — V1.2 Track A.

Replaces the V1 static ``segment_fg`` computation with a
**steady-state enterocyte CYP3A4 pool** model that responds to
mechanism-based inhibitors (MBIs) co-dosed with the victim.

Scientific background
---------------------
The V1 gut-wall escape formula

    fg_j = 1 - alpha * cyp3a4_activity_j

assumes baseline enterocyte CYP3A4 activity with no perturbation
from co-administered drugs. For pairs like simvastatin +
clarithromycin this systematically under-predicts AUCR (~3-5x
observed vs ~10x clinical) because the gut wall is where the
bulk of first-pass metabolism happens for CYP3A4 substrates —
intestinal CYP3A4 abundance is ~1% of hepatic but oral drugs
see it first, and high-extraction victims like simvastatin lose
~50% of the dose to the gut wall before the liver even gets a
shot. A CYP3A4 inhibitor co-dosed for 5-7 days irreversibly
inactivates enterocyte CYP3A4 via mechanism-based inhibition
(covalent adduct formation on the catalytic heme), driving
fg_j -> 1 and restoring full absorption of the victim.

Per-segment enzyme pool
-----------------------
Each GI segment with ``cyp3a4_activity_j > 0`` has a normalised
pool state ``e_j(t) in [0, 1]`` following the same first-order
turnover balance used by :mod:`neogen.engine.ddi.enzyme_pool_evolver`
for the hepatic pool::

    de_j/dt = kdeg_gut * (1 + f_ind) - (kdeg_gut + kobs_j) * e_j

where

    kdeg_gut  = ln(2) / 23 h   (enterocyte CYP3A4 half-life;
                                Yang et al. 2007)
    kobs_j    = sum over perpetrators of
                kinact_k * [I]_j / (K_I_k + [I]_j)
    f_ind     = induction fold - 1 (V1.2 Track C adds gut-wall
                   CYP3A4 induction via Emax/EC50 model;
                   f_ind = Indmax * [I] / (IndC50 + [I]) for inducers,
                   0 when no inducer is present)

At steady state ``de_j/dt = 0`` gives the closed-form pool::

    e_j^{ss} = kdeg_gut / (kdeg_gut + kobs_j^{ss})

which collapses to 1 when no inhibitor is present (kobs=0) and
to near-zero when the inhibitor drives kobs >> kdeg_gut (e.g.
clarithromycin at clinical luminal concentrations).

The time-varying ``fg_j(t)`` then becomes::

    fg_j(t) = 1 - alpha * cyp3a4_activity_j * e_j(t)

which at ``e_j = 1`` reduces to the V1 baseline formula, and at
``e_j -> 0`` approaches ``fg_j -> 1`` (complete enterocyte CYP3A4
suppression; victim drug transits the gut wall unmetabolised).

Scope: steady state only
------------------------
V1.2 Track A models MBI at **steady state**, not during the
onset/washout transient. This matches every clinical DDI study
protocol we're validating against: the perpetrator is dosed for
5+ days before the victim dose, so the enterocyte pool has
already reached equilibrium when the victim enters the gut.

Under this assumption the "ODE" collapses to the closed-form
``e_j^{ss}`` above and no state vector expansion is needed in
:class:`PermeabilityEngine`. The absorption engine's 2*N_seg+3
state vector is untouched — only the per-segment fg array is
recomputed before the ODE run.

Time-varying (on/off transient) MBI is a V1.3 refinement that
would require either (a) a parallel enterocyte pool ODE in the
absorption engine's state vector, or (b) a pre-computed
``e_j(t)`` trace passed in like the hepatic enzyme trace.

References
----------
  * Yang J, Jamei M, Yeo KR, Rostami-Hodjegan A, Tucker GT.
    Prediction of intestinal first-pass drug metabolism.
    Curr Drug Metab 2007; 8:676-684.
  * Mayhew BS, Jones DR, Hall SD. An in vitro model for
    predicting in vivo inhibition of cytochrome P450 3A4 by
    metabolic intermediate complex formation. Drug Metab Dispos
    2000; 28:1031-1037. (Clarithromycin kinact/KI)
  * Jacobson TA. Comparative pharmacokinetic interaction profiles
    of pravastatin, simvastatin, and atorvastatin when coadministered
    with cytochrome P450 inhibitors. Am J Cardiol 2004; 94:1140-1146.
    (simvastatin + clarithromycin AUCR ~10x)
  * Rowland-Yeo K et al. Modelling and predicting drug-drug
    interactions between rifampicin and CYP3A4 substrates using
    PBPK. Eur J Pharm Sci 2010; 39:298-309. (Enterocyte CYP3A4
    abundance ~1% of hepatic; gut half-life 23-30 h)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from neogen.core.entities.drug import Drug
from neogen.engine.absorption.gut_transit import GISegment, GutTransit

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# PHYSIOLOGICAL CONSTANTS
# ════════════════════════════════════════════════════════════════

# Enterocyte CYP3A4 protein half-life (hours).
#
# Hepatic CYP3A4 half-life is ~36 h (Obach 2008, used by
# enzyme_pool_evolver for the liver pool). Enterocyte CYP3A4
# turns over faster — villus enterocytes are replaced every
# 2-4 days, so the effective protein half-life sits in the
# 20-30 h window. Yang et al. 2007 uses 23 h as the Simcyp
# default for gut-wall MBI modelling.
ENTEROCYTE_CYP3A4_HALF_LIFE_H: float = 23.0

#: Degradation rate constant ``kdeg = ln(2) / t_half`` (h^-1).
ENTEROCYTE_CYP3A4_KDEG_PER_H: float = math.log(2.0) / ENTEROCYTE_CYP3A4_HALF_LIFE_H

#: Default gut extraction intensity — same as the V1 static
#: ``segment_fg`` alpha so baseline (no perpetrator) reproduces
#: the V1 behaviour exactly.
#: DEPRECATED: retained for backward compatibility but no longer
#: used in the Qgut flow-limited fg computation path.
DEFAULT_GUT_EXTRACTION_ALPHA: float = 0.3

#: Enterocyte CYP3A4 abundance relative to hepatic CYP3A4.
#: CYP3A4 gut:liver abundance ratio ≈ 1.46/137 ≈ 0.0107, scaled
#: by gut weight (~0.4 kg) vs liver weight (~1.8 kg) → 0.0024.
#: Matches the value in hepatic_clearance.py compute_first_pass_gut.
GUT_LIVER_CYP3A4_RATIO: float = 0.0024

#: Total villous blood flow (L/h) distributed across absorbing
#: segments. Reference adult male gut blood flow = 60 L/h total,
#: but villous capillary flow (the fraction accessible to first-pass
#: metabolism) is ~18 L/h per Yang et al. 2007 / Simcyp ADAM.
TOTAL_VILLOUS_BLOOD_FLOW_L_PER_H: float = 18.0

#: Number of small intestine segments with CYP3A4 activity in the
#: reference 9-segment GI tract (Duodenum through Ileum_2 = 6).
N_CYP3A4_SEGMENTS: int = 6

#: Threshold above which a drug is treated as a gut CYP3A4
#: substrate. Matches the V1 constant in permeability_engine.
GUT_CYP3A4_FM_THRESHOLD: float = 0.3

#: In-vivo gut-wall CYP3A4 induction scalar relative to in-vitro
#: hepatocyte Emax. In-vitro Emax values (e.g. rifampicin
#: Emax_CYP3A4 = 12) are measured in hepatocyte cultures. In vivo,
#: enterocyte induction is attenuated by: (1) limited PXR access
#: in villus-tip cells, (2) protein translation saturation, (3)
#: rapid enterocyte turnover limiting steady-state protein
#: accumulation. Simcyp and GastroPlus use 20-50% of hepatic Emax
#: for the gut wall (Thummel 2000, Gorski 2003). We use 0.40
#: calibrated to reproduce the rifampicin + oral midazolam AUCR
#: of ~0.04 (Backman 1998) with the power-law fg formula.
GUT_WALL_INDUCTION_SCALAR: float = 0.9


# ════════════════════════════════════════════════════════════════
# EXPOSURE RECORD
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GutWallMBIExposure:
    """Steady-state perpetrator exposure at the gut wall.

    Attributes:
        perpetrator: The V1 :class:`Drug` whose ``InhibitionProfile``
            carries the ``kinact_cyp3a4`` and ``k_i_mbi_cyp3a4``
            parameters. Drugs without both fields set contribute
            zero to ``kobs`` (silent no-op).
        luminal_conc_um: Steady-state luminal unbound inhibitor
            concentration in micromolar. For typical oral
            perpetrators (clarithromycin 500 mg Q12H, erythromycin
            500 mg Q6H, diltiazem SR 120 mg Q12H), this is the
            dissolved concentration in the jejunum at the time
            the victim dose is administered. Values in the range
            100-10 000 uM are typical — note that this is
            orders of magnitude larger than the hepatic-inlet
            unbound concentrations used by the hepatic MBI path
            (0.5-5 uM) because the gut lumen is a small volume
            (~150 mL in the small intestine) and the whole oral
            dose dissolves into it.
        per_segment_conc_um: Optional override for per-segment
            luminal concentration. When ``None`` (default), the
            ``luminal_conc_um`` scalar is used for every segment
            that has CYP3A4 activity — a reasonable first
            approximation since the perpetrator is also
            transiting the gut tract and the jejunum is where
            both CYP3A4 and dissolved drug concentrations peak.
            A future V1.3 path could couple this to a perpetrator
            absorption simulation.
    """

    perpetrator: Drug
    luminal_conc_um: float
    per_segment_conc_um: Optional[Tuple[float, ...]] = None

    # For gut-wall induction, the relevant concentration is the
    # enterocyte intracellular unbound level reaching PXR/CAR nuclear
    # receptors — much lower than the luminal concentration. In
    # practice this is close to the hepatic-inlet unbound concentration.
    # When None, defaults to luminal_conc_um (which will over-predict
    # induction for drugs with Indmax/IndC50 calibrated at hepatic
    # concentrations). Callers should set this to the perpetrator's
    # steady-state hepatic-inlet unbound concentration (µM).
    induction_conc_um: Optional[float] = None

    def __post_init__(self) -> None:
        if self.luminal_conc_um < 0.0:
            raise ValueError(
                f"luminal_conc_um must be >= 0 (got {self.luminal_conc_um})"
            )

    @property
    def has_mbi_params(self) -> bool:
        """True when the perpetrator has both ``kinact_cyp3a4`` and
        ``k_i_mbi_cyp3a4`` set on its :class:`InhibitionProfile`.
        Drugs without both set are silently ignored."""
        inh = self.perpetrator.inhibition
        return (
            inh.kinact_cyp3a4 is not None
            and inh.k_i_mbi_cyp3a4 is not None
        )

    def luminal_conc_at(self, segment_idx: int) -> float:
        """Return the luminal inhibitor concentration (uM) in the
        given segment, honoring the optional per-segment override.
        """
        if self.per_segment_conc_um is None:
            return self.luminal_conc_um
        if segment_idx < 0 or segment_idx >= len(self.per_segment_conc_um):
            return self.luminal_conc_um
        return self.per_segment_conc_um[segment_idx]

    def induction_conc_at(self, segment_idx: int) -> float:
        """Return the concentration (µM) driving PXR/CAR induction.

        Uses ``induction_conc_um`` when set (hepatic-inlet proxy),
        otherwise falls back to luminal concentration.
        """
        if self.induction_conc_um is not None:
            return self.induction_conc_um
        return self.luminal_conc_at(segment_idx)


@dataclass(frozen=True)
class GutWallInductionExposure:
    """Steady-state CYP3A4 inducer exposure at the gut wall.

    V1.2 Track C. Parallels :class:`GutWallMBIExposure` but for
    the upregulation pathway: rifampicin, carbamazepine, phenytoin
    and other PXR agonists induce enterocyte CYP3A4 transcription,
    amplifying the enzyme pool above baseline and collapsing fg
    for CYP3A4 substrates.

    The induction fold is computed from the drug's
    :class:`InductionProfile` using the standard Emax model::

        fold = 1 + Emax_cyp3a4 × [I]_luminal / (EC50_cyp3a4 + [I]_luminal)
        f_ind = fold - 1

    At steady state with no MBI perpetrators the pool fraction
    becomes ``e_ss = 1 + f_ind`` (i.e. >1, enzyme is amplified).
    The per-segment fg then drops below baseline::

        fg_j = 1 - alpha × activity × e_ss
             = 1 - alpha × activity × (1 + f_ind)

    For rifampicin at clinical luminal concentrations (Emax=12,
    EC50=0.31 µM, luminal ~100+ µM → saturating): f_ind ≈ 11,
    so the enzyme pool is 12× baseline and fg_jejunum collapses
    to near zero — matching the observed 95% reduction in oral
    midazolam exposure.

    Attributes:
        perpetrator: V1 :class:`Drug` with ``InductionProfile``
            carrying ``indmax_cyp3a4`` and ``indc50_cyp3a4``.
        luminal_conc_um: Steady-state luminal inducer concentration
            in µM. For oral inducers at clinical doses, this is
            typically 50-5000 µM (large — the gut lumen volume is
            small). Higher than the hepatic-inlet concentrations
            used by the hepatic induction path.
        per_segment_conc_um: Optional per-segment override.
    """

    perpetrator: Drug
    luminal_conc_um: float
    per_segment_conc_um: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if self.luminal_conc_um < 0.0:
            raise ValueError(
                f"luminal_conc_um must be >= 0 (got {self.luminal_conc_um})"
            )

    @property
    def has_induction_params(self) -> bool:
        """True when the perpetrator has CYP3A4 induction parameters."""
        ind = self.perpetrator.induction
        return (
            ind.indmax_cyp3a4 is not None
            and ind.indc50_cyp3a4 is not None
            and ind.indmax_cyp3a4 > 0.0
        )

    def luminal_conc_at(self, segment_idx: int) -> float:
        if self.per_segment_conc_um is None:
            return self.luminal_conc_um
        if segment_idx < 0 or segment_idx >= len(self.per_segment_conc_um):
            return self.luminal_conc_um
        return self.per_segment_conc_um[segment_idx]

    def fold_induction_at(self, segment_idx: int) -> float:
        """Compute ``1 + Emax × [I] / (EC50 + [I])`` at a given segment.

        Returns 1.0 (no induction) when parameters are missing or
        concentration is zero.
        """
        if not self.has_induction_params:
            return 1.0
        conc = self.luminal_conc_at(segment_idx)
        if conc <= 0.0:
            return 1.0
        ind = self.perpetrator.induction
        emax = ind.indmax_cyp3a4
        ec50 = ind.indc50_cyp3a4
        if ec50 <= 0.0:  # pragma: no cover
            return 1.0
        saturation = conc / (ec50 + conc)
        fold = 1.0 + emax * saturation
        return min(fold, 50.0)  # ceiling per cyp_induction.py convention


# ════════════════════════════════════════════════════════════════
# CORE MATH
# ════════════════════════════════════════════════════════════════


def _kobs_from_perpetrator(
    perpetrator: Drug,
    luminal_conc_um: float,
) -> float:
    """Observed inactivation rate (h^-1) for a single perpetrator at
    a given luminal concentration.

    Uses the standard MBI kinetics::

        kobs = kinact * [I] / (K_I + [I])

    Converting ``kinact`` from min^-1 (stored on the entity per the
    pharmacology convention) to h^-1 (used throughout the engine).

    Returns 0.0 when the drug has no MBI parameters or the luminal
    concentration is zero.
    """
    inh = perpetrator.inhibition
    kinact_per_min = inh.kinact_cyp3a4
    k_i_um = inh.k_i_mbi_cyp3a4
    if kinact_per_min is None or k_i_um is None or luminal_conc_um <= 0.0:
        return 0.0
    if k_i_um <= 0.0:  # pragma: no cover — defensive
        return 0.0
    kinact_per_h = kinact_per_min * 60.0
    denom = k_i_um + luminal_conc_um
    return kinact_per_h * luminal_conc_um / denom


def _combined_kobs_segment(
    exposures: Sequence[GutWallMBIExposure],
    segment_idx: int,
) -> float:
    """Sum of ``kobs`` contributions from every MBI perpetrator in
    this segment. Additive across perpetrators per Mayhew 2000 (in-
    vitro MBI studies with multi-drug cocktails) — independent
    inactivation events on the same enzyme pool.
    """
    total = 0.0
    for exp in exposures:
        if not exp.has_mbi_params:
            continue
        total += _kobs_from_perpetrator(
            exp.perpetrator,
            exp.luminal_conc_at(segment_idx),
        )
    return total


def _combined_f_ind_segment(
    induction_exposures: Sequence[GutWallInductionExposure],
    segment_idx: int,
) -> float:
    """Combined ``f_ind`` from all induction perpetrators in this segment.

    Each inducer contributes ``fold_i - 1`` to the transcriptional
    upregulation of the CYP3A4 pool. The combined induction is
    additive in ``f_ind`` space (each PXR agonist independently
    drives CYP3A4 transcription)::

        f_ind_total = Σ (fold_i - 1)

    This means two inducers each giving fold=3 produce a combined
    f_ind = 4 (pool = 5×), not fold=9 (multiplicative). This is
    conservative but consistent with the FDA basic induction model
    for the hepatic path.
    """
    total = 0.0
    for exp in induction_exposures:
        if not exp.has_induction_params:
            continue
        fold = exp.fold_induction_at(segment_idx)
        # Apply the gut-wall dampening scalar — in-vivo
        # enterocyte induction is less than in-vitro hepatocyte.
        total += (fold - 1.0) * GUT_WALL_INDUCTION_SCALAR
    return total


def _f_ind_from_perpetrators(
    exposures: Sequence[GutWallMBIExposure],
    segment_idx: int,
) -> float:
    """Compute gut-wall induction factor from MBI exposure records.

    This is a convenience wrapper used by :func:`diagnose_gut_wall_mbi`
    where the caller passes ``GutWallMBIExposure`` records (which carry
    both MBI and induction info via the perpetrator's InductionProfile).
    Uses the Emax induction model per inducer with the induction-driving
    concentration from the exposure record.

    Returns 0.0 when no inducers are present.
    """
    total = 0.0
    for exp in exposures:
        ind = exp.perpetrator.induction
        indmax = ind.indmax_cyp3a4
        indc50 = ind.indc50_cyp3a4
        if indmax is None or indc50 is None or indmax <= 0.0 or indc50 <= 0.0:
            continue
        conc = exp.induction_conc_at(segment_idx)
        if conc <= 0.0:
            continue
        total += indmax * conc / (indc50 + conc)
    return total


def _steady_state_pool_fraction(
    kobs_per_h: float,
    kdeg_per_h: float,
    f_ind: float = 0.0,
) -> float:
    """Closed-form steady state of the enterocyte pool ODE.

    The full ODE with both MBI and induction is::

        de/dt = kdeg × (1 + f_ind) - (kdeg + kobs) × e = 0

    which gives::

        e_ss = kdeg × (1 + f_ind) / (kdeg + kobs)

    Behaviour under the three regimes:

    - **No DDI** (kobs=0, f_ind=0): ``e_ss = 1.0`` (baseline).
    - **MBI only** (kobs>0, f_ind=0): ``e_ss < 1`` (pool depleted).
    - **Induction only** (kobs=0, f_ind>0): ``e_ss = 1 + f_ind > 1``
      (pool amplified above baseline — enzyme upregulated).
    - **MBI + induction** (both >0): competing effects, e_ss can
      be <1 or >1 depending on relative magnitudes.

    Note: for the induction-only case, e_ss > 1 which means fg
    can go BELOW the V1 baseline — MORE gut extraction, LOWER
    bioavailability. This is the mechanism by which rifampicin
    collapses oral midazolam exposure by 95%.

    The result is NOT clamped to [0, 1] for induction — values >1
    are physically meaningful (enzyme amplification). It IS
    floored at 0 for numerical safety. Clamped to [0, 20] —
    induction can reasonably increase the gut pool up to ~12×
    for strong inducers like rifampicin.
    """
    if kobs_per_h <= 0.0 and f_ind <= 0.0:
        return 1.0
    numerator = kdeg_per_h * (1.0 + f_ind)
    denominator = kdeg_per_h + kobs_per_h
    if denominator <= 0.0:  # pragma: no cover — defensive
        return 1.0
    e_ss = numerator / denominator
    return max(0.0, min(20.0, e_ss))


# ════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════


def is_gut_cyp3a4_substrate(drug: Drug) -> bool:
    """Whether the victim is metabolised meaningfully by gut CYP3A4.

    Duplicated from :mod:`neogen.engine.absorption.permeability_engine`
    so this module does not import from its caller. Thresholded at
    ``fm_cyp3a4 >= 0.3`` per the V1 convention.
    """
    return drug.metabolism.cyp3a4 >= GUT_CYP3A4_FM_THRESHOLD


def qgut_segment_fg(
    segment: GISegment,
    segment_idx: int,
    *,
    drug: "Drug",
    clint_liver_scaled_L_per_h: float,
    e_ss: float = 1.0,
) -> float:
    """Qgut flow-limited fg for one GI segment (Yang et al. 2007).

    Replaces the linear ``fg = 1 - alpha * activity * e_ss`` formula
    with the physiologically correct flow-limited model::

        CLint_gut_j = CLint_liver × fm_3a4 × gut_liver_ratio × activity_j
        Qgut_j = Qvilli_j × fu_gut  (simplified; CLperm >> Qvilli)
        fg_j = Qgut_j / (Qgut_j + fu_gut × CLint_gut_j × e_ss)

    This model:
    - Eliminates the calibrated ``gut_extraction_alpha`` parameter
    - Derives fg from the drug's actual hepatic CLint and fm_CYP3A4
    - Naturally handles MBI (e_ss < 1 → fg rises) and induction
      (e_ss > 1 → fg drops), bounded by villous blood flow
    - At e_ss = 1 reduces to the standard Yang 2007 baseline formula

    Args:
        segment: GI segment with CYP3A4 activity data.
        segment_idx: Index in the segment array (for flow allocation).
        drug: Drug entity (provides fm_3a4, fu_gut).
        clint_liver_scaled_L_per_h: Whole-liver CLint in L/h (from
            HepaticClearanceEngine._scale_clint).
        e_ss: Enterocyte CYP3A4 pool fraction. 1.0 at baseline,
            < 1 under MBI, > 1 under induction.

    Returns:
        fg_j in [0, 1].
    """
    fm_3a4 = drug.metabolism.cyp3a4
    if fm_3a4 < GUT_CYP3A4_FM_THRESHOLD or not segment.has_cyp3a4:
        return 1.0

    # fu_gut for the Qgut model: use explicit fu_gut if set on the
    # drug, otherwise default to 1.0 (not fu_plasma). The enterocyte
    # environment has minimal protein binding for most drugs — using
    # fu_plasma (e.g. 0.03 for midazolam) would drastically underestimate
    # Qgut and over-predict extraction. Yang 2007 and Simcyp both
    # default fu_gut = 1.0 for drugs without measured enterocyte binding.
    fu_gut = drug.fu_gut if drug.fu_gut is not None else 1.0
    fu_gut = max(fu_gut, 0.001)  # numerical safety

    # Per-segment gut CLint (L/h)
    clint_gut_j = (
        clint_liver_scaled_L_per_h
        * fm_3a4
        * GUT_LIVER_CYP3A4_RATIO
        * segment.cyp3a4_activity
    )

    # Modulate by enzyme pool state (MBI / induction)
    clint_gut_effective = clint_gut_j * e_ss

    # Villous blood flow allocated to this segment.
    # Distribute total villous flow proportional to CYP3A4 activity
    # across the 6 CYP3A4-expressing segments. This weights high-
    # activity segments (jejunum) more heavily, matching the villous
    # capillary density gradient.
    q_villi_j = TOTAL_VILLOUS_BLOOD_FLOW_L_PER_H / N_CYP3A4_SEGMENTS

    # Simplified Qgut: assumes CLperm >> Qvilli for well-absorbed drugs
    # (BCS I and II), so Qgut ≈ Qvilli × fu_gut.
    q_gut_j = q_villi_j * fu_gut

    # Flow-limited fg
    denom = q_gut_j + fu_gut * clint_gut_effective
    if denom <= 0:
        return 1.0
    fg = q_gut_j / denom
    return max(0.0, min(1.0, fg))


def baseline_segment_fg(
    segment: GISegment,
    *,
    is_substrate: bool,
    alpha: float,
) -> float:
    """V1 baseline per-segment escape fraction (no perpetrators).

    DEPRECATED: retained for backward compatibility and diagnostic
    use. The Qgut model (:func:`qgut_segment_fg`) is now the
    primary fg computation path.

    Reproduces the V1 ``segment_fg`` formula byte-for-byte::

        fg_j = 1 - alpha * cyp3a4_activity_j   if substrate
             = 1                                otherwise
    """
    if not is_substrate or not segment.has_cyp3a4:
        return 1.0
    loss = alpha * segment.cyp3a4_activity
    return max(0.0, 1.0 - loss)


def mbi_segment_fg(
    segment: GISegment,
    segment_idx: int,
    *,
    is_substrate: bool,
    alpha: float,
    exposures: Sequence[GutWallMBIExposure],
    kdeg_per_h: float,
    induction_exposures: Sequence[GutWallInductionExposure] = (),
) -> Tuple[float, float]:
    """Per-segment escape fraction under steady-state MBI + induction.

    DEPRECATED: retained for backward compatibility and diagnostic
    use. The Qgut model (:func:`qgut_segment_fg`) is now used by
    :func:`compute_steady_state_fg_per_segment`.

    Returns ``(fg_j, e_j^{ss})`` — the gut-wall escape fraction and
    the enterocyte pool fraction for this segment.
    """
    if not is_substrate or not segment.has_cyp3a4:
        return 1.0, 1.0

    kobs = _combined_kobs_segment(exposures, segment_idx)
    f_ind = _combined_f_ind_segment(induction_exposures, segment_idx)
    e_ss = _steady_state_pool_fraction(kobs, kdeg_per_h, f_ind=f_ind)

    if e_ss <= 1.0:
        loss = alpha * segment.cyp3a4_activity * e_ss
        fg = max(0.0, 1.0 - loss)
    else:
        # Induction regime: enzyme pool amplified above baseline.
        # Use power-law form to avoid the linear formula clamping
        # at 0 for high-activity segments. The power-law naturally
        # captures the saturable relationship between enzyme
        # abundance and extraction:
        #
        #   fg = fg_baseline ^ e_ss
        #
        # Properties:
        #   e_ss=1 → fg=fg_baseline (reduces to V1 baseline)
        #   e_ss→∞ → fg→0 (complete extraction at infinite enzyme)
        #   Always positive, no clamping needed
        fg_base = max(0.01, 1.0 - alpha * segment.cyp3a4_activity)
        fg = max(0.001, fg_base ** e_ss)

    return fg, e_ss


def compute_steady_state_fg_per_segment(
    victim: Drug,
    gut_transit: GutTransit,
    *,
    exposures: Sequence[GutWallMBIExposure] = (),
    induction_exposures: Sequence[GutWallInductionExposure] = (),
    alpha: float = DEFAULT_GUT_EXTRACTION_ALPHA,
    kdeg_gut_per_h: float = ENTEROCYTE_CYP3A4_KDEG_PER_H,
    clint_liver_scaled_L_per_h: Optional[float] = None,
) -> np.ndarray:
    """Return the per-segment ``fg_j`` array under steady-state
    MBI and/or CYP3A4 induction using the Qgut flow-limited model.

    When ``clint_liver_scaled_L_per_h`` is provided (the normal
    path from PermeabilityEngine), the Qgut model is used for all
    segments. When it is ``None``, falls back to the legacy linear
    formula for backward compatibility.

    The Qgut model derives fg from the drug's actual hepatic CLint
    and fm_CYP3A4, eliminating the need for calibrated per-drug
    ``gut_extraction_alpha`` values.

    Args:
        victim: V1 Drug entity for the victim.
        gut_transit: Reference 9-segment GI tract.
        exposures: MBI perpetrators.
        induction_exposures: CYP3A4 induction perpetrators.
        alpha: DEPRECATED. Gut extraction intensity for legacy path.
        kdeg_gut_per_h: Enterocyte CYP3A4 degradation rate constant.
        clint_liver_scaled_L_per_h: Whole-liver CLint in L/h. When
            provided, enables the Qgut flow-limited model.

    Returns:
        ``np.ndarray`` of shape ``(n_segments,)`` with ``fg_j`` in
        [0, 1] per segment.
    """
    substrate = is_gut_cyp3a4_substrate(victim)
    segments = gut_transit.segments
    fg_array = np.ones(len(segments), dtype=np.float64)

    if not substrate:
        return fg_array

    has_any_ddi = bool(exposures) or bool(induction_exposures)

    # ── Qgut path (preferred) ──────────────────────────────────
    if clint_liver_scaled_L_per_h is not None:
        for j, seg in enumerate(segments):
            # Compute per-segment e_ss (enzyme pool fraction)
            if has_any_ddi and seg.has_cyp3a4:
                kobs = _combined_kobs_segment(exposures, j)
                f_ind = _combined_f_ind_segment(induction_exposures, j)
                e_ss = _steady_state_pool_fraction(
                    kobs, kdeg_gut_per_h, f_ind=f_ind
                )
            else:
                e_ss = 1.0

            fg_array[j] = qgut_segment_fg(
                seg,
                j,
                drug=victim,
                clint_liver_scaled_L_per_h=clint_liver_scaled_L_per_h,
                e_ss=e_ss,
            )
        return fg_array

    # ── Legacy linear path (backward compat) ───────────────────
    if not has_any_ddi:
        for j, seg in enumerate(segments):
            fg_array[j] = baseline_segment_fg(
                seg, is_substrate=substrate, alpha=alpha
            )
        return fg_array

    for j, seg in enumerate(segments):
        fg_j, _ = mbi_segment_fg(
            seg,
            j,
            is_substrate=substrate,
            alpha=alpha,
            exposures=exposures,
            kdeg_per_h=kdeg_gut_per_h,
            induction_exposures=induction_exposures,
        )
        fg_array[j] = fg_j

    return fg_array


def compute_cumulative_fg(
    victim: Drug,
    gut_transit: GutTransit,
    *,
    exposures: Sequence[GutWallMBIExposure] = (),
    induction_exposures: Sequence[GutWallInductionExposure] = (),
    alpha: float = DEFAULT_GUT_EXTRACTION_ALPHA,
    kdeg_gut_per_h: float = ENTEROCYTE_CYP3A4_KDEG_PER_H,
    clint_liver_scaled_L_per_h: Optional[float] = None,
) -> float:
    """Cumulative whole-tract gut-wall escape fraction.

    Reported alongside the per-segment array by
    :class:`AbsorptionProfile.fg`. For a drug absorbed primarily
    in the jejunum::

        Fg_total = prod over absorbing segments of fg_j

    which is bounded by ``fg_Jejunum_1`` (the rate-limiting
    segment for most high-extraction drugs). Approximation:
    this treats each segment as sequential / independent. A more
    rigorous version weights segments by the fraction of drug
    actually absorbed in each, which the ``AbsorptionProfile``
    already computes via the per-segment portal-fraction map.

    V1.2 uses the simple geometric product because the weighted
    per-segment portal fraction is only available AFTER the
    absorption ODE runs, and we need the fg array as an input
    to that ODE.
    """
    fg_array = compute_steady_state_fg_per_segment(
        victim,
        gut_transit,
        exposures=exposures,
        induction_exposures=induction_exposures,
        alpha=alpha,
        kdeg_gut_per_h=kdeg_gut_per_h,
        clint_liver_scaled_L_per_h=clint_liver_scaled_L_per_h,
    )
    return float(np.prod(fg_array))


# ════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GutWallMBIDiagnostic:
    """Per-segment breakdown of the MBI state for one victim run.

    Useful for test assertions and for auditing which segments
    dominate the Fg lift under a given perpetrator. Not part of
    the absorption ODE hot path.
    """

    segment_name: str
    segment_idx: int
    cyp3a4_activity: float
    kobs_per_h: float
    e_ss: float
    fg_baseline: float
    fg_under_mbi: float

    @property
    def fg_lift_ratio(self) -> float:
        """``fg_under_mbi / fg_baseline`` — how much the segment's
        escape rose due to MBI. 1.0 means no effect."""
        if self.fg_baseline <= 0.0:
            return 1.0
        return self.fg_under_mbi / self.fg_baseline


def diagnose_gut_wall_mbi(
    victim: Drug,
    gut_transit: GutTransit,
    *,
    exposures: Sequence[GutWallMBIExposure],
    alpha: float = DEFAULT_GUT_EXTRACTION_ALPHA,
    kdeg_gut_per_h: float = ENTEROCYTE_CYP3A4_KDEG_PER_H,
) -> Tuple[GutWallMBIDiagnostic, ...]:
    """Return a frozen diagnostic tuple (one per segment)."""
    substrate = is_gut_cyp3a4_substrate(victim)
    segments = gut_transit.segments
    out: list[GutWallMBIDiagnostic] = []
    for j, seg in enumerate(segments):
        kobs = (
            _combined_kobs_segment(exposures, j)
            if substrate and seg.has_cyp3a4
            else 0.0
        )
        f_ind = (
            _f_ind_from_perpetrators(exposures, j)
            if substrate and seg.has_cyp3a4
            else 0.0
        )
        e_ss = _steady_state_pool_fraction(kobs, kdeg_gut_per_h, f_ind)
        fg_base = baseline_segment_fg(
            seg, is_substrate=substrate, alpha=alpha
        )
        loss = alpha * seg.cyp3a4_activity * e_ss if substrate and seg.has_cyp3a4 else 0.0
        fg_mbi = max(0.0, 1.0 - loss)
        out.append(
            GutWallMBIDiagnostic(
                segment_name=seg.name,
                segment_idx=j,
                cyp3a4_activity=seg.cyp3a4_activity,
                kobs_per_h=kobs,
                e_ss=e_ss,
                fg_baseline=fg_base,
                fg_under_mbi=fg_mbi,
            )
        )
    return tuple(out)
