"""Mechanistic dissolution engine — V1 P2 Step 3.

Implements pH-dependent Noyes-Whitney dissolution across the 9-segment
reference GI tract (Step 2). This module is the **mass source** for the
P2 absorption pipeline: it tells the Step 4 permeability engine how much
drug is dissolved (and therefore available for transmembrane uptake) in
each GI segment at every instant.

This is a clean V1 port of ``neogen/simulation/dissolution.py`` with the
following changes:

1. No global state. The GI tract is passed in explicitly.
2. No mutation of input entities. Drug/Formulation/GutTransit are consumed
   as read-only references.
3. Frozen ``DissolutionTrace`` result (numpy arrays inside a frozen
   dataclass, mass-balance checked).
4. Dissolution is decoupled from absorption — Step 4 composes this engine
   with the permeability engine through an explicit absorption sink.
5. ODE state is integrated by ``scipy.solve_ivp`` (RK45) rather than a
   fixed-step Euler loop.

Physics
-------
- Solubility: Henderson-Hasselbalch applied to ``drug.pka_acidic`` /
  ``drug.pka_basic``, scaled by a linear bile-salt enhancement term.
- Particle surface area: spherical monodisperse assumption, radius held
  constant → SA ∝ remaining undissolved mass (linear regime).
- Dissolution rate: ``dM/dt = D · SA · (Cs·B − C) / h`` where h = 30 µm
  stagnant layer, D = 5e-6 cm²/s, B = bile-salt enhancement factor.
- Transit: first-order with ``k_transit = 1 / segment.transit_time_h`` —
  drug flows stomach → duodenum → … → colon → excreted.
- Saturation clamp: dissolution halts whenever the dissolved concentration
  reaches saturation in a segment (``C ≥ Cs·B``).

State vector
------------
For an N-segment tract the ODE state is length ``2N + 1``:

    y = [U_1, U_2, ..., U_N,   D_1, D_2, ..., D_N,   cum_dissolved]

where ``U_j`` is undissolved mass (mg) in segment j, ``D_j`` is dissolved
mass (mg), and ``cum_dissolved`` is the cumulative mass that has *ever*
been dissolved (monotonic). Mass conservation:

    Σ U_j + Σ D_j + excreted = dose      (excreted = dose − Σy[:2N])

``cum_dissolved`` is tracked as a separate integrator so that downstream
metrics ("% dissolved by time t") never get confused by later absorption
or transit out of the GI tract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from neogen.core.entities.drug import BCSClass, Drug, Ionization
from neogen.core.entities.formulation import Formulation, FormulationType
from neogen.engine.absorption.gut_transit import GISegment, GutTransit


# ════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (reference fasted-state values)
# ════════════════════════════════════════════════════════════════

# Aqueous diffusion coefficient of a typical small molecule (cm²/s).
# Reference: Avdeef (2012) "Absorption and Drug Development", 2nd ed.
_DIFFUSION_COEFF_CM2_PER_S: float = 5.0e-6

# Stagnant (Nernst) diffusion layer thickness (cm). 30 µm is the
# conventional value used in ADAM / GastroPlus.
_DIFFUSION_LAYER_CM: float = 30.0e-4

# Particle density default (g/cm³) — typical organic crystalline solid.
_DEFAULT_PARTICLE_DENSITY_G_PER_CM3: float = 1.2

# Bile-salt enhancement slope: B = 1 + k · [bile_mM]. Small for V1 MVP;
# refined in V2 via drug-specific logP scaling.
_BILE_SALT_ENHANCEMENT_PER_MM: float = 0.1

# Physical ceiling on the effective saturation concentration used by the
# Noyes-Whitney rate. Henderson-Hasselbalch can return astronomical
# ionized-form solubilities (e.g. metformin at pH 1.5 → 10¹³ mg/mL) that
# are not physically realisable: in practice ionic drug solubility is
# bounded by counter-ion availability, common-ion effects, and the
# density of liquid water (~1000 mg/mL). We cap at 1 g/mL — beyond that
# the boundary-layer dissolution model is not valid anyway. This keeps
# the ODE numerically well-behaved for basic drugs in the stomach
# without impacting BCS II / IV dissolution-limited regimes.
_MAX_EFFECTIVE_SOLUBILITY_MG_PER_ML: float = 1000.0

# Fallback intrinsic solubility (mg/mL at pH 6.8) when
# ``drug.solubility_mg_per_ml`` is missing, keyed on BCS class.
# Sources: FDA BCS Guidance (2015), Butler & Dressman (2010). These are
# intentionally coarse — they exist so the mechanistic P2 path never
# hard-crashes on a drug that lacks a measured solubility, not as a
# substitute for real literature values.
_BCS_SOLUBILITY_FALLBACK_MG_PER_ML: dict = {
    BCSClass.I: 10.0,     # High solubility reference (e.g. propranolol)
    BCSClass.II: 0.05,    # Low solubility (e.g. ibuprofen, atorvastatin)
    BCSClass.III: 50.0,   # High solubility, low permeability (e.g. metformin)
    BCSClass.IV: 0.02,    # Low / low (e.g. furosemide)
}


def intrinsic_solubility_mg_per_ml(drug: Drug) -> float:
    """Return the drug's intrinsic (pH-independent) solubility in mg/mL.

    Falls back to a BCS-class reference if ``drug.solubility_mg_per_ml`` is
    None. The caller (``solubility_at_ph``) then applies Henderson-Hasselbalch
    ionisation on top of this intrinsic value.
    """
    s0 = drug.solubility_mg_per_ml
    if s0 is not None and s0 > 0.0:
        return float(s0)
    return float(_BCS_SOLUBILITY_FALLBACK_MG_PER_ML[drug.bcs_class])


# ════════════════════════════════════════════════════════════════
# PURE HELPERS (stateless)
# ════════════════════════════════════════════════════════════════


def solubility_at_ph(drug: Drug, pH: float) -> float:
    """pH-dependent aqueous solubility via Henderson-Hasselbalch (mg/mL).

    For a weak acid with strongest acidic pKa:
        S(pH) = S₀ · (1 + 10^(pH − pKa))

    For a weak base with strongest basic pKa:
        S(pH) = S₀ · (1 + 10^(pKa − pH))

    Neutral drugs are treated as pH-independent. Zwitterions / ampholytes
    are approximated as the minimum-solubility enhancement of whichever
    pKa is available (conservative — underestimates dissolution).
    """
    s0 = intrinsic_solubility_mg_per_ml(drug)
    if s0 <= 0.0:
        raise ValueError(
            f"Drug {drug.name!r} has non-positive solubility ({s0})"
        )

    ion = drug.ionization_class

    if ion is Ionization.NEUTRAL:
        return s0

    if ion is Ionization.ACID:
        if drug.pka_acidic is None:
            return s0
        return s0 * (1.0 + 10.0 ** (pH - drug.pka_acidic))

    if ion is Ionization.BASE:
        if drug.pka_basic is None:
            return s0
        return s0 * (1.0 + 10.0 ** (drug.pka_basic - pH))

    # Zwitterion / ampholyte: apply whichever correction is available.
    # Conservative: multiply by the minimum of the two factors.
    factors = [1.0]
    if drug.pka_acidic is not None:
        factors.append(1.0 + 10.0 ** (pH - drug.pka_acidic))
    if drug.pka_basic is not None:
        factors.append(1.0 + 10.0 ** (drug.pka_basic - pH))
    return s0 * min(factors)


def bile_salt_enhancement(bile_salt_mM: float) -> float:
    """Linear bile-salt enhancement factor (dimensionless, ≥ 1).

    Very lipophilic drugs benefit from higher slopes — that refinement
    lives in V2. For V1 MVP we apply a uniform slope.
    """
    return 1.0 + _BILE_SALT_ENHANCEMENT_PER_MM * max(bile_salt_mM, 0.0)


def particle_surface_area_cm2(
    undissolved_mg: float,
    particle_radius_um: float,
    density_g_per_cm3: float = _DEFAULT_PARTICLE_DENSITY_G_PER_CM3,
) -> float:
    """Total particle surface area for a spherical monodisperse dose (cm²).

    Radius is held constant (linear-regime approximation): SA scales
    linearly with remaining undissolved mass.

    SA = 3 · m / (r · ρ)   (derivation: n × 4πr² with n = m / (4/3·πr³·ρ))
    """
    if undissolved_mg <= 0.0 or particle_radius_um <= 0.0:
        return 0.0
    r_cm = particle_radius_um * 1.0e-4
    mass_g = undissolved_mg * 1.0e-3
    return (3.0 * mass_g) / (r_cm * density_g_per_cm3)


def noyes_whitney_rate_mg_per_h(
    undissolved_mg: float,
    dissolved_mg: float,
    segment: GISegment,
    drug: Drug,
    formulation: Formulation,
) -> float:
    """Noyes-Whitney dissolution rate (mg/h) for one GI segment.

    Returns 0 when the dissolved pool is already saturated or the particle
    mass has been exhausted.

    ``dM/dt = (D · SA · (Cs·B − C)) / h``       (mg·cm⁻¹·s⁻¹ → mg/h below)
    """
    if undissolved_mg <= 0.0:
        return 0.0

    cs_mg_per_ml = solubility_at_ph(drug, segment.pH)
    bile = bile_salt_enhancement(segment.bile_salt_mM)
    cs_effective = min(
        cs_mg_per_ml * bile,
        _MAX_EFFECTIVE_SOLUBILITY_MG_PER_ML,
    )

    # Segment fluid volume: convert mL → mL (already mL). Current dissolved
    # concentration is expressed in mg/mL to match Cs units.
    volume_mL = max(segment.volume_mL, 0.001)
    c_current = dissolved_mg / volume_mL

    if c_current >= cs_effective:
        return 0.0

    sa_cm2 = particle_surface_area_cm2(
        undissolved_mg,
        formulation.particle_radius_um,
    )
    if sa_cm2 == 0.0:
        return 0.0

    # Noyes-Whitney — D in cm²/s, SA in cm², concentration in mg/mL ≡ mg/cm³,
    # layer h in cm → rate in mg/s. Multiply by 3600 for mg/h.
    rate_mg_per_s = (
        _DIFFUSION_COEFF_CM2_PER_S
        * sa_cm2
        * (cs_effective - c_current)
        / _DIFFUSION_LAYER_CM
    )
    return rate_mg_per_s * 3600.0


# ════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DissolutionTrace:
    """Immutable output of one ``DissolutionEngine.simulate`` run.

    Arrays are numpy ndarrays; they should be treated as read-only by
    callers. All masses are milligrams, all times are hours.
    """

    dose_mg: float
    drug_name: str
    formulation_type: FormulationType

    t_h: np.ndarray
    """1-D array of shape (T,) — time grid in hours."""

    segment_names: Tuple[str, ...]
    """N segment names in transit order."""

    undissolved_mg: np.ndarray
    """Shape (T, N). Undissolved mass per segment at each timestep (mg)."""

    dissolved_mg: np.ndarray
    """Shape (T, N). Dissolved mass per segment at each timestep (mg)."""

    cumulative_dissolved_mg: np.ndarray
    """Shape (T,). Monotonically increasing cumulative dissolved mass (mg)."""

    excreted_mg: np.ndarray
    """Shape (T,). Mass that has transited out of the final segment (mg)."""

    solver_success: bool

    def __post_init__(self) -> None:
        T = len(self.t_h)
        N = len(self.segment_names)
        if self.undissolved_mg.shape != (T, N):
            raise ValueError(
                f"undissolved_mg shape {self.undissolved_mg.shape} != ({T},{N})"
            )
        if self.dissolved_mg.shape != (T, N):
            raise ValueError(
                f"dissolved_mg shape {self.dissolved_mg.shape} != ({T},{N})"
            )
        if self.cumulative_dissolved_mg.shape != (T,):
            raise ValueError("cumulative_dissolved_mg must be 1-D length T")
        if self.excreted_mg.shape != (T,):
            raise ValueError("excreted_mg must be 1-D length T")

    # ── Convenience projections ─────────────────────────────────

    @property
    def n_segments(self) -> int:
        return len(self.segment_names)

    @property
    def total_dissolved_mg(self) -> np.ndarray:
        """Total dissolved across all segments at each timestep (T,)."""
        return self.dissolved_mg.sum(axis=1)

    @property
    def total_undissolved_mg(self) -> np.ndarray:
        """Total undissolved across all segments at each timestep (T,)."""
        return self.undissolved_mg.sum(axis=1)

    @property
    def fraction_dissolved_cumulative(self) -> np.ndarray:
        """Cumulative dissolved fraction ∈ [0, 1]."""
        if self.dose_mg <= 0.0:
            return np.zeros_like(self.cumulative_dissolved_mg)
        return self.cumulative_dissolved_mg / self.dose_mg

    @property
    def final_cumulative_fraction(self) -> float:
        """Total dissolved fraction at the last time point."""
        return float(self.fraction_dissolved_cumulative[-1])

    @property
    def mass_balance_error(self) -> float:
        """Maximum absolute mass-balance error over the full trace (mg).

        total_mass(t) = Σ undissolved + Σ dissolved + excreted should
        equal ``dose_mg`` at every step. Uses (cumulative_dissolved −
        current_dissolved) as the proxy for mass that dissolved and then
        either got excreted or stayed in transit.
        """
        inventory = (
            self.total_undissolved_mg
            + self.total_dissolved_mg
            + self.excreted_mg
        )
        return float(np.max(np.abs(inventory - self.dose_mg)))

    def dissolved_fraction_in_segment(self, name: str) -> np.ndarray:
        """Per-segment dissolved mass curve for a named segment (T,)."""
        if name not in self.segment_names:
            raise KeyError(f"No segment named {name!r}")
        idx = self.segment_names.index(name)
        return self.dissolved_mg[:, idx]


# ════════════════════════════════════════════════════════════════
# DISSOLUTION ENGINE
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DissolutionEngine:
    """Stateless dissolution simulator.

    Instances are frozen dataclasses with no runtime state; the ``simulate``
    method is effectively a pure function of its arguments. A class is used
    (instead of a bare function) only for interface symmetry with the rest
    of the V1 engines (``HepaticClearanceEngine``, ``TissuePartitioner``).
    """

    solver_method: str = "RK45"
    rtol: float = 1.0e-6
    atol: float = 1.0e-9

    def simulate(
        self,
        drug: Drug,
        formulation: Formulation,
        dose_mg: float,
        gut_transit: Optional[GutTransit] = None,
        t_end_h: float = 24.0,
        n_points: int = 121,
    ) -> DissolutionTrace:
        """Integrate dissolution across the full GI transit for one dose.

        Parameters
        ----------
        drug
            V1 ``Drug`` entity — requires ``solubility_mg_per_ml`` and the
            relevant pKa fields to be populated.
        formulation
            ``Formulation`` describing particle size, disintegration delay,
            and dose form.
        dose_mg
            Oral dose in milligrams.
        gut_transit
            The GI tract to simulate in. Defaults to the fasted-state
            reference tract from ``build_reference_gi_tract``.
        t_end_h
            Simulation horizon in hours (default 24 h).
        n_points
            Number of output time points — the ODE is solved with dense
            output and sampled on a uniform grid.
        """
        if dose_mg <= 0.0:
            raise ValueError(f"dose_mg must be > 0 (got {dose_mg})")
        if t_end_h <= 0.0:
            raise ValueError(f"t_end_h must be > 0 (got {t_end_h})")
        if n_points < 2:
            raise ValueError("n_points must be ≥ 2")

        if gut_transit is None:
            from neogen.engine.absorption.gut_transit import (
                build_reference_gi_tract,
            )

            gut_transit = build_reference_gi_tract()

        segments: Tuple[GISegment, ...] = gut_transit.segments
        n_seg = len(segments)

        # Precompute per-segment constants.
        transit_k = np.array(
            [1.0 / max(s.transit_time_h, 1.0e-6) for s in segments]
        )

        # Disintegration delay in hours (tablet/capsule forms only).
        disintegration_h = formulation.disintegration_time_min / 60.0
        needs_disintegration = formulation.formulation_type in (
            FormulationType.IR_TABLET,
            FormulationType.IR_CAPSULE,
        )

        # ── Initial conditions ──────────────────────────────────
        # y layout: [U_0..U_{N-1}, D_0..D_{N-1}, cum_dissolved, excreted]
        # Total length = 2N + 2
        y0 = np.zeros(2 * n_seg + 2)

        if formulation.formulation_type is FormulationType.SOLUTION:
            # Pre-dissolved in stomach.
            y0[n_seg + 0] = dose_mg
            y0[-2] = dose_mg  # cumulative dissolved counts the solution dose
        elif formulation.formulation_type is FormulationType.ENTERIC_COATED:
            # Coating protects drug until duodenum (first SI segment). For
            # the reference tract that is index 1. Fall back to index 0
            # if no SI segment exists.
            entry_idx = 0
            for idx, seg in enumerate(segments):
                if seg.is_small_intestine:
                    entry_idx = idx
                    break
            y0[entry_idx] = dose_mg  # undissolved, bypassing stomach
        else:
            # IR tablet / capsule / other solid in stomach.
            y0[0] = dose_mg

        # ── RHS closure ─────────────────────────────────────────

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            dy = np.zeros_like(y)
            undiss = y[:n_seg]
            diss = y[n_seg : 2 * n_seg]

            # Honor disintegration delay: before t_disint, tablets are
            # intact and neither dissolve nor transit (stay in stomach).
            suppress_stomach_ops = (
                needs_disintegration and t < disintegration_h
            )

            for j, seg in enumerate(segments):
                u = undiss[j]
                d = diss[j]

                # Dissolution rate
                if suppress_stomach_ops and j == 0:
                    diss_rate = 0.0
                else:
                    diss_rate = noyes_whitney_rate_mg_per_h(
                        u, d, seg, drug, formulation
                    )

                # Transit out of this segment (first-order)
                if suppress_stomach_ops and j == 0:
                    out_u = 0.0
                    out_d = 0.0
                else:
                    out_u = transit_k[j] * u
                    out_d = transit_k[j] * d

                # dU_j / dt
                dy[j] += -diss_rate - out_u
                # dD_j / dt
                dy[n_seg + j] += diss_rate - out_d

                # Transit into next segment (or excretion if last)
                if j < n_seg - 1:
                    dy[j + 1] += out_u
                    dy[n_seg + j + 1] += out_d
                else:
                    dy[-1] += out_u + out_d  # excreted

                # Cumulative dissolved (monotonic)
                dy[-2] += diss_rate

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
            dense_output=False,
        )

        # Clamp numerical negatives (RK45 can undershoot by ~atol)
        y_out = np.clip(sol.y, a_min=0.0, a_max=None)

        undiss_mat = y_out[:n_seg, :].T        # shape (T, N)
        diss_mat = y_out[n_seg : 2 * n_seg, :].T
        cum_dissolved = y_out[-2, :]
        excreted = y_out[-1, :]

        return DissolutionTrace(
            dose_mg=dose_mg,
            drug_name=drug.name,
            formulation_type=formulation.formulation_type,
            t_h=t_eval,
            segment_names=tuple(s.name for s in segments),
            undissolved_mg=undiss_mat,
            dissolved_mg=diss_mat,
            cumulative_dissolved_mg=cum_dissolved,
            excreted_mg=excreted,
            solver_success=bool(sol.success),
        )
