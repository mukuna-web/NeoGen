"""Frozen P2 result contract — ``AbsorptionProfile``.

This is the only data structure that Step 5 (PBPK integration) will read
out of the P2 pipeline. It is intentionally self-contained: callers do not
need a handle on the engine or the dissolution trace to inject the oral
forcing function into the PBPK ODE.

Architecture reference: V1 Blueprint — Layer 2, F = Fa · Fg · Fh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from neogen.core.entities.formulation import FormulationType


@dataclass(frozen=True)
class AbsorptionProfile:
    """Immutable output of ``PermeabilityEngine.simulate``.

    Carries every piece of information Step 5 needs to treat oral
    absorption as a time-resolved forcing function on the PBPK system.
    """

    # ── Identity ────────────────────────────────────────────────
    drug_name: str
    dose_mg: float
    formulation_type: FormulationType

    # ── Summary metrics (Fa · Fg) ───────────────────────────────
    fa: float
    """Fraction absorbed from lumen into enterocytes ∈ [0, 1]."""

    fg: float
    """Fraction escaping gut-wall metabolism (enterocyte → portal) ∈ [0, 1]."""

    f_portal: float
    """Combined portal-side bioavailability, ``fa × fg``. Excludes hepatic
    first-pass (Fh is applied downstream by the PBPK / hepatic clearance
    engine)."""

    effective_ka_per_h: float
    """First-order ka fitted to the cumulative portal-entry curve (1/h)."""

    tlag_h: float
    """Absorption lag time — time to reach 5% of final portal mass (h)."""

    # ── Time-resolved state ─────────────────────────────────────
    t_h: np.ndarray
    """Uniform time grid (T,) in hours."""

    cum_permeated_mg: np.ndarray
    """(T,) cumulative drug that crossed gut wall (Fa numerator)."""

    cum_portal_mg: np.ndarray
    """(T,) cumulative drug delivered to portal vein after gut-wall metabolism."""

    cum_gut_metabolized_mg: np.ndarray
    """(T,) cumulative drug lost to enterocyte CYP3A4 extraction."""

    undissolved_mg: np.ndarray
    """(T, N) per-segment undissolved mass trace."""

    dissolved_mg: np.ndarray
    """(T, N) per-segment dissolved mass trace."""

    excreted_mg: np.ndarray
    """(T,) cumulative drug that transited out of the final segment."""

    # ── Metadata ────────────────────────────────────────────────
    segment_names: Tuple[str, ...]
    per_segment_portal_fraction: Dict[str, float]
    """Fraction of total portal absorption originating from each segment.
    Sums to approximately 1.0 over absorbing segments."""

    solver_success: bool
    solver_method: str

    # ────────────────────────────────────────────────────────────
    # Shape / invariants
    # ────────────────────────────────────────────────────────────

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
        for name, arr in (
            ("cum_permeated_mg", self.cum_permeated_mg),
            ("cum_portal_mg", self.cum_portal_mg),
            ("cum_gut_metabolized_mg", self.cum_gut_metabolized_mg),
            ("excreted_mg", self.excreted_mg),
        ):
            if arr.shape != (T,):
                raise ValueError(f"{name} must be 1-D length {T}")
        if not (0.0 <= self.fa <= 1.0 + 1e-6):
            raise ValueError(f"fa must be in [0, 1] (got {self.fa})")
        if not (0.0 <= self.fg <= 1.0 + 1e-6):
            raise ValueError(f"fg must be in [0, 1] (got {self.fg})")
        if self.effective_ka_per_h < 0.0:
            raise ValueError("effective_ka_per_h must be ≥ 0")
        if self.tlag_h < 0.0:
            raise ValueError("tlag_h must be ≥ 0")

    # ────────────────────────────────────────────────────────────
    # Projections
    # ────────────────────────────────────────────────────────────

    @property
    def n_segments(self) -> int:
        return len(self.segment_names)

    @property
    def total_portal_mg(self) -> float:
        """Total drug delivered to portal vein at end of simulation."""
        return float(self.cum_portal_mg[-1])

    @property
    def total_permeated_mg(self) -> float:
        return float(self.cum_permeated_mg[-1])

    @property
    def mass_balance_error_mg(self) -> float:
        """Max |dose − (U+D+permeated+excreted)| across the full trace."""
        inventory = (
            self.undissolved_mg.sum(axis=1)
            + self.dissolved_mg.sum(axis=1)
            + self.cum_permeated_mg
            + self.excreted_mg
        )
        return float(np.max(np.abs(inventory - self.dose_mg)))

    def portal_input_rate_at(self, t_h: float) -> float:
        """Instantaneous portal-entry mass flux at time ``t_h`` (mg/h).

        Computed by finite-differencing the cumulative portal curve with
        linear interpolation. This is the callable shape Step 5 will wire
        into the PBPK RHS as an oral forcing term.
        """
        if len(self.t_h) < 2:
            return 0.0
        if t_h <= self.t_h[0] or t_h >= self.t_h[-1]:
            return 0.0
        # Linear-interpolated derivative via neighbouring grid points.
        idx = int(np.searchsorted(self.t_h, t_h))
        idx = max(1, min(idx, len(self.t_h) - 1))
        dt = self.t_h[idx] - self.t_h[idx - 1]
        if dt <= 0.0:
            return 0.0
        dp = self.cum_portal_mg[idx] - self.cum_portal_mg[idx - 1]
        rate = dp / dt
        return max(rate, 0.0)

    def portal_dissolved_fraction_in_segment(self, name: str) -> float:
        """Fraction of total portal absorption from a named segment."""
        return self.per_segment_portal_fraction.get(name, 0.0)

    def __repr__(self) -> str:
        return (
            f"AbsorptionProfile({self.drug_name!r}, "
            f"dose={self.dose_mg}mg, "
            f"Fa={self.fa:.3f}, Fg={self.fg:.3f}, "
            f"F_portal={self.f_portal:.3f}, "
            f"ka={self.effective_ka_per_h:.3f}/h, "
            f"tlag={self.tlag_h:.2f}h)"
        )
