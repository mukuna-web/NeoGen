"""9-segment GI tract transit model (CAT-style) — V1 P2 Step 2.

Ports the legacy ``neogen/simulation/dissolution.py::GI_TRACT`` reference data
into a clean, frozen, standalone module that the dissolution and permeability
engines (Step 3 and Step 4) will consume.

Design notes
------------
- Self-contained: this module does NOT extend ``PhysiologyModel``. The GI tract
  is a drug-absorption construct used only by the P2 oral absorption pipeline,
  and it lives alongside ``tissue_partitioner.RR_TISSUES`` as a module-level
  frozen reference constant. Keeping it out of ``PhysiologyModel`` avoids
  polluting the P1 systemic physiology with oral-absorption concerns.
- Frozen everywhere: ``GISegment`` and ``GutTransit`` are ``frozen=True``
  dataclasses. The canonical tract is a module-level ``tuple``.
- Reference: Yu & Amidon (1999), Willmann et al. (2004), Jamei et al. (2009)
  Simcyp ADAM. Values match the legacy ``GI_TRACT`` verbatim so Step 3 can be
  validated against the same numbers.

Physiology (reference fasted state)
-----------------------------------
- 1 stomach + 6 small intestine segments + 2 colon segments
- Small intestine transit ≈ 2.6 h (within the 2-5 h physiological window)
- Total 9-segment transit ≈ 12.9 h (including slow colonic residence)
- pH profile: 1.5 (stomach) → 6-7.2 (SI) → 7.5-7.8 (colon)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# ════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GISegment:
    """Physiological parameters for one GI segment.

    All values are for a fasted-state reference adult (70 kg male).
    Units are explicit in every field name.
    """

    name: str
    """Human-readable segment name (e.g. "Jejunum_1")."""

    pH: float
    """Luminal pH (dimensionless)."""

    volume_mL: float
    """Fluid volume available for dissolution (mL)."""

    surface_area_cm2: float
    """Effective absorptive surface area (cm²). Includes villus amplification."""

    transit_time_h: float
    """Mean residence time of a bolus in this segment (h)."""

    radius_cm: float
    """Segment inner radius (cm) — used by permeation geometry."""

    bile_salt_mM: float = 0.0
    """Bile-salt concentration (mM). Aids solubilisation of lipophilic drugs."""

    has_cyp3a4: bool = False
    """Whether enterocytes in this segment express CYP3A4."""

    has_ugt: bool = False
    """Whether enterocytes in this segment express UGT isoforms."""

    cyp3a4_activity: float = 0.0
    """Relative CYP3A4 activity (0-1), scaled against the jejunal maximum."""

    def __post_init__(self) -> None:
        if self.pH <= 0.0:
            raise ValueError(f"pH must be > 0 (got {self.pH})")
        if self.volume_mL <= 0.0:
            raise ValueError(f"volume_mL must be > 0 (got {self.volume_mL})")
        if self.surface_area_cm2 <= 0.0:
            raise ValueError(
                f"surface_area_cm2 must be > 0 (got {self.surface_area_cm2})"
            )
        if self.transit_time_h <= 0.0:
            raise ValueError(
                f"transit_time_h must be > 0 (got {self.transit_time_h})"
            )
        if self.radius_cm <= 0.0:
            raise ValueError(f"radius_cm must be > 0 (got {self.radius_cm})")
        if self.bile_salt_mM < 0.0:
            raise ValueError("bile_salt_mM must be ≥ 0")
        if not (0.0 <= self.cyp3a4_activity <= 1.0):
            raise ValueError("cyp3a4_activity must be in [0, 1]")

    @property
    def is_stomach(self) -> bool:
        return self.name.lower().startswith("stomach")

    @property
    def is_small_intestine(self) -> bool:
        lname = self.name.lower()
        return (
            lname.startswith("duodenum")
            or lname.startswith("jejunum")
            or lname.startswith("ileum")
        )

    @property
    def is_colon(self) -> bool:
        return self.name.lower().startswith("colon")


@dataclass(frozen=True)
class GutTransit:
    """Immutable container for an ordered GI tract.

    Segments are stored as a tuple to preserve order and guarantee
    immutability. All bulk queries (total transit, cumulative entry times,
    segment lookup by clock time) are pure functions over that tuple.
    """

    segments: Tuple[GISegment, ...]

    def __post_init__(self) -> None:
        if len(self.segments) == 0:
            raise ValueError("GutTransit must contain at least one segment")
        # Name uniqueness keeps downstream lookups safe.
        names = [s.name for s in self.segments]
        if len(set(names)) != len(names):
            raise ValueError(f"Segment names must be unique (got {names})")

    # ── Aggregate timing ────────────────────────────────────────

    @property
    def total_transit_time_h(self) -> float:
        """Sum of residence times across every segment (h)."""
        return sum(s.transit_time_h for s in self.segments)

    @property
    def small_intestine_transit_h(self) -> float:
        """Total small-intestine residence time (h)."""
        return sum(
            s.transit_time_h for s in self.segments if s.is_small_intestine
        )

    @property
    def gastric_emptying_h(self) -> float:
        """Stomach residence time (h). 0 if no stomach segment."""
        return sum(s.transit_time_h for s in self.segments if s.is_stomach)

    @property
    def colonic_transit_h(self) -> float:
        """Total colonic residence time (h)."""
        return sum(s.transit_time_h for s in self.segments if s.is_colon)

    # ── Per-segment geometry sums ───────────────────────────────

    @property
    def total_absorptive_surface_cm2(self) -> float:
        """Total effective absorptive surface (cm²) across all segments."""
        return sum(s.surface_area_cm2 for s in self.segments)

    @property
    def small_intestine_surface_cm2(self) -> float:
        """Total SI absorptive surface (cm²) — dominant absorption site."""
        return sum(
            s.surface_area_cm2 for s in self.segments if s.is_small_intestine
        )

    @property
    def total_fluid_volume_mL(self) -> float:
        """Total dissolution fluid volume (mL) across all segments."""
        return sum(s.volume_mL for s in self.segments)

    # ── Timing helpers for ODE forcing ──────────────────────────

    @property
    def cumulative_entry_times_h(self) -> Tuple[float, ...]:
        """Clock time (h) at which a bolus first enters each segment.

        Segment 0 (stomach) is entered at t = 0. Segment k is entered at the
        cumulative transit time of segments 0..k-1.
        """
        out: list[float] = []
        running = 0.0
        for seg in self.segments:
            out.append(running)
            running += seg.transit_time_h
        return tuple(out)

    @property
    def cumulative_exit_times_h(self) -> Tuple[float, ...]:
        """Clock time (h) at which a bolus finishes each segment."""
        entries = self.cumulative_entry_times_h
        return tuple(
            entry + seg.transit_time_h
            for entry, seg in zip(entries, self.segments)
        )

    def segment_for_time(self, t_h: float) -> Optional[GISegment]:
        """Return the segment a bolus is in at clock time ``t_h``.

        Returns ``None`` if ``t_h`` is negative or past the final exit time
        (drug has left the GI tract).
        """
        if t_h < 0.0:
            return None
        entries = self.cumulative_entry_times_h
        exits = self.cumulative_exit_times_h
        for seg, enter, leave in zip(self.segments, entries, exits):
            if enter <= t_h < leave:
                return seg
        return None

    def segment_by_name(self, name: str) -> GISegment:
        """Look up a segment by exact name. Raises ``KeyError`` if missing."""
        for seg in self.segments:
            if seg.name == name:
                return seg
        raise KeyError(f"No GI segment named {name!r}")


# ════════════════════════════════════════════════════════════════
# REFERENCE GI TRACT (Yu & Amidon 1999, Willmann 2004)
# ════════════════════════════════════════════════════════════════

# Tuple is module-level and frozen — treat as a canonical physiological
# constant, never mutate. Values are verbatim from the legacy
# ``simulation/dissolution.py::GI_TRACT`` list.

REFERENCE_GI_SEGMENTS: Tuple[GISegment, ...] = (
    GISegment(
        name="Stomach",
        pH=1.5,
        volume_mL=250.0,
        surface_area_cm2=3.5,
        transit_time_h=0.25,
        radius_cm=5.0,
        bile_salt_mM=0.0,
        has_cyp3a4=False,
        has_ugt=False,
        cyp3a4_activity=0.0,
    ),
    GISegment(
        name="Duodenum",
        pH=6.0,
        volume_mL=40.0,
        surface_area_cm2=150.0,
        transit_time_h=0.26,
        radius_cm=1.75,
        bile_salt_mM=10.0,
        has_cyp3a4=True,
        has_ugt=True,
        cyp3a4_activity=0.4,
    ),
    GISegment(
        name="Jejunum_1",
        pH=6.2,
        volume_mL=120.0,
        surface_area_cm2=550.0,
        transit_time_h=0.47,
        radius_cm=1.75,
        bile_salt_mM=8.0,
        has_cyp3a4=True,
        has_ugt=True,
        cyp3a4_activity=1.0,
    ),
    GISegment(
        name="Jejunum_2",
        pH=6.4,
        volume_mL=120.0,
        surface_area_cm2=550.0,
        transit_time_h=0.47,
        radius_cm=1.75,
        bile_salt_mM=6.0,
        has_cyp3a4=True,
        has_ugt=True,
        cyp3a4_activity=0.8,
    ),
    GISegment(
        name="Ileum_1",
        pH=6.6,
        volume_mL=120.0,
        surface_area_cm2=350.0,
        transit_time_h=0.47,
        radius_cm=1.5,
        bile_salt_mM=4.0,
        has_cyp3a4=True,
        has_ugt=True,
        cyp3a4_activity=0.5,
    ),
    GISegment(
        name="Ileum_2",
        pH=7.0,
        volume_mL=120.0,
        surface_area_cm2=350.0,
        transit_time_h=0.47,
        radius_cm=1.5,
        bile_salt_mM=2.0,
        has_cyp3a4=True,
        has_ugt=False,
        cyp3a4_activity=0.3,
    ),
    GISegment(
        name="Ileum_3",
        pH=7.2,
        volume_mL=120.0,
        surface_area_cm2=200.0,
        transit_time_h=0.47,
        radius_cm=1.5,
        bile_salt_mM=1.0,
        has_cyp3a4=False,
        has_ugt=False,
        cyp3a4_activity=0.1,
    ),
    GISegment(
        name="Colon_asc",
        pH=7.8,
        volume_mL=150.0,
        surface_area_cm2=50.0,
        transit_time_h=4.0,
        radius_cm=3.0,
        bile_salt_mM=0.5,
        has_cyp3a4=False,
        has_ugt=False,
        cyp3a4_activity=0.0,
    ),
    GISegment(
        name="Colon_trans",
        pH=7.5,
        volume_mL=100.0,
        surface_area_cm2=30.0,
        transit_time_h=6.0,
        radius_cm=2.5,
        bile_salt_mM=0.2,
        has_cyp3a4=False,
        has_ugt=False,
        cyp3a4_activity=0.0,
    ),
)


def build_reference_gi_tract() -> GutTransit:
    """Return the canonical fasted-state reference GI tract.

    This is the default used by every P2 module unless a caller explicitly
    supplies a modified tract (fed state, pediatric, gastric-bypass, etc.).
    """
    return GutTransit(segments=REFERENCE_GI_SEGMENTS)


# ════════════════════════════════════════════════════════════════
# FED-STATE GI TRACT  (V2 extension — food effect)
# ════════════════════════════════════════════════════════════════
#
# The fasted reference above is the standard Yu & Amidon / Jamei 2009
# ADAM baseline. Fed-state physiology differs in four places that
# materially change oral bioavailability:
#
#   1. Gastric pH rises from ~1.5 to ~4.5 as the meal buffers HCl.
#      Weak-base drugs precipitate faster fasted (ionised at low pH →
#      dissolved) and slower fed; weak acids do the opposite. Both show
#      up as dissolution-rate changes inside the dissolution engine
#      (pH-dependent solubility).
#   2. Gastric emptying slows from 0.25 h (fasted liquid) to ~1.0 h
#      (homogenised meal) — large tablets take 2–4 h, but the clinically
#      observed food-effect exposures are dominated by the liquid /
#      chyme phase, so 1.0 h is the defensible single-compartment value.
#   3. Bile-salt concentration in duodenum and upper jejunum jumps from
#      ~8–10 mM (fasted basal) to ~15–20 mM following gallbladder
#      contraction. Bile salts solubilise lipophilic drugs (BCS II / IV)
#      and are the main reason those drugs see a positive food effect.
#   4. Upper small intestine luminal volume rises roughly 30 % due to
#      meal-triggered secretion (saliva, gastric juice, pancreatic
#      fluid, bile). The dissolution engine uses this as the sink
#      volume for the Noyes–Whitney / Nernst–Brunner equation.
#
# References: Jamei et al. 2009 (Simcyp ADAM, DDD 24 fed parameters),
# Dressman 1998 (fed-state SIF composition), Fleisher 1999 (bile salt
# kinetics post-meal), Mudie 2014 (fed SI fluid volumes by MRI).

# Multiplicative factors are centralised so a future calibration sweep
# can tune them from one place.
_FED_GASTRIC_PH: float = 4.5
_FED_GASTRIC_EMPTYING_H: float = 1.0
_FED_DUODENUM_BILE_SALT_MM: float = 18.0
_FED_UPPER_JEJUNUM_BILE_SALT_MM: float = 14.0
_FED_LOWER_JEJUNUM_BILE_SALT_MM: float = 10.0
_FED_UPPER_SI_VOLUME_FACTOR: float = 1.3  # +30 % fluid in duodenum + upper SI


def build_fed_gi_tract() -> GutTransit:
    """Return a fed-state GI tract derived from the fasted reference.

    The modifications are additive patches to the fasted reference
    segments — we do not construct a brand-new segment list because the
    overall 9-compartment geometry is the same between fed and fasted
    states. Only the pH, emptying time, bile-salt load, and luminal
    volume for the first few segments change.

    Returns a brand-new :class:`GutTransit` — the fasted reference is
    never mutated.
    """
    patched: list[GISegment] = []
    for seg in REFERENCE_GI_SEGMENTS:
        if seg.is_stomach:
            patched.append(
                GISegment(
                    name=seg.name,
                    pH=_FED_GASTRIC_PH,
                    volume_mL=seg.volume_mL,
                    surface_area_cm2=seg.surface_area_cm2,
                    transit_time_h=_FED_GASTRIC_EMPTYING_H,
                    radius_cm=seg.radius_cm,
                    bile_salt_mM=seg.bile_salt_mM,
                    has_cyp3a4=seg.has_cyp3a4,
                    has_ugt=seg.has_ugt,
                    cyp3a4_activity=seg.cyp3a4_activity,
                )
            )
            continue

        if seg.name == "Duodenum":
            patched.append(
                GISegment(
                    name=seg.name,
                    pH=seg.pH,
                    volume_mL=seg.volume_mL * _FED_UPPER_SI_VOLUME_FACTOR,
                    surface_area_cm2=seg.surface_area_cm2,
                    transit_time_h=seg.transit_time_h,
                    radius_cm=seg.radius_cm,
                    bile_salt_mM=_FED_DUODENUM_BILE_SALT_MM,
                    has_cyp3a4=seg.has_cyp3a4,
                    has_ugt=seg.has_ugt,
                    cyp3a4_activity=seg.cyp3a4_activity,
                )
            )
            continue

        if seg.name == "Jejunum_1":
            patched.append(
                GISegment(
                    name=seg.name,
                    pH=seg.pH,
                    volume_mL=seg.volume_mL * _FED_UPPER_SI_VOLUME_FACTOR,
                    surface_area_cm2=seg.surface_area_cm2,
                    transit_time_h=seg.transit_time_h,
                    radius_cm=seg.radius_cm,
                    bile_salt_mM=_FED_UPPER_JEJUNUM_BILE_SALT_MM,
                    has_cyp3a4=seg.has_cyp3a4,
                    has_ugt=seg.has_ugt,
                    cyp3a4_activity=seg.cyp3a4_activity,
                )
            )
            continue

        if seg.name == "Jejunum_2":
            patched.append(
                GISegment(
                    name=seg.name,
                    pH=seg.pH,
                    volume_mL=seg.volume_mL,
                    surface_area_cm2=seg.surface_area_cm2,
                    transit_time_h=seg.transit_time_h,
                    radius_cm=seg.radius_cm,
                    bile_salt_mM=_FED_LOWER_JEJUNUM_BILE_SALT_MM,
                    has_cyp3a4=seg.has_cyp3a4,
                    has_ugt=seg.has_ugt,
                    cyp3a4_activity=seg.cyp3a4_activity,
                )
            )
            continue

        # Ileum + colon are unchanged by meal ingestion in the single-
        # compartment averaged description. A future refinement could
        # taper bile salt decay through the ileum; today the fasted
        # values are already small relative to the upper-SI jump.
        patched.append(seg)

    return GutTransit(segments=tuple(patched))


def build_gi_tract_for_patient(*, is_fasted: bool) -> GutTransit:
    """Choose the fed or fasted GI tract based on a patient flag.

    This is the canonical entry point the PK orchestrator uses so that
    ``PatientProfile.is_fasted`` drives absorption physiology through a
    single, testable function.
    """
    return build_reference_gi_tract() if is_fasted else build_fed_gi_tract()
