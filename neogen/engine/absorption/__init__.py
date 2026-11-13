"""P2 — Absorption & Oral Bioavailability engine.

Modules:
    gut_transit           — 9-segment CAT-style GI transit model (Step 2)
    dissolution_engine    — Noyes-Whitney dissolution           (Step 3)
    permeability_engine   — Peff → per-segment ka               (Step 4)
    bioavailability       — Fa · Fg · Fh decomposition          (Step 4)
    absorption_profile    — frozen P2 result                    (Step 4)

Only Step 2 (gut_transit) is implemented so far.
"""

from neogen.engine.absorption.gut_transit import (
    GISegment,
    GutTransit,
    REFERENCE_GI_SEGMENTS,
    build_fed_gi_tract,
    build_gi_tract_for_patient,
    build_reference_gi_tract,
)
from neogen.engine.absorption.dissolution_engine import (
    DissolutionEngine,
    DissolutionTrace,
    bile_salt_enhancement,
    noyes_whitney_rate_mg_per_h,
    particle_surface_area_cm2,
    solubility_at_ph,
)
from neogen.engine.absorption.absorption_profile import AbsorptionProfile
from neogen.engine.absorption.bioavailability import (
    BioavailabilityEngine,
    BioavailabilityReport,
    classify_bcs,
    estimate_fa,
    estimate_fg,
    estimate_fh,
)
from neogen.engine.absorption.permeability_engine import (
    PermeabilityEngine,
    effective_peff_cm_per_s,
    is_gut_cyp3a4_substrate,
    segment_fg,
)

__all__ = [
    "GISegment",
    "GutTransit",
    "REFERENCE_GI_SEGMENTS",
    "build_reference_gi_tract",
    "build_fed_gi_tract",
    "build_gi_tract_for_patient",
    "DissolutionEngine",
    "DissolutionTrace",
    "solubility_at_ph",
    "bile_salt_enhancement",
    "particle_surface_area_cm2",
    "noyes_whitney_rate_mg_per_h",
    "AbsorptionProfile",
    "BioavailabilityEngine",
    "BioavailabilityReport",
    "classify_bcs",
    "estimate_fa",
    "estimate_fg",
    "estimate_fh",
    "PermeabilityEngine",
    "effective_peff_cm_per_s",
    "is_gut_cyp3a4_substrate",
    "segment_fg",
]
