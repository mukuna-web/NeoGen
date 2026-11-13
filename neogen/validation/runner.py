"""ValidationRunner — top-level orchestrator for the P6 harness.

Runs any combination of tiers and assembles a ValidationReport.
This is the object the CLI calls; library users can also import
it directly:

    from neogen.validation import ValidationRunner
    report = ValidationRunner().run_all()
    print(report.to_markdown())
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from neogen.engine.orchestrator_v2 import ENGINE_VERSION
from neogen.validation.report import ValidationReport
from neogen.validation.tier1 import Tier1Harness, Tier1Report
from neogen.validation.tier2 import Tier2Harness, Tier2Report
from neogen.validation.tier3 import Tier3Harness, Tier3Report
from neogen.validation.tier4 import Tier4Harness, Tier4Report

logger = logging.getLogger(__name__)


# Known-limitation notes — surfaced in every report so consumers
# are not surprised by ~70% rather than 100% pass rates.
_KNOWN_NOTES: tuple = (
    "Tier 2 uses the low-level DDIOrchestrator + PBPKSolver path "
    "directly. OrchestratorV2 does not yet wire DDI end-to-end "
    "(SimulationResultV2.ddi is always None). Integration is "
    "scheduled for the P7 API phase.",
    "Tier 3 fa/fg/fh decomposition is reported only when the "
    "absorption engine exposes them through the orchestrator "
    "result. The empirical F from AUC_oral/AUC_iv is always "
    "available.",
    "Blind holdout Tier 1 cases that are not in the literature DB "
    "are built from SMILES alone via DrugFactory.create_from_smiles. "
    "This is the hardest mode — no literature override, no "
    "ML-predicted fu beyond the logP fallback.",
)


@dataclass(frozen=True)
class RunTimings:
    """Wall-clock timings per tier. Reported in the final report notes."""

    tier1_s: float = 0.0
    tier2_s: float = 0.0
    tier3_s: float = 0.0
    tier4_s: float = 0.0
    total_s: float = 0.0

    def to_note(self) -> str:
        return (
            f"Runtime (s): tier1={self.tier1_s:.1f}, "
            f"tier2={self.tier2_s:.1f}, "
            f"tier3={self.tier3_s:.1f}, "
            f"tier4={self.tier4_s:.1f}, "
            f"total={self.total_s:.1f}"
        )


class ValidationRunner:
    """Top-level validation harness runner.

    All tiers default-on in :meth:`run_all`. For faster iteration use
    the per-tier runners, which take roughly:

      * Tier 4:  ~1 second  (one reference IV sim + invariant checks)
      * Tier 2:  ~20-40 s   (5 pairs × 2 simulations each, long tails)
      * Tier 3:  ~5-15 s    (15 drugs × 2 simulations each, short tails)
      * Tier 1:  ~60-180 s  (50 drugs × 1 simulation each, many ML fallbacks)
    """

    def __init__(
        self,
        tier1: Optional[Tier1Harness] = None,
        tier2: Optional[Tier2Harness] = None,
        tier3: Optional[Tier3Harness] = None,
        tier4: Optional[Tier4Harness] = None,
    ) -> None:
        self._t1 = tier1 or Tier1Harness()
        self._t2 = tier2 or Tier2Harness()
        self._t3 = tier3 or Tier3Harness()
        self._t4 = tier4 or Tier4Harness()

    def run_tier1(self) -> Tier1Report:
        return self._t1.run()

    def run_tier2(self) -> Tier2Report:
        return self._t2.run()

    def run_tier3(self) -> Tier3Report:
        return self._t3.run()

    def run_tier4(self) -> Tier4Report:
        return self._t4.run()

    def run_all(
        self,
        *,
        include_tier1: bool = True,
        include_tier2: bool = True,
        include_tier3: bool = True,
        include_tier4: bool = True,
    ) -> ValidationReport:
        """Run selected tiers and assemble the composite report."""
        t_start = time.time()

        r4: Optional[Tier4Report] = None
        r1: Optional[Tier1Report] = None
        r2: Optional[Tier2Report] = None
        r3: Optional[Tier3Report] = None

        # Tier 4 first — if the engine is broken this fails fast.
        tier4_s = 0.0
        if include_tier4:
            logger.info("Running Tier 4 (invariants)...")
            t0 = time.time()
            r4 = self._t4.run()
            tier4_s = time.time() - t0
            logger.info(
                "Tier 4: %d/%d invariants passed (%.1fs)",
                r4.n_passed,
                r4.n_total,
                tier4_s,
            )

        tier2_s = 0.0
        if include_tier2:
            logger.info("Running Tier 2 (DDI pairs)...")
            t0 = time.time()
            r2 = self._t2.run()
            tier2_s = time.time() - t0
            logger.info(
                "Tier 2: %d/%d pairs within 2-fold (%.1f%%, %.1fs)",
                r2.n_passed,
                r2.n_total,
                r2.pct_passed,
                tier2_s,
            )

        tier3_s = 0.0
        if include_tier3:
            logger.info("Running Tier 3 (bioavailability)...")
            t0 = time.time()
            r3 = self._t3.run()
            tier3_s = time.time() - t0
            logger.info(
                "Tier 3: %d/%d cases within ±%.0f%% (%.1fs)",
                r3.n_passed,
                r3.n_total,
                r3.acceptance_band_pct,
                tier3_s,
            )

        tier1_s = 0.0
        if include_tier1:
            logger.info("Running Tier 1 (single-drug PK)...")
            t0 = time.time()
            r1 = self._t1.run()
            tier1_s = time.time() - t0
            logger.info(
                "Tier 1: %d/%d scorable cases passed (%.1f%%, %.1fs)",
                r1.n_passed,
                r1.n_scorable,
                r1.pct_passed,
                tier1_s,
            )

        total_s = time.time() - t_start
        timings = RunTimings(
            tier1_s=tier1_s,
            tier2_s=tier2_s,
            tier3_s=tier3_s,
            tier4_s=tier4_s,
            total_s=total_s,
        )

        notes = tuple(list(_KNOWN_NOTES) + [timings.to_note()])

        return ValidationReport(
            engine_version=ENGINE_VERSION,
            run_timestamp_utc=ValidationReport.now_timestamp(),
            tier1=r1,
            tier2=r2,
            tier3=r3,
            tier4=r4,
            notes=notes,
        )
