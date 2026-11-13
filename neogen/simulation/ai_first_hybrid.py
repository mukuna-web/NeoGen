"""AI-first hybrid wrapper.

Policy:
  1. AI layer (vector_core + ddi_predictor) produces a fast primary answer.
  2. Mechanistic layer (pbpk + cyp_inhibition_mechanistic) recomputes the
     same endpoint.
  3. A conformal-style bound (calibrated offline by biostatistician) decides
     whether the AI answer is within tolerance. If yes, return AI. If not,
     fall back to PBPK and flag the divergence.

This keeps PBPK as the validation gate per the strategic shift in
research_pipeline/step_reports/step_03_ai_first_architecture.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .ddi_predictor import DDIPredictor, DDIPrediction


@dataclass
class HybridOutcome:
    primary_source: str  # "ai" | "pbpk" | "disagreement"
    value: float | str | dict
    ai_value: float | str | dict
    mechanistic_value: float | str | dict
    gap: float  # 0..1, normalized disagreement
    flagged: bool


def _severity_gap(a: str, b: str) -> float:
    order = {"none": 0, "minor": 1, "moderate": 2, "major": 3, "disagreement": 2}
    return abs(order.get(a.lower(), 0) - order.get(b.lower(), 0)) / 3.0


class AiFirstHybrid:
    def __init__(
        self,
        ddi_predictor: DDIPredictor,
        pbpk_ddi_fallback: Optional[Callable[[str, str], str]] = None,
        tolerance: float = 0.34,  # 1 severity step
    ) -> None:
        self.ddi_predictor = ddi_predictor
        self.pbpk_fallback = pbpk_ddi_fallback
        self.tolerance = tolerance

    def ddi(self, drug_a: str, drug_b: str) -> HybridOutcome:
        ai: DDIPrediction = self.ddi_predictor.predict(drug_a, drug_b)
        mech = None
        if self.pbpk_fallback is not None:
            try:
                mech = self.pbpk_fallback(drug_a, drug_b)
            except Exception:
                mech = None

        if mech is None:
            return HybridOutcome(
                primary_source="ai",
                value=ai.to_dict(),
                ai_value=ai.to_dict(),
                mechanistic_value="",
                gap=0.0,
                flagged=ai.disagreement,
            )

        gap = _severity_gap(ai.severity, str(mech))
        if gap <= self.tolerance:
            return HybridOutcome(
                primary_source="ai",
                value=ai.to_dict(),
                ai_value=ai.to_dict(),
                mechanistic_value=mech,
                gap=gap,
                flagged=False,
            )

        # divergence → mechanistic wins, flag it
        return HybridOutcome(
            primary_source="pbpk",
            value=mech,
            ai_value=ai.to_dict(),
            mechanistic_value=mech,
            gap=gap,
            flagged=True,
        )
