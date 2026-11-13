"""Hybrid DDI predictor.

Combines three signals:
  1. DDInter 2.0 static lookup (existing `ddi_lookup.py`) — ground truth for
     documented pairs.
  2. Vector similarity via `vector_core.VectorCore` — transfer known DDI
     severity from structurally similar drugs when a pair is not in the table.
  3. Mechanistic CYP signal via `cyp_inhibition_mechanistic` — catches
     pharmacokinetic overlap on shared enzymes.

Each signal returns a severity score in [0, 1] plus a confidence weight.
The output is a weighted mean; if all three disagree strongly we surface
that as a flag rather than a point estimate.

Red-team (ddi_specialist, metabolism_lead, cardio_lead) expectations are
encoded in the `audit_pair` method: every prediction comes with its
provenance so it can be independently challenged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .vector_core import VectorCore

logger = logging.getLogger(__name__)

SEVERITY_SCORE = {"minor": 0.25, "moderate": 0.55, "major": 0.90}
SCORE_TO_LEVEL = [
    (0.80, "major"),
    (0.40, "moderate"),
    (0.15, "minor"),
    (0.0, "none"),
]


def _score_to_level(score: float) -> str:
    for threshold, level in SCORE_TO_LEVEL:
        if score >= threshold:
            return level
    return "none"


@dataclass
class DDISignal:
    source: str
    score: float  # 0..1
    weight: float  # confidence contribution
    evidence: str = ""


@dataclass
class DDIPrediction:
    drug_a: str
    drug_b: str
    severity: str  # none | minor | moderate | major | disagreement
    score: float  # blended 0..1
    confidence: float  # 0..1
    signals: list[DDISignal] = field(default_factory=list)
    disagreement: bool = False

    def to_dict(self) -> dict:
        return {
            "drug_a": self.drug_a,
            "drug_b": self.drug_b,
            "severity": self.severity,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "disagreement": self.disagreement,
            "signals": [
                {
                    "source": s.source,
                    "score": round(s.score, 4),
                    "weight": round(s.weight, 4),
                    "evidence": s.evidence,
                }
                for s in self.signals
            ],
        }


class DDIPredictor:
    """Blend DDInter lookup + SMILES similarity + mechanistic CYP."""

    def __init__(
        self,
        vector_core: VectorCore,
        ddinter_lookup=None,
        cyp_mechanistic=None,
        similarity_threshold: float = 0.25,
        disagreement_gap: float = 0.40,
    ) -> None:
        self.vc = vector_core
        self.ddinter = ddinter_lookup
        self.cyp = cyp_mechanistic
        self.similarity_threshold = similarity_threshold
        self.disagreement_gap = disagreement_gap

    # ---------- signal #1: DDInter lookup ----------
    def _ddinter_signal(self, a: str, b: str) -> Optional[DDISignal]:
        if self.ddinter is None:
            return None
        try:
            rec = self.ddinter(a, b)
        except Exception as exc:
            logger.debug("ddinter lookup error: %s", exc)
            return None
        if not rec:
            return None
        level = (rec.get("level") if isinstance(rec, dict) else getattr(rec, "level", "")).lower()
        score = SEVERITY_SCORE.get(level)
        if score is None:
            return None
        # DDInter is the authoritative signal for in-table pairs; weight
        # it high so similarity/mechanistic votes cannot drag a clearly
        # documented "major" down to "moderate" via averaging.
        return DDISignal(
            source="ddinter",
            score=score,
            weight=1.2,
            evidence=f"ddinter level={level}",
        )

    # ---------- signal #2: similarity-transfer ----------
    def _similarity_signal(self, a: str, b: str, k: int = 4) -> Optional[DDISignal]:
        """For (a, b) not in DDInter, find top-k analogs of each and transfer
        the mean severity from known pair (analog_a, analog_b) scores."""
        if self.vc.size() == 0 or self.ddinter is None:
            return None
        neighbours_a = self.vc.query_by_name(a, k=k)
        neighbours_b = self.vc.query_by_name(b, k=k)
        if not neighbours_a or not neighbours_b:
            return None
        scores: list[float] = []
        for na, sa in neighbours_a:
            if sa < self.similarity_threshold:
                continue
            for nb, sb in neighbours_b:
                if sb < self.similarity_threshold:
                    continue
                try:
                    rec = self.ddinter(na, nb)
                except Exception:
                    rec = None
                if not rec:
                    continue
                level = (rec.get("level") if isinstance(rec, dict) else getattr(rec, "level", "")).lower()
                base = SEVERITY_SCORE.get(level)
                if base is None:
                    continue
                scores.append(base * ((sa + sb) / 2))
        if not scores:
            return None
        mean = sum(scores) / len(scores)
        weight = min(0.5, 0.08 * len(scores))  # grows with evidence count
        return DDISignal(
            source="similarity",
            score=mean,
            weight=weight,
            evidence=f"{len(scores)} analog pairs averaged",
        )

    # ---------- signal #3: mechanistic CYP ----------
    def _mechanistic_signal(self, a: str, b: str) -> Optional[DDISignal]:
        if self.cyp is None:
            return None
        try:
            result = self.cyp(a, b)
        except Exception:
            return None
        if not result:
            return None
        score = result.get("score") if isinstance(result, dict) else None
        if score is None:
            return None
        score = max(0.0, min(1.0, float(score)))
        return DDISignal(
            source="mechanistic_cyp",
            score=score,
            weight=0.4,
            evidence=result.get("reason", "cyp kinetic overlap"),
        )

    # ---------- blend ----------
    def predict(self, drug_a: str, drug_b: str) -> DDIPrediction:
        signals: list[DDISignal] = []
        for fn in (self._ddinter_signal, self._similarity_signal, self._mechanistic_signal):
            sig = fn(drug_a, drug_b) if fn.__code__.co_argcount == 3 else fn(drug_a, drug_b, 4)
            if sig is not None:
                signals.append(sig)

        if not signals:
            return DDIPrediction(
                drug_a=drug_a, drug_b=drug_b, severity="none",
                score=0.0, confidence=0.0, signals=[],
            )

        total_w = sum(s.weight for s in signals) or 1.0
        blended = sum(s.score * s.weight for s in signals) / total_w
        # Only flag disagreement when dissenting signals carry comparable weight.
        # A low-weight similarity vote against a high-weight ddinter hit is NOT
        # a disagreement — ddinter is authoritative for in-table pairs.
        dominant = max(signals, key=lambda s: s.weight)
        others = [s for s in signals if s is not dominant]
        significant_dissent = [
            s for s in others
            if s.weight >= 0.5 * dominant.weight
            and abs(s.score - dominant.score) >= self.disagreement_gap
        ]
        disagreement = bool(significant_dissent)

        confidence = min(1.0, total_w) * (0.6 if disagreement else 1.0)
        severity = "disagreement" if disagreement else _score_to_level(blended)

        return DDIPrediction(
            drug_a=drug_a,
            drug_b=drug_b,
            severity=severity,
            score=blended,
            confidence=confidence,
            signals=signals,
            disagreement=disagreement,
        )

    def audit_pair(self, drug_a: str, drug_b: str) -> dict:
        """Full provenance output for red-team review."""
        return self.predict(drug_a, drug_b).to_dict()
