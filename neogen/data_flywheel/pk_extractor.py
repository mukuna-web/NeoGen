"""Extract AUC/Cmax exposure changes from label text — quantitative only (GAP-008).

Hard rules:
- Only parse a number as AUC/Cmax if an AUC/Cmax token is nearby.
- Distinguish fold ("1.5-fold") from percent ("increased by 50%").
- Distinguish AUC from Cmax.
- Never invent a value; a warning with no number yields nothing.
- Direction must be explicit (increase/decrease).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_AUC = r"AUC(?:\s?0\s?[-–]\s?(?:inf|∞|t|24|12))?"
_CMAX = r"C\s?max"
_METRIC = re.compile(rf"({_AUC}|{_CMAX})", re.IGNORECASE)
_FOLD = re.compile(
    r"(increased|decreased|reduced|elevated)?[^.]{0,40}?(\d+(?:\.\d+)?)\s*[-–]?\s*fold",
    re.IGNORECASE,
)
_PERCENT = re.compile(
    r"(increased|decreased|reduced|elevated)\s+(?:by\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_WINDOW = 90  # chars around a metric mention to search for a magnitude


@dataclass(frozen=True)
class PKEffect:
    metric: str          # AUC | Cmax
    kind: str            # fold | percent
    magnitude: float
    direction: str       # increase | decrease
    ratio: float | None  # normalized fold-ratio when derivable, else None
    span: str


def _norm_metric(raw: str) -> str:
    return "Cmax" if raw.lower().replace(" ", "").startswith("cmax") else "AUC"


def _direction(word: str | None) -> str | None:
    if not word:
        return None
    w = word.lower()
    if w in ("increased", "elevated"):
        return "increase"
    if w in ("decreased", "reduced"):
        return "decrease"
    return None


def extract_pk_effects(text: str) -> list[PKEffect]:
    """Return explicit AUC/Cmax effects found in text (possibly empty)."""
    if not text:
        return []
    out: list[PKEffect] = []
    seen: set[tuple] = set()
    for m in _METRIC.finditer(text):
        metric = _norm_metric(m.group(0))
        lo = max(0, m.start() - _WINDOW)
        hi = min(len(text), m.end() + _WINDOW)
        window = text[lo:hi]

        fold = _FOLD.search(window)
        if fold:
            direction = _direction(fold.group(1)) or _infer_dir_from_window(window)
            mag = float(fold.group(2))
            if direction:
                ratio = mag if direction == "increase" else (1.0 / mag if mag else None)
                key = (metric, "fold", mag, direction)
                if key not in seen:
                    seen.add(key)
                    out.append(PKEffect(metric, "fold", mag, direction, ratio, window.strip()[:200]))
                continue

        pct = _PERCENT.search(window)
        if pct:
            direction = _direction(pct.group(1))
            mag = float(pct.group(2))
            if direction:
                ratio = (1 + mag / 100.0) if direction == "increase" else max(0.0, 1 - mag / 100.0)
                key = (metric, "percent", mag, direction)
                if key not in seen:
                    seen.add(key)
                    out.append(PKEffect(metric, "percent", mag, direction, ratio, window.strip()[:200]))
    return out


def _infer_dir_from_window(window: str) -> str | None:
    w = window.lower()
    if any(t in w for t in ("increas", "elevat", "higher", "rais")):
        return "increase"
    if any(t in w for t in ("decreas", "reduc", "lower")):
        return "decrease"
    return None
