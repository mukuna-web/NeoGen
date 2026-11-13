"""Extract perpetrator/victim directionality ONLY when explicit (GAP-004).

Hard rule: direction is NEVER inferred from drug-pair ordering. It is assigned
only when a directional verb (inhibit/induce/increase/decrease exposure …)
explicitly links one named drug acting on the other. If both directions match →
bidirectional_needs_review. If neither → unknown (caller opens a gap).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Directional VERB forms only. Deliberately excludes the plural NOUNS
# "inhibitors"/"inducers", which appear as column headers in label interaction
# TABLES and must NOT be read as a directional action.
_DIR_VERB = (
    r"(inhibits?|inhibited|inhibiting|induces?|induced|inducing|"
    r"increases?|increased|increasing|decreases?|decreased|decreasing|"
    r"reduces?|reduced|reducing|elevates?|elevated|elevating|"
    r"potentiates?|potentiated|lowers?|lowered|raises?|raised)"
)

# Table/list signals: flattened interaction tables encode NO direction.
_LIST_SIGNALS = re.compile(
    r"examples of|inhibitors\s+inducers|inducers\s+inhibitors|"
    r"substrates\s+inhibitors|table of",
    re.IGNORECASE,
)


# Drug-class header tokens that mark a flattened interaction TABLE row.
_CLASS_TOKENS = re.compile(
    r"\b(beta-?blockers|antagonists|agonists|statins|substrates|"
    r"receptor antagonist|calcium channel)\b",
    re.IGNORECASE,
)


def looks_like_list(text: str) -> bool:
    """True if the span is a flattened table/list (no reliable direction)."""
    if _LIST_SIGNALS.search(text):
        return True
    if _CLASS_TOKENS.search(text):
        return True
    # Repeated "increase/decrease exposure" cells = a table, not a sentence.
    if len(re.findall(r"(increase|decrease)\s+exposure", text, re.IGNORECASE)) >= 2:
        return True
    return text.count(",") >= 4  # many comma-separated drug tokens -> list


@dataclass(frozen=True)
class Direction:
    perpetrator: str | None
    victim: str | None
    confidence: str  # explicit_label | bidirectional_needs_review | unknown
    matched_span: str | None


def _matches(text: str, a: str, b: str) -> bool:
    """True if '<a> <dir-verb> ... <b>' appears in the same sentence."""
    pat = re.compile(
        rf"\b{re.escape(a)}\b[^.;:]*\b{_DIR_VERB}\b[^.;:]*\b{re.escape(b)}\b",
        re.IGNORECASE,
    )
    return bool(pat.search(text))


def _span(text: str, a: str, b: str) -> str | None:
    pat = re.compile(
        rf"[^.;:]*\b{re.escape(a)}\b[^.;:]*\b{_DIR_VERB}\b[^.;:]*\b{re.escape(b)}\b[^.;:]*",
        re.IGNORECASE,
    )
    m = pat.search(text)
    return m.group(0).strip()[:300] if m else None


def extract_direction(text: str, drug_a: str, drug_b: str) -> Direction:
    """Return directionality, or unknown. Never uses argument order as a signal."""
    if not text:
        return Direction(None, None, "unknown", None)
    # Flattened tables/lists carry no reliable direction -> never assert.
    if looks_like_list(text):
        return Direction(None, None, "unknown", None)
    a_to_b = _matches(text, drug_a, drug_b)
    b_to_a = _matches(text, drug_b, drug_a)
    if a_to_b and b_to_a:
        return Direction(None, None, "bidirectional_needs_review", _span(text, drug_a, drug_b))
    if a_to_b:
        return Direction(drug_a, drug_b, "explicit_label", _span(text, drug_a, drug_b))
    if b_to_a:
        return Direction(drug_b, drug_a, "explicit_label", _span(text, drug_b, drug_a))
    return Direction(None, None, "unknown", None)
