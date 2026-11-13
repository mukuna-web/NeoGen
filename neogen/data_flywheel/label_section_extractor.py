"""Extract interaction-relevant sections from openFDA label JSON.

Pulls DRUG INTERACTIONS, CLINICAL PHARMACOLOGY, WARNINGS, and CONTRAINDICATIONS
text, finds explicitly named partner drugs, and conservatively infers a
mechanism ONLY when the text uses explicit mechanism language.

No network. Pure functions over label dicts so tests run offline.
"""

from __future__ import annotations

import re

# openFDA label field -> canonical section name we expose.
SECTION_FIELDS: dict[str, str] = {
    "drug_interactions": "DRUG INTERACTIONS",
    "clinical_pharmacology": "CLINICAL PHARMACOLOGY",
    "warnings_and_cautions": "WARNINGS AND PRECAUTIONS",
    "warnings": "WARNINGS",
    "contraindications": "CONTRAINDICATIONS",
}

# Sections that are interaction-relevant by name.
INTERACTION_SECTION_NAMES = {
    "DRUG INTERACTIONS",
    "CLINICAL PHARMACOLOGY",
    "WARNINGS AND PRECAUTIONS",
    "WARNINGS",
    "CONTRAINDICATIONS",
}

# Conservative mechanism cues. Only set a mechanism when the text explicitly
# contains one of these phrases.
_MECHANISM_CUES: list[tuple[str, str]] = [
    (r"\binhibitor of cyp", "cyp_inhibition"),
    (r"\bcyp\w*\s+inhibitor", "cyp_inhibition"),
    (r"\binhibit(s|ion)?\b.*\bcyp", "cyp_inhibition"),
    (r"\binducer of cyp", "cyp_induction"),
    (r"\bcyp\w*\s+induc(er|tion)", "cyp_induction"),
    (r"\binduc(e|es|tion)\b.*\bcyp", "cyp_induction"),
    (r"\b(p-?gp|p-glycoprotein|oatp1b1|bcrp)\b.*inhibit", "transporter_inhibition"),
    (r"\b(p-?gp|p-glycoprotein|oatp1b1|bcrp)\b.*induc", "transporter_induction"),
]

_ENZYME_RE = re.compile(
    r"\bCYP\s?\d[A-Z]\d{0,2}\b|\bP-?gp\b|\bOATP1B1\b|\bBCRP\b", re.IGNORECASE
)


def extract_sections(result: dict) -> dict[str, str]:
    """Return {canonical_section_name: text} for present interaction sections."""
    out: dict[str, str] = {}
    for field, name in SECTION_FIELDS.items():
        val = result.get(field)
        if isinstance(val, list) and val:
            text = " ".join(str(x) for x in val).strip()
        elif isinstance(val, str):
            text = val.strip()
        else:
            text = ""
        if text:
            # Keep only the first occurrence per canonical name.
            out.setdefault(name, text)
    return out


def split_sentences(text: str) -> list[str]:
    """Naive sentence split good enough for snippet citation."""
    parts = re.split(r"(?<=[.;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def find_named_partners(
    text: str, partner_lexicon: list[str]
) -> list[tuple[str, str]]:
    """Find explicit mentions of partner drugs; return (partner, sentence).

    Only returns a partner when its name appears as a whole word in the text.
    The returned sentence is the citation span (truncated by the caller).
    """
    found: list[tuple[str, str]] = []
    sentences = split_sentences(text)
    seen: set[str] = set()
    for partner in partner_lexicon:
        pat = re.compile(rf"\b{re.escape(partner)}\b", re.IGNORECASE)
        for sent in sentences:
            if pat.search(sent):
                key = partner.lower()
                if key not in seen:
                    found.append((partner.lower(), sent))
                    seen.add(key)
                break
    return found


def infer_mechanism(text: str) -> str | None:
    """Return a mechanism ONLY if the text explicitly supports it, else None."""
    low = text.lower()
    for pattern, mech in _MECHANISM_CUES:
        if re.search(pattern, low):
            return mech
    return None


def find_enzymes(text: str) -> list[str]:
    """Return explicitly named enzymes/transporters (deduped, uppercased)."""
    hits = [m.group(0).upper().replace(" ", "") for m in _ENZYME_RE.finditer(text)]
    out: list[str] = []
    for h in hits:
        if h not in out:
            out.append(h)
    return out
