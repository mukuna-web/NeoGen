"""Hallucination / risky-claim sweep over code, reports, artifacts, and feeds.

Detects regulatory over-claims, uncited clinical language, fake human review,
placeholder/hallucinated citations, and unbenchmarked performance claims.

Crucially, a match is **allowed** (not a problem) when it sits in an attributed
context: a disclaimer/negation ("No FDA approval ... claim"), a quoted source
span, a cited line (source_url / http / setid), or a JSON citation field. This
keeps the sweep from flagging the project's own safety disclaimers or quoted
FDA/label text — only *unattributed* claims become findings.
"""

from __future__ import annotations

import re
from pathlib import Path

from neogen.accuracy.contracts import CodeSweepResult, HallucinationFinding

# (name, regex, severity, tier). Tier A = regulatory/medical over-claim, dangerous
# anywhere. Tier B = clinical-modeling vocabulary, expected inside the DDI engine
# as enum/field/regex constructs; only a problem in *prose* (reports/data/chats).
RISKY_PATTERNS: list[tuple[str, re.Pattern, str, str]] = [
    ("fda approved", re.compile(r"\bfda[ \-]?(approv|clear)\w*", re.I), "high", "A"),
    ("fda accepted", re.compile(r"\bfda[ \-]?accept\w*", re.I), "high", "A"),
    # claim form only ("clinically validated"/"clinical validity"), NOT the noun
    # "clinical validation" (an activity / dashboard name).
    ("clinically validated", re.compile(
        r"\bclinical(?:ly|-)?\s*validat(?:ed|ity)\b|\bclinically[ \-]proven\b", re.I), "high", "A"),
    ("medical advice", re.compile(r"\bmedical advice\b", re.I), "high", "A"),
    ("dosage recommendation", re.compile(r"\b(dosag|dosing)\w*[ \-]recommendation\w*", re.I), "high", "A"),
    ("clinically significant", re.compile(r"\bclinical(ly)?[ \-]significan\w*", re.I), "medium", "B"),
    ("major interaction", re.compile(r"\bmajor[ \-]interaction\b", re.I), "medium", "B"),
    ("severe interaction", re.compile(r"\bsevere[ \-]interaction\b", re.I), "medium", "B"),
    ("contraindicated", re.compile(r"\bcontraindicat\w*", re.I), "medium", "B"),
    # NB: fake human review is checked precisely in evidence_audit (a machine
    # reviewer performing human_reviewed) — a phrase sweep for "human reviewed"
    # is too noisy (it matches every description of the review workflow).
]

# Unambiguous fake-citation / placeholder-content markers (always flagged).
PLACEHOLDER_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("placeholder citation", re.compile(
        r"\[citation needed\]|\blorem ipsum\b|REPLACE_ME|example\.com|"
        r"doi:0(?!\.)|<source[ >]", re.I), "high"),
]

# A line carrying one of these is a disclaimer/notice — every risky match on it
# is attributed (handles long "does not claim FDA approval, ..., clinical
# validity" lists where the negation is far from the last item).
_DISCLAIMER_LINE = (
    "does not claim", "do not claim", "not claim", "no claim", "not a claim",
    "disclaimer", "regulatory wording", "research/prototype", "research / prototype",
    "not fda", "no fda", "not medical advice", "no medical", "no severity",
    "≠", "prototype scope",
)
# A heading/line that opens a "what must NOT be claimed" / caveats block — the
# following bulleted items deny claims rather than assert them.
_DISCLAIMER_BLOCK = (
    "must not be claimed", "must not claim", "what still must not", "do not claim",
    "what not to claim", "caveat", "limitation", "safety rules", "what we do not claim",
)
# "FDA-approved drugs/medications/products/labels" describes the drug universe,
# not a claim that NeoGen is approved.
_FDA_DRUG_NOUN = re.compile(r"\s*[-]?(drug|drugs|medication|medications|product|products|label|labels|list)\b", re.I)

# "placeholder/TBD/TODO/FIXME" is only a *citation* problem when it sits next to
# a citation/source/evidence word — otherwise it is ordinary code scaffolding.
_SCOPED_PLACEHOLDER = re.compile(r"\b(placeholder|tbd|fixme|todo)\b", re.I)
_PROVENANCE_WORD = re.compile(
    r"\b(citation|source_url|source|provenance|evidence|reference|url|doi|spl)\b", re.I)

# Unbenchmarked performance claim: a percent / SOTA / outperform claim sitting
# next to an ML-metric word, with no test/benchmark reference. ("beats" is
# deliberately excluded — it matches cardiac "heart beats" / "per beat".)
_PERF_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?\s*%|state[ \-]of[ \-]the[ \-]art|\bsota\b|outperform\w*)", re.I)
_PERF_CONTEXT = re.compile(r"\b(accuracy|precision|recall|f1|auc|state[ \-]of[ \-]the[ \-]art|sota|outperform)\w*", re.I)

_NEGATION = (
    "no ", "not ", "never", "n't", "without", "cannot", "can't", "must not",
    "do not", "don't", "avoid", "prohibit", "disclaim", "≠", "!=", "no-claim",
    "no claim", "is not", "are not", "nor ", "neither",
)
_CITATION_TOKENS = (
    "http", "source_url", "fda.gov", "dailymed", "setid=", "doi", "pubmed",
    "nlm.nih.gov", "rxnav", "pubchem",
)
_JSON_CITATION_KEYS = (
    "citation_span", "evidence_text", "row_text", "citation", "snippet",
    "matched_span", "quote", "source_text", "note", "description", "title",
    "table_name", "page_title", "reason", "evidence_level",
)

_SKIP_DIRS = {"node_modules", ".venv", ".git", "__pycache__", ".next", "dist", "build"}


def _window_before(line: str, idx: int, n: int = 64) -> str:
    return line[max(0, idx - n): idx].lower()


def is_allowed_context(line: str, start: int, file_path: str) -> tuple[bool, str]:
    """Decide whether a risky match at ``start`` is attributed (allowed)."""
    low = line.lower()
    before = _window_before(line, start)
    # 1. disclaimer / negation near the match
    if any(tok in before for tok in _NEGATION) or "≠" in low:
        return True, "disclaimed/negated in context"
    # 2. inside a quoted span on this line
    pre = line[:start]
    if pre.count('"') % 2 == 1 or pre.count("“") > pre.count("”"):
        return True, "inside quoted source span"
    # 3. JSON citation field anywhere on this line (pretty-printed JSON keeps a
    #    field's key and quoted source value on one line).
    if file_path.endswith((".json", ".jsonl")):
        if any(f'"{k}"' in low for k in _JSON_CITATION_KEYS):
            return True, "JSON citation/source field"
    # 4. cited line (carries a source attribution)
    if any(tok in low for tok in _CITATION_TOKENS):
        return True, "cited line (source attribution present)"
    return False, "unattributed claim"


def scan_text(text: str, file_path: str) -> list[HallucinationFinding]:
    """Scan one document's text for risky/placeholder/perf claims.

    Python source is scanned in *code mode*: clinical-modeling vocabulary (Tier
    B) is the DDI engine's design and is skipped; only regulatory/medical
    over-claims (Tier A) are checked. Prose (md/json/txt/log) is scanned fully.
    """
    # Code + config files: Tier B (clinical-modeling vocabulary) is design, not a
    # claim — scan Tier A only.
    code_mode = file_path.endswith((".py", ".yaml", ".yml"))
    out: list[HallucinationFinding] = []
    disclaimer_block = 0   # lines remaining in a "must not be claimed" block
    for lineno, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        if any(t in low for t in _DISCLAIMER_BLOCK):
            disclaimer_block = 10
        elif line.lstrip().startswith("#") and "##" in line:
            disclaimer_block = 0   # new section heading ends the block
        disclaimer = any(p in low for p in _DISCLAIMER_LINE) or disclaimer_block > 0
        if disclaimer_block > 0:
            disclaimer_block -= 1
        for name, rx, sev, tier in RISKY_PATTERNS:
            if code_mode and tier == "B":
                continue  # modeling vocabulary in code is not a claim
            for m in rx.finditer(line):
                if disclaimer:
                    allowed, reason = True, "disclaimer/notice line"
                elif m.group(0).isupper() and len(m.group(0)) > 3:
                    # CONTRAINDICATIONS / WARNINGS = FDA label section name or an
                    # enum constant, not an asserted drug claim.
                    allowed, reason = True, "uppercase section-name/enum reference"
                elif name == "fda approved" and _FDA_DRUG_NOUN.match(line[m.end():]):
                    allowed, reason = True, "describes FDA-approved drug universe, not NeoGen"
                else:
                    allowed, reason = is_allowed_context(line, m.start(), file_path)
                out.append(HallucinationFinding(
                    pattern=name, matched_text=line.strip()[:200], file_path=file_path,
                    line=lineno, severity=sev, allowed=allowed, reason=reason,
                ))
        for name, rx, sev in PLACEHOLDER_PATTERNS:
            for m in rx.finditer(line):
                # Unambiguous fake citations are never excused by quoting.
                out.append(HallucinationFinding(
                    pattern=name, matched_text=line.strip()[:200], file_path=file_path,
                    line=lineno, severity=sev, allowed=False,
                    reason="placeholder/hallucinated citation marker",
                ))
        # placeholder/TBD/TODO next to a citation/source word == fake provenance;
        # a bare "# Placeholder for model" code comment is NOT flagged.
        if _SCOPED_PLACEHOLDER.search(line) and _PROVENANCE_WORD.search(line):
            out.append(HallucinationFinding(
                pattern="placeholder in citation context", matched_text=line.strip()[:200],
                file_path=file_path, line=lineno, severity="high", allowed=False,
                reason="placeholder/TBD adjacent to a citation/source field",
            ))
        # performance claim without a benchmark/test reference on the line.
        # Skips format-spec percentages like "{x:.2%}", quoted/cited spans, and
        # quoted FDA source values (a % inside a citation_span is the label's,
        # not a NeoGen accuracy claim). Code comments are low; reports medium.
        perf_m = _PERF_RE.search(line)
        is_format_spec = bool(perf_m and re.search(r":\.?\d*%|:%", line))
        if perf_m and _PERF_CONTEXT.search(line) and not is_format_spec:
            allowed, _why = is_allowed_context(line, perf_m.start(), file_path)
            cited = any(t in low for t in ("test", "benchmark", "dataset", "n=",
                                           "pytest", "passed", "holdout", "cohort", "gold-standard"))
            if not allowed and not cited and not disclaimer:
                out.append(HallucinationFinding(
                    pattern="unbenchmarked performance claim",
                    matched_text=line.strip()[:200], file_path=file_path,
                    line=lineno, severity="medium" if file_path.endswith(".md") else "low",
                    allowed=False,
                    reason="performance/accuracy claim without a test/benchmark reference",
                ))
    return out


def iter_target_files(root: Path, suffixes: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[Path]:
    """Collect files under ``root`` with the given suffixes, skipping junk dirs."""
    out: list[Path] = []
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix in suffixes else []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in suffixes:
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if any(ex in str(p) for ex in exclude):
            continue
        out.append(p)
    return out


def scan_paths(paths: list[Path], *, exclude: tuple[str, ...] = ()) -> CodeSweepResult:
    """Run the sweep over a list of files/dirs, return an aggregate result."""
    findings = []
    risky = allowed = flagged = 0
    files = 0
    seen: set[str] = set()
    suffixes = (".py", ".md", ".json", ".jsonl", ".log", ".txt", ".yaml", ".yml")
    targets: list[Path] = []
    for p in paths:
        for f in iter_target_files(Path(p), suffixes, exclude=exclude):
            if str(f) not in seen:
                seen.add(str(f))
                targets.append(f)
    for f in targets:
        files += 1
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        for hf in scan_text(text, str(f)):
            risky += 1
            if hf.allowed:
                allowed += 1
            else:
                flagged += 1
                findings.append(hf.to_finding())
    return CodeSweepResult(
        files_scanned=files, risky_matches=risky, flagged=flagged,
        allowed=allowed, findings=tuple(findings),
    )
