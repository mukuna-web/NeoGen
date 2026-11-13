"""Parse FDA CYP/transporter DDI tables (markdown) into structured records.

Targets two official FDA pages:
- Clinical examples page: a per-drug matrix where each non-empty cell encodes
  ``<entity(ies)> <strength> <role>`` (e.g. "2C9 moderate inhibitor",
  "3A sensitive substrate", "P-gp inhibitor").
- In vitro design page: ``| Enzyme | Inhibitor |`` selective-inhibitor table
  (lower confidence; evidence_context=in_vitro).

Never silently drops row text; ambiguous cells are skipped (not guessed).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from neogen.data_flywheel.fda_ddi_table_schema import (
    FDADDITableRecord,
    classify_system,
)
from neogen.data_flywheel.provenance import content_hash


class _TableExtractor(HTMLParser):
    """Extract <tr>/<td|th> cell text from HTML tables (stdlib only)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._cur: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._cur = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "tr" and self._cur is not None:
            self.rows.append(self._cur)
            self._cur = None
        elif tag in ("td", "th") and self._cell is not None and self._cur is not None:
            self._cur.append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def extract_html_table_rows(html: str) -> list[list[str]]:
    p = _TableExtractor()
    p.feed(html)
    return [r for r in p.rows if r]

_ROLES = ("substrate", "inhibitor", "inducer")
_STRENGTHS = ("moderate sensitive", "strong", "moderate", "weak", "sensitive")
_CYP_TOKEN = re.compile(r"^\d+[A-Z]\d*$")  # 3A, 2C9, 2D6, 1A2, 2C19
# Canonical CYP families (longest-first) so footnote-contaminated tokens like
# "1A220" (= 1A2 + superscript footnote 20) resolve to CYP1A2 instead of a
# bogus CYP1A220. Unknown exotic families fall back to the permissive regex.
_KNOWN_CYP = sorted(
    ("1A1", "1A2", "1B1", "2A6", "2B6", "2C8", "2C9", "2C18", "2C19",
     "2D6", "2E1", "2J2", "3A4", "3A5", "3A7", "3A", "4F2", "4F12"),
    key=len, reverse=True,
)
# Trailing 1-2 digit footnote marker on a drug name, e.g. "warfarin19",
# "diltiazem7", "St. John's wort18". Bounded to <=2 digits so real names are safe.
_FOOTNOTE = re.compile(r"(?<=[A-Za-z])(\d{1,2})$")


def _split_footnote(name: str) -> tuple[str, str | None]:
    """Split a trailing footnote marker off a drug name.

    "warfarin19" -> ("warfarin", "19"); "amiodarone" -> ("amiodarone", None).
    Never drops the marker: it is returned for capture in ``footnotes``.
    """
    m = _FOOTNOTE.search(name)
    if not m:
        return name, None
    return name[: m.start()].strip(), m.group(1)


def normalize_entity(token: str) -> str | None:
    """Normalize one entity token to a canonical key, or None if unusable."""
    t = token.strip().upper().replace(" ", "")
    t = t.split("/")[0]  # "3A4/5" -> "3A4"; "MATE1/MATE-2K" -> "MATE1"
    if not t:
        return None
    if t.startswith("CYP"):
        t = t[3:]                   # normalize "CYP3A4" -> "3A4"
    # CYP family = a known family followed by nothing or only footnote DIGITS.
    # This strips "1A220" (1A2 + footnote 20) -> CYP1A2 without greedily grabbing
    # "3A" out of prose like "3Amoderately".
    for fam in _KNOWN_CYP:
        if t == fam or (t.startswith(fam) and t[len(fam):].isdigit()):
            return "CYP" + fam      # 3A -> CYP3A, 2C9 -> CYP2C9, 1A220 -> CYP1A2
    if _CYP_TOKEN.fullmatch(t):
        return "CYP" + t            # exotic family not in the known list
    if t in ("P-GP", "PGP", "ABCB1"):
        return "P-gp"
    # Known transporters, longest-prefix first; strips trailing footnote digits
    # like "OATP1B113" -> "OATP1B1".
    for pat in ("OATP1B1", "OATP1B3", "OATP", "OAT3", "OAT1", "OCT2",
                "MATE2", "MATE1", "BCRP", "BSEP"):
        if t.startswith(pat):
            return pat
    return None


def parse_cell(cell: str) -> list[tuple[str, str, str]]:
    """Parse one matrix cell -> list of (entity, strength, role). May be empty."""
    s = re.sub(r"\s+", " ", cell).strip()
    if not s:
        return []
    low = s.lower()
    role = None
    for r in _ROLES:
        if re.search(rf"\b{r}\b", low):
            role = r  # last match wins (cells are single-role in practice)
    if role is None:
        return []
    strength = "unknown"
    for st in _STRENGTHS:
        if re.search(rf"\b{re.escape(st)}\b", low):
            strength = st.replace(" ", "_")
            break
    # Entities = head before any strength/role keyword.
    cut = re.split(rf"\b({'|'.join(_STRENGTHS)}|{role})\b", low)[0]
    parts = re.split(r"[;,]", cut)
    out: list[tuple[str, str, str]] = []
    for p in parts:
        ent = normalize_entity(p)
        if ent:
            out.append((ent, strength, role))
    return out


def _table_rows(markdown: str) -> list[list[str]]:
    rows = []
    for line in markdown.splitlines():
        if line.count("|") >= 2 and not re.fullmatch(r"\s*\|[\s|:-]+\|\s*", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
    return rows


def _rows_to_records(
    rows: list[list[str]], *, source_url: str, retrieved_at: str, page_title: str
) -> list[FDADDITableRecord]:
    out: list[FDADDITableRecord] = []
    for cells in rows:
        if not cells:
            continue
        raw = cells[0].strip()
        # Skip section-header rows whose col0 is an enzyme/transporter name
        # (e.g. "P-gp", "OATP1B1/OATP1B3") — no drug is named that.
        head_ent = normalize_entity(raw)
        if head_ent and classify_system(head_ent) != "unknown":
            continue
        drug, footnote = _split_footnote(raw)
        # Skip header / legend / non-drug rows.
        if not drug or len(drug) > 60:
            continue
        if drug.lower() in (
            "drug", "drug or other substance", "potency category",
            "transporter system", "enzyme",
        ):
            continue
        fn = (f"footnote:{footnote}",) if footnote else tuple()
        for cell in cells[1:]:
            for ent, strength, role in parse_cell(cell):
                row_text = f"{raw} | {ent} {strength} {role}"
                out.append(FDADDITableRecord(
                    source_url=source_url, retrieved_at=retrieved_at,
                    page_title=page_title, table_id="clinical_table_1",
                    table_name="CYP/Transporter Clinical Substrates, Inhibitors, or Inducers",
                    table_category=("cyp" if classify_system(ent) == "CYP" else "transporter"),
                    system=classify_system(ent), entity=ent, gene_symbol=None,
                    drug_name=drug, normalized_drug_name=drug.lower(),
                    role_class=role,
                    evidence_context="clinical_example",
                    strength=strength,
                    footnotes=fn, row_text=row_text,
                    provenance_hash=content_hash(f"{source_url}|{drug.lower()}|{ent}|{role}|{strength}"),
                    reviewer_status="needs_review",
                ))
    return out


def parse_clinical_matrix(
    markdown: str, *, source_url: str, retrieved_at: str,
    page_title: str = "FDA Clinical Substrates, Inhibitors, Inducers",
) -> list[FDADDITableRecord]:
    """Parse the per-drug clinical matrix from markdown table text."""
    return _rows_to_records(
        _table_rows(markdown), source_url=source_url,
        retrieved_at=retrieved_at, page_title=page_title,
    )


def parse_clinical_matrix_html(
    html: str, *, source_url: str, retrieved_at: str,
    page_title: str = "FDA Clinical Substrates, Inhibitors, Inducers",
) -> list[FDADDITableRecord]:
    """Parse the per-drug clinical matrix from raw FDA HTML (full A-Z table).

    Static HTML is the authoritative source: Firecrawl markdown truncates this
    page mid-alphabet, so we parse the real ``<table>`` with the stdlib parser.
    """
    return _rows_to_records(
        extract_html_table_rows(html), source_url=source_url,
        retrieved_at=retrieved_at, page_title=page_title,
    )


_PAREN_FOOTNOTE = re.compile(r"\s*\([^)]*\)\s*$")  # trailing "(a)", "(4,5)"


def parse_invitro_inhibitors(
    text: str, *, source_url: str, retrieved_at: str,
    page_title: str = "FDA in vitro selective inhibitors",
) -> list[FDADDITableRecord]:
    """Parse the in vitro ``Enzyme | Inhibitor`` table (lower confidence).

    Accepts raw HTML or markdown. The FDA design page stacks several 2-column
    tables under ``Enzyme | <kind>`` headers (Marker reaction, Inhibitor,
    Inducer). We track the current section and collect ONLY rows under an
    *Inhibitor* header — marker reactions (e.g. "S-warfarin 7-hydroxylation")
    and inducers are deliberately excluded so a probe substrate is never
    misread as an inhibitor.
    """
    rows = (
        extract_html_table_rows(text) if "<table" in text.lower()
        else _table_rows(text)
    )
    out: list[FDADDITableRecord] = []
    section: str | None = None
    for cells in rows:
        if len(cells) < 2:
            continue
        col0, col1 = cells[0].strip(), cells[1].strip()
        if col0.lower() == "enzyme":
            section = col1.lower()        # "marker reaction" | "inhibitor" | "inducer"
            continue
        if section != "inhibitor":
            continue                      # skip marker-reaction / inducer sections
        ent = normalize_entity(col0)
        if not ent or classify_system(ent) != "CYP":
            continue
        for drug in re.split(r"[;,]", col1):
            drug = _PAREN_FOOTNOTE.sub("", drug).strip()
            if not drug:
                continue
            out.append(FDADDITableRecord(
                source_url=source_url, retrieved_at=retrieved_at,
                page_title=page_title, table_id="invitro_inhibitors",
                table_name="In vitro selective inhibitors for CYP-mediated metabolism",
                table_category="cyp", system="CYP", entity=ent, gene_symbol=None,
                drug_name=drug, normalized_drug_name=drug.lower(),
                role_class="inhibitor", evidence_context="in_vitro", strength="unknown",
                footnotes=tuple(), row_text=f"{col0} | {drug} (in vitro inhibitor)",
                provenance_hash=content_hash(f"{source_url}|invitro|{drug.lower()}|{ent}|inhibitor"),
                reviewer_status="needs_review",
            ))
    return out
