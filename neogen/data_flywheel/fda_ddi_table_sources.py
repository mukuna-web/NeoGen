"""FDA DDI table source documents: fetch + provenance + content hash (GAP-014).

Two official, public-domain FDA pages drive structured CYP/transporter
directionality:

- ``FDA_CLINICAL_EXAMPLES_URL`` — the per-drug clinical matrix (each row is a
  drug; cells encode ``<entity> <strength> <role>``). This is the authoritative
  source for "drug X is a substrate/inhibitor/inducer of enzyme/transporter Y".
- ``FDA_DESIGN_TABLES_URL`` — the study-design tables, including the in vitro
  ``| Enzyme | Inhibitor |`` selective-inhibitor table (lower confidence).

The static HTML is the source of record: Firecrawl markdown truncates the
clinical page mid-alphabet, so we fetch and hash the raw ``<table>`` HTML.
Fetching is injectable (``fetch_fn``) so tests run fully offline.

No claim of FDA approval/clearance/clinical-validity. These tables are
mechanistic classification resources, not severity labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from neogen.data_flywheel.provenance import content_hash

FDA_CLINICAL_EXAMPLES_URL = (
    "https://www.fda.gov/drugs/drug-interactions-labeling/"
    "healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems"
)
FDA_DESIGN_TABLES_URL = (
    "https://www.fda.gov/drugs/drug-interactions-labeling/"
    "drug-development-and-drug-interactions-table-substrates-inhibitors-and-inducers"
)

# Public-domain US government data. Not a license to assert clinical severity.
FDA_LICENSE_STATUS = "open-us-gov-data"

_USER_AGENT = "Mozilla/5.0 (research; NeoGen data flywheel; +contact via repo)"

FetchFn = "callable[[str], tuple[int, str]]"  # (http_status, html_text)


@dataclass(frozen=True)
class FDASourceDoc:
    """An immutable fetched FDA page with provenance for replay."""

    source_id: str
    source_url: str
    retrieved_at: str          # ISO-8601 UTC
    http_status: int
    content_sha256: str        # hash of the raw html body (replayable)
    byte_count: int
    page_title: str
    table_name: str
    license_status: str
    html: str                  # raw body (kept for parse; written to raw/ on disk)

    def meta(self) -> dict:
        """Provenance/meta dict WITHOUT the heavy html body (for manifests)."""
        d = asdict(self)
        d.pop("html", None)
        return d


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_fetch(url: str) -> tuple[int, str]:
    """Fetch a public FDA page over HTTPS (stdlib only)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (https gov url)
        status = getattr(resp, "status", 200) or 200
        body = resp.read().decode("utf-8", errors="replace")
    return status, body


def fetch_fda_page(
    url: str,
    *,
    source_id: str,
    page_title: str,
    table_name: str,
    fetch_fn=None,
    retrieved_at: str | None = None,
) -> FDASourceDoc:
    """Fetch one FDA page and wrap it with provenance + content hash.

    ``fetch_fn(url) -> (http_status, html)`` is injectable for offline tests.
    Never fabricates content: a non-200 status is recorded as-is and the body
    is hashed exactly as received.
    """
    fetcher = fetch_fn or _default_fetch
    status, html = fetcher(url)
    return FDASourceDoc(
        source_id=source_id,
        source_url=url,
        retrieved_at=retrieved_at or _utc_now_iso(),
        http_status=status,
        content_sha256=content_hash(html),
        byte_count=len(html.encode("utf-8")),
        page_title=page_title,
        table_name=table_name,
        license_status=FDA_LICENSE_STATUS,
        html=html,
    )


def fetch_clinical_examples(fetch_fn=None, retrieved_at: str | None = None) -> FDASourceDoc:
    """Fetch the per-drug clinical CYP/transporter matrix page."""
    return fetch_fda_page(
        FDA_CLINICAL_EXAMPLES_URL,
        source_id="fda_clinical_examples_matrix",
        page_title=(
            "Healthcare Professionals: FDA's Examples of Drugs that Interact "
            "with CYP Enzymes and Transporter Systems"
        ),
        table_name="CYP/Transporter Clinical Substrates, Inhibitors, or Inducers",
        fetch_fn=fetch_fn,
        retrieved_at=retrieved_at,
    )


def fetch_design_tables(fetch_fn=None, retrieved_at: str | None = None) -> FDASourceDoc:
    """Fetch the study-design substrates/inhibitors/inducers tables page."""
    return fetch_fda_page(
        FDA_DESIGN_TABLES_URL,
        source_id="fda_design_substrate_inhibitor_inducer_tables",
        page_title="Drug Development and Drug Interactions | Table of Substrates, Inhibitors and Inducers | FDA",
        table_name="In vitro selective inhibitors for CYP-mediated metabolism",
        fetch_fn=fetch_fn,
        retrieved_at=retrieved_at,
    )
