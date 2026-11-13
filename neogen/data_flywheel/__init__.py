"""NeoGen data-flywheel package.

The data flywheel is the bounded, cited, replayable ingestion layer that feeds
the DDI pipeline. It is intentionally separated from the DDI *science engine*
(`neogen/engine/ddi/`): this package handles **provenance, identity, evidence,
and crawl bookkeeping**, not pharmacology simulation.

Safety posture (enforced in code where possible):
- No DDI severity without a citation (see ``evidence_schema``).
- Every ingested datum carries provenance (see ``provenance``).
- The source registry catalogs sources; it never authorizes a crawl.
"""

from __future__ import annotations

__all__ = [
    "source_registry",
    "provenance",
    "identity_spine",
    "evidence_schema",
    "crawl_manifest",
]

__version__ = "0.1.0"
