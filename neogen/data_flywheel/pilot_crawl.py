"""Bounded P0 pilot crawl over public biomedical landing/docs pages.

Uses the Firecrawl CLOUD API (key read from the ``FIRECRAWL_API_KEY`` env var,
never hardcoded, never written to disk). Extraction is **metadata-only**:
we store title/description/status, a short snippet, a content hash, char/link
counts, and discovered links — NOT full copyrighted page bodies.

Safety:
- Only sources with ``crawl_allowed: true`` are fetched.
- Honors a small inter-request delay (politeness).
- Writes raw metadata to ``data/flywheel/raw/<source_id>/`` and a normalized
  summary to ``data/flywheel/normalized/<source_id>.json``.
- The API key is never logged or written to any output file.

Run:
    FIRECRAWL_API_KEY=fc-... .venv/bin/python -m neogen.data_flywheel.pilot_crawl
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from neogen.data_flywheel.source_registry import load_registry

_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _REPO_ROOT / "data" / "flywheel" / "raw"
NORM_DIR = _REPO_ROOT / "data" / "flywheel" / "normalized"
FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape"
REQUEST_DELAY_S = 2.0  # politeness between requests
SNIPPET_CHARS = 400
MAX_LINKS_STORED = 40

# P0 pilot set (matches the registry P0 + the V37 plan).
PILOT_SOURCE_IDS = [
    "fda_cyp_transporter_tables",
    "dailymed_spl",
    "openfda_labels",
    "pubchem",
    "drugcentral",
    "ddinter",
    "chembl",
    "rxnorm",
    "unichem",
    "pk_db",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _firecrawl_scrape(url: str, api_key: str, timeout: int = 60) -> dict:
    """Call the Firecrawl cloud scrape API. Returns parsed JSON or raises."""
    body = json.dumps({"url": url, "formats": ["markdown", "links"],
                        "onlyMainContent": True}).encode("utf-8")
    req = urllib.request.Request(
        FIRECRAWL_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _summarize(source: dict, result: dict) -> dict:
    """Reduce a Firecrawl response to metadata-only (no full page body)."""
    data = result.get("data") or {}
    md = data.get("markdown") or ""
    meta = data.get("metadata") or {}
    links = data.get("links") or []
    sha = hashlib.sha256(md.encode("utf-8")).hexdigest()
    return {
        "source_id": source["source_id"],
        "source_name": source["name"],
        "category": source["category"],
        "url": source["source_url"],
        "retrieved_at": _now(),
        "tool_used": "firecrawl_cloud",
        "http_status": meta.get("statusCode"),
        "title": (meta.get("title") or "")[:200],
        "description": (meta.get("description") or "")[:500],
        "content_sha256": sha,
        "markdown_char_count": len(md),
        "link_count": len(links),
        "snippet": md[:SNIPPET_CHARS].replace("\n", " ").strip(),
        "discovered_links": links[:MAX_LINKS_STORED],
        "license_status": source["license_status"],
        "robots_tos_status": source["robots_tos_status"],
        "ingestion_priority": source["ingestion_priority"],
        "note": "metadata-only extraction; full page body intentionally not stored",
    }


def run_pilot(source_ids: list[str] | None = None) -> dict:
    """Run the bounded pilot crawl. Returns a run summary (no key, ever)."""
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FIRECRAWL_API_KEY not set in environment. Refusing to run."
        )

    reg = load_registry()
    ids = source_ids or PILOT_SOURCE_IDS
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for sid in ids:
        src = reg.get(sid)
        entry = {"source_id": sid}
        if src is None:
            entry.update(status="skipped", reason="not in registry")
            results.append(entry)
            continue
        if src.get("crawl_allowed") is not True:
            entry.update(status="skipped", reason="crawl_allowed=false")
            results.append(entry)
            continue
        url = src["source_url"]
        try:
            res = _firecrawl_scrape(url, api_key)
            if not res.get("success"):
                entry.update(status="error", reason=str(res.get("error"))[:200])
                results.append(entry)
                time.sleep(REQUEST_DELAY_S)
                continue
            summary = _summarize(src, res)
            raw_dir = RAW_DIR / sid
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "firecrawl_meta.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True)
            )
            (NORM_DIR / f"{sid}.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True)
            )
            entry.update(
                status="ok",
                http_status=summary["http_status"],
                title=summary["title"],
                chars=summary["markdown_char_count"],
                links=summary["link_count"],
            )
        except Exception as exc:  # pragma: no cover - network dependent
            entry.update(status="error", reason=str(exc)[:200])
        results.append(entry)
        time.sleep(REQUEST_DELAY_S)

    ok = [r for r in results if r.get("status") == "ok"]
    summary = {
        "crawl_id": f"pilot-{_now()}",
        "attempted": len(ids),
        "successful": len(ok),
        "results": results,
    }
    return summary


if __name__ == "__main__":  # pragma: no cover
    out = run_pilot()
    # Print summary WITHOUT any key material.
    print(json.dumps(out, indent=2))
