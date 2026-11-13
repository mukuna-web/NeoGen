"""DailyMed SPL verification for label-derived DDI evidence.

Confirms that the openFDA-derived evidence resolves to a real, authoritative
DailyMed Structured Product Label (SPL) by ``set_id``, using the official
DailyMed v2 services API. Stores set_id, source URL, SPL version/title, a
provenance hash, and a ``verified_against_spl`` flag.

No full label bodies are stored. The fetcher is injectable so tests run offline.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from neogen.data_flywheel.provenance import content_hash

DAILYMED_SPL_META = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?setid={set_id}"
DAILYMED_SETID_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"

# Fetcher signature: (set_id) -> metadata dict or None on failure.
Fetcher = Callable[[str], dict | None]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spl_metadata_url(set_id: str) -> str:
    return DAILYMED_SPL_META.format(set_id=set_id)


def spl_source_url(set_id: str) -> str:
    return DAILYMED_SETID_URL.format(set_id=set_id)


def _http_fetch(set_id: str, timeout: int = 30) -> dict | None:  # pragma: no cover - network
    url = spl_metadata_url(set_id)
    req = urllib.request.Request(url, headers={"User-Agent": "neogen-flywheel/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


@dataclass(frozen=True)
class SPLVerification:
    set_id: str
    verified_against_spl: bool
    source_url: str
    verified_at: str
    title: str | None = None
    spl_version: str | None = None
    published_date: str | None = None
    content_hash: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "set_id": self.set_id,
            "verified_against_spl": self.verified_against_spl,
            "source_url": self.source_url,
            "verified_at": self.verified_at,
            "title": self.title,
            "spl_version": self.spl_version,
            "published_date": self.published_date,
            "content_hash": self.content_hash,
            "note": self.note,
        }


def verify_set_id(set_id: str, *, fetcher: Fetcher | None = None) -> SPLVerification:
    """Verify one set_id against DailyMed. ``fetcher`` is injectable for tests."""
    fetch = fetcher or _http_fetch
    meta = fetch(set_id)
    src = spl_source_url(set_id)
    if not meta:
        return SPLVerification(
            set_id=set_id,
            verified_against_spl=False,
            source_url=src,
            verified_at=_now(),
            note="DailyMed metadata not retrieved",
        )
    # DailyMed v2 returns {"data": [ {...} ], "metadata": {...}}. Take the
    # first matching SPL record; tolerate a dict for robustness.
    raw = meta.get("data")
    if isinstance(raw, list):
        if not raw:
            return SPLVerification(
                set_id=set_id,
                verified_against_spl=False,
                source_url=src,
                verified_at=_now(),
                note="DailyMed returned no SPL for this set_id",
            )
        data = raw[0]
    elif isinstance(raw, dict):
        data = raw
    else:
        data = meta
    title = (data.get("title") or "").strip() or None
    spl_version = data.get("spl_version")
    published = data.get("published_date") or data.get("published")
    # Deterministic provenance hash over the stable identity of the SPL.
    chash = content_hash(f"{set_id}|{spl_version}|{title}")
    return SPLVerification(
        set_id=set_id,
        verified_against_spl=True,
        source_url=src,
        verified_at=_now(),
        title=title,
        spl_version=str(spl_version) if spl_version is not None else None,
        published_date=str(published) if published is not None else None,
        content_hash=chash,
        note="set_id resolves to an authoritative DailyMed SPL",
    )


def verify_set_ids(set_ids: list[str], *, fetcher: Fetcher | None = None) -> dict[str, SPLVerification]:
    out: dict[str, SPLVerification] = {}
    for sid in dict.fromkeys(set_ids):  # de-dup, preserve order
        out[sid] = verify_set_id(sid, fetcher=fetcher)
    return out
