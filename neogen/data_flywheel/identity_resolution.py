"""Drug-name -> RxCUI resolution (RxNorm) for the identity spine.

Uses the official RxNav REST API. Resolves a normalized drug name to an RxNorm
RxCUI and builds provisional ``IdentitySpineRecord`` objects. Unresolved names
are returned as ``None`` so the caller can open a data-gap entry rather than
fabricate an identifier.

The fetcher is injectable so tests run offline. No keys; RxNav is open.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Callable

from neogen.data_flywheel.identity_spine import IdentitySpineRecord

RXNAV_RXCUI = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name={name}"

# Fetcher: (name) -> parsed JSON dict or None.
Fetcher = Callable[[str], dict | None]


def rxcui_url(name: str) -> str:
    return RXNAV_RXCUI.format(name=urllib.parse.quote(name))


def _http_fetch(name: str, timeout: int = 20) -> dict | None:  # pragma: no cover - network
    req = urllib.request.Request(
        rxcui_url(name), headers={"User-Agent": "neogen-flywheel/0.1"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def resolve_rxcui(name: str, *, fetcher: Fetcher | None = None) -> str | None:
    """Return the first RxCUI for a drug name, or None if unresolved."""
    fetch = fetcher or _http_fetch
    data = fetch(name)
    if not data:
        return None
    ids = (data.get("idGroup") or {}).get("rxnormId") or []
    return str(ids[0]) if ids else None


def resolve_names(
    names: list[str], *, fetcher: Fetcher | None = None
) -> dict[str, str | None]:
    """Resolve many names; preserves order, de-dups (case-insensitive)."""
    out: dict[str, str | None] = {}
    for raw in names:
        key = raw.strip().lower()
        if not key or key in out:
            continue
        out[key] = resolve_rxcui(key, fetcher=fetcher)
    return out


def build_identity_spine(
    name_to_rxcui: dict[str, str | None]
) -> list[IdentitySpineRecord]:
    """Build provisional identity-spine records from resolved names.

    Resolved names get ``rxnorm_rxcui``; unresolved names still get a record
    (so the spine is complete) but with ``rxnorm_rxcui=None`` and a provisional
    internal id, signalling a gap.
    """
    records: list[IdentitySpineRecord] = []
    for name, rxcui in name_to_rxcui.items():
        internal = f"neogen-drug-{name}" if rxcui else f"neogen-drug-{name}-provisional"
        records.append(
            IdentitySpineRecord(
                internal_drug_id=internal,
                normalized_name=name,
                rxnorm_rxcui=rxcui,
                # structure ids (CID/InChIKey) are filled in a later cycle
            )
        )
    return records


def unresolved_names(name_to_rxcui: dict[str, str | None]) -> list[str]:
    return [n for n, r in name_to_rxcui.items() if not r]
