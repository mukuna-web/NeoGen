"""openFDA drug-label fetch layer.

Thin, dependency-free client for the openFDA drug/label API
(https://open.fda.gov/apis/drug/label/). openFDA is public; an API key is
optional (it only raises rate limits) and is NEVER logged.

Fetching is intentionally separated from parsing so the extractor/normalizer
(and their tests) can run fully offline on label JSON dicts.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

OPENFDA_LABEL_ENDPOINT = "https://api.fda.gov/drug/label.json"
DAILYMED_SETID_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"


def build_query_url(generic_name: str, limit: int = 1, api_key: str | None = None) -> str:
    """Build an openFDA label query URL for one generic name.

    The api_key, if present, is added as a query param. Callers must not log
    the returned URL when an api_key is included.
    """
    params = {
        "search": f'openfda.generic_name:"{generic_name}"',
        "limit": str(limit),
    }
    if api_key:
        params["api_key"] = api_key
    return f"{OPENFDA_LABEL_ENDPOINT}?{urllib.parse.urlencode(params)}"


def fetch_label(
    generic_name: str,
    *,
    api_key: str | None = None,
    limit: int = 1,
    timeout: int = 30,
) -> dict:
    """Fetch a label JSON from openFDA. Returns the parsed JSON (with results).

    Raises ``LookupError`` if the API reports no matches, ``RuntimeError`` on
    transport/HTTP errors. Network-dependent; not exercised by unit tests.
    """
    url = build_query_url(generic_name, limit=limit, api_key=api_key)
    req = urllib.request.Request(url, headers={"User-Agent": "neogen-flywheel/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        if exc.code == 404:
            raise LookupError(f"openFDA: no label found for {generic_name!r}") from exc
        raise RuntimeError(f"openFDA HTTP {exc.code} for {generic_name!r}") from exc
    except Exception as exc:  # pragma: no cover - network
        raise RuntimeError(f"openFDA fetch failed for {generic_name!r}: {exc}") from exc
    if not data.get("results"):
        raise LookupError(f"openFDA: empty results for {generic_name!r}")
    return data


def first_result(label_json: dict) -> dict:
    """Return the first label result mapping."""
    results = label_json.get("results") or []
    if not results:
        raise LookupError("label_json has no results")
    return results[0]


def label_openfda_ids(result: dict) -> dict:
    """Extract identity fields from a label result's openfda block + top level."""
    of = result.get("openfda") or {}

    def first(key: str) -> str | None:
        v = of.get(key)
        if isinstance(v, list) and v:
            return str(v[0])
        return None

    return {
        "set_id": result.get("set_id"),
        "spl_id": first("spl_id") or result.get("id"),
        "generic_name": first("generic_name"),
        "brand_name": first("brand_name"),
        "rxcui": first("rxcui"),
        "unii": first("unii"),
        "effective_time": result.get("effective_time"),
    }


def label_source_url(result: dict) -> str:
    """Best citable URL for a label result (DailyMed by set_id when available)."""
    set_id = result.get("set_id")
    if set_id:
        return DAILYMED_SETID_URL.format(set_id=set_id)
    return OPENFDA_LABEL_ENDPOINT
