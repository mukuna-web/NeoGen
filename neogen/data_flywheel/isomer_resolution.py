"""Enantiomer-aware structure resolution (GAP-011).

Resolves a drug *or enantiomer* name to a PubChem structure and detects the
stereochemistry-collapse failure mode: an explicitly enantiomeric query (e.g.
"(S)-warfarin") that returns a stereo-UNSPECIFIED parent. When that happens we
do not pretend it is stereo-resolved — we flag it for a gap.

Reuses ``structure_cache`` (PubChem fetch + RDKit stereo). Fetcher injectable
for offline tests. No clinical claims; no severity.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from neogen.data_flywheel.provenance import content_hash
from neogen.data_flywheel.structure_cache import (
    PUBCHEM_COMPOUND_URL,
    analyze_stereo,
    fetch_pubchem_props,
)

# Heuristics for "this query asked for a specific stereoisomer".
_ENANTIOMER_QUERY = re.compile(
    r"\(\s*[rs]\s*\)|\b(es|dex|levo|lev)[a-z]", re.IGNORECASE
)
# The InChIKey 2nd block for a fully stereo-unspecified molecule.
_UNSPECIFIED_STEREO_BLOCK = "UHFFFAOYSA"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def looks_enantiomeric(query: str) -> bool:
    return bool(_ENANTIOMER_QUERY.search(query))


def inchikey_blocks(inchikey: str | None) -> tuple[str | None, str | None]:
    if not inchikey:
        return None, None
    parts = inchikey.split("-")
    skeleton = parts[0] if len(parts) >= 1 else None
    stereo = parts[1] if len(parts) >= 2 else None
    return skeleton, stereo


@dataclass(frozen=True)
class IsomerResolution:
    query_name: str
    pubchem_cid: str | None
    canonical_smiles: str | None
    isomeric_smiles: str | None
    inchikey: str | None
    inchikey_skeleton: str | None
    inchikey_stereo_block: str | None
    n_stereocenters_total: int
    n_specified: int
    n_unspecified: int
    required_feature_schema: str
    stereo_unspecified: bool
    enantiomeric_query: bool
    enantiomer_query_collapsed: bool  # adversarial flag
    status: str  # resolved | resolved_unspecified | collapsed_needs_gap | unresolved
    source_url: str
    retrieved_at: str
    content_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_isomer(query_name: str, *, fetcher=None, retrieved_at: str | None = None) -> IsomerResolution:
    retrieved_at = retrieved_at or _now()
    q = query_name.strip()
    is_enantiomeric = looks_enantiomeric(q)
    props = fetch_pubchem_props(q, fetcher=fetcher)
    if not props:
        return IsomerResolution(
            query_name=q, pubchem_cid=None, canonical_smiles=None, isomeric_smiles=None,
            inchikey=None, inchikey_skeleton=None, inchikey_stereo_block=None,
            n_stereocenters_total=0, n_specified=0, n_unspecified=0,
            required_feature_schema="legacy_2d_v1", stereo_unspecified=True,
            enantiomeric_query=is_enantiomeric, enantiomer_query_collapsed=False,
            status="unresolved", source_url="https://pubchem.ncbi.nlm.nih.gov/",
            retrieved_at=retrieved_at, content_hash=content_hash(f"{q}|unresolved"),
        )
    iso = props.get("isomeric_smiles") or ""
    stereo = analyze_stereo(iso) if iso else analyze_stereo("")
    inchikey = props.get("inchikey")
    skel, stereo_block = inchikey_blocks(inchikey)
    unspecified = (stereo_block == _UNSPECIFIED_STEREO_BLOCK) or (
        stereo["n_specified"] == 0 and stereo["n_stereocenters_total"] > 0
    )
    collapsed = is_enantiomeric and unspecified
    if collapsed:
        status = "collapsed_needs_gap"
    elif unspecified and stereo["n_stereocenters_total"] > 0:
        status = "resolved_unspecified"
    else:
        status = "resolved"
    cid = props.get("cid")
    src = PUBCHEM_COMPOUND_URL.format(cid=cid) if cid else "https://pubchem.ncbi.nlm.nih.gov/"
    return IsomerResolution(
        query_name=q, pubchem_cid=cid,
        canonical_smiles=props.get("canonical_smiles"), isomeric_smiles=iso or None,
        inchikey=inchikey, inchikey_skeleton=skel, inchikey_stereo_block=stereo_block,
        n_stereocenters_total=stereo["n_stereocenters_total"],
        n_specified=stereo["n_specified"], n_unspecified=stereo["n_unspecified"],
        required_feature_schema=stereo["required_feature_schema"],
        stereo_unspecified=unspecified, enantiomeric_query=is_enantiomeric,
        enantiomer_query_collapsed=collapsed, status=status,
        source_url=src, retrieved_at=retrieved_at,
        content_hash=content_hash(f"{q}|{cid}|{inchikey}|{iso}"),
    )


def same_cid(a: IsomerResolution, b: IsomerResolution) -> bool:
    """Adversarial: two enantiomer queries returning the same CID = collapse."""
    return bool(a.pubchem_cid and b.pubchem_cid and a.pubchem_cid == b.pubchem_cid)
