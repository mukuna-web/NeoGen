"""Structure/isomerism cache pilot.

Resolves a drug name to PubChem CID + InChIKey + isomeric/canonical SMILES
(PUG-REST), then uses LOCAL RDKit to compute stereochemistry descriptors and
decide the required feature schema tier (legacy_2d / stereo_2d / conformer_3d).

This is the bridge from identity → structure-aware features. The PubChem fetch
is injectable for offline tests; the RDKit analysis is local CPU only (no GPU,
no large jobs). Each record is provenance-hashed and replayable.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

from neogen.data_flywheel.provenance import content_hash

# PubChem deprecated CanonicalSMILES/IsomericSMILES; the current property names
# are SMILES (absolute/isomeric) and ConnectivitySMILES (connectivity only).
# We request the new names and keep backward-compat for the old ones.
PUBCHEM_PROPS = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "InChIKey,SMILES,ConnectivitySMILES/JSON"
)
PUBCHEM_COMPOUND_URL = "https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"

Fetcher = Callable[[str], dict | None]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pubchem_props_url(name: str) -> str:
    return PUBCHEM_PROPS.format(name=urllib.parse.quote(name))


def _http_fetch(name: str, timeout: int = 25) -> dict | None:  # pragma: no cover - network
    req = urllib.request.Request(
        pubchem_props_url(name), headers={"User-Agent": "neogen-flywheel/0.1"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def fetch_pubchem_props(name: str, *, fetcher: Fetcher | None = None) -> dict | None:
    """Return {cid, inchikey, canonical_smiles, isomeric_smiles} or None."""
    data = (fetcher or _http_fetch)(name)
    if not data:
        return None
    props = (data.get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return None
    p = props[0]
    # New PubChem: "SMILES" = isomeric/absolute, "ConnectivitySMILES" = flat.
    # Legacy fallback: IsomericSMILES / CanonicalSMILES.
    isomeric = p.get("SMILES") or p.get("IsomericSMILES")
    connectivity = p.get("ConnectivitySMILES") or p.get("CanonicalSMILES")
    return {
        "cid": str(p.get("CID")) if p.get("CID") is not None else None,
        "inchikey": p.get("InChIKey"),
        "canonical_smiles": connectivity,
        "isomeric_smiles": isomeric or connectivity,
    }


def analyze_stereo(isomeric_smiles: str) -> dict:
    """Compute stereo descriptors from an isomeric SMILES using local RDKit.

    Returns counts + a stereochemistry_status + stereo_relevance + the required
    feature schema tier. Raises ImportError if RDKit is unavailable.
    """
    from rdkit import Chem  # local, CPU-only

    mol = Chem.MolFromSmiles(isomeric_smiles)
    if mol is None:
        return {
            "valid": False,
            "n_stereocenters_total": 0,
            "n_specified": 0,
            "n_unspecified": 0,
            "n_ez_specified": 0,
            "stereochemistry_status": "unknown",
            "stereo_relevance": "unknown",
            "required_feature_schema": "legacy_2d_v1",
            "requires_conformers": False,
        }
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    n_total = len(centers)
    n_specified = sum(1 for _, lab in centers if lab in ("R", "S"))
    n_unspecified = n_total - n_specified

    n_ez = 0
    for bond in mol.GetBonds():
        st = bond.GetStereo()
        if st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            n_ez += 1

    if n_total == 0 and n_ez == 0:
        status = "achiral"
        relevance = "none"
    elif n_unspecified > 0:
        status = "unspecified"
        relevance = "enantiomer_sensitive" if n_total > 0 else "ez_isomer_sensitive"
    elif n_specified > 0:
        status = "single_enantiomer"
        relevance = "enantiomer_sensitive"
    elif n_ez > 0:
        status = "ez_defined"
        relevance = "ez_isomer_sensitive"
    else:
        status = "unknown"
        relevance = "unknown"

    required_schema = "legacy_2d_v1" if relevance == "none" else "stereo_2d_v1"
    # Conservative: many stereocenters often means 3D shape matters; flag (not force).
    requires_conformers = n_total >= 3

    return {
        "valid": True,
        "n_stereocenters_total": n_total,
        "n_specified": n_specified,
        "n_unspecified": n_unspecified,
        "n_ez_specified": n_ez,
        "stereochemistry_status": status,
        "stereo_relevance": relevance,
        "required_feature_schema": required_schema,
        "requires_conformers": requires_conformers,
    }


@dataclass(frozen=True)
class StructureStereoRecord:
    drug: str
    rxnorm_rxcui: str | None
    pubchem_cid: str | None
    inchikey: str | None
    isomeric_smiles: str | None
    canonical_smiles: str | None
    n_stereocenters_total: int
    n_specified: int
    n_unspecified: int
    n_ez_specified: int
    stereochemistry_status: str
    stereo_relevance: str
    required_feature_schema: str
    requires_conformers: bool
    source_url: str
    retrieved_at: str
    content_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_structure_record(
    drug: str,
    props: dict,
    *,
    rxcui: str | None = None,
    retrieved_at: str | None = None,
) -> StructureStereoRecord:
    retrieved_at = retrieved_at or _now()
    iso = props.get("isomeric_smiles") or ""
    stereo = analyze_stereo(iso) if iso else analyze_stereo("")
    cid = props.get("cid")
    src = PUBCHEM_COMPOUND_URL.format(cid=cid) if cid else "https://pubchem.ncbi.nlm.nih.gov/"
    chash = content_hash(f"{drug}|{cid}|{props.get('inchikey')}|{iso}")
    return StructureStereoRecord(
        drug=drug,
        rxnorm_rxcui=rxcui,
        pubchem_cid=cid,
        inchikey=props.get("inchikey"),
        isomeric_smiles=iso or None,
        canonical_smiles=props.get("canonical_smiles"),
        n_stereocenters_total=stereo["n_stereocenters_total"],
        n_specified=stereo["n_specified"],
        n_unspecified=stereo["n_unspecified"],
        n_ez_specified=stereo["n_ez_specified"],
        stereochemistry_status=stereo["stereochemistry_status"],
        stereo_relevance=stereo["stereo_relevance"],
        required_feature_schema=stereo["required_feature_schema"],
        requires_conformers=stereo["requires_conformers"],
        source_url=src,
        retrieved_at=retrieved_at,
        content_hash=chash,
    )
