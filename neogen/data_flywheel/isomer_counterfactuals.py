"""Isomer counterfactual records (GAP-011 / isomer_counterfactuals table).

A counterfactual pairs two structures that share connectivity but differ in
stereochemistry (enantiomers / diastereomers / E-Z). It encodes the testable
expectation that a chirality-blind 2D feature schema collides them while a
stereo-aware schema separates them — the core of NeoGen's stereo moat.

Pure comparison logic over ``IsomerResolution`` objects. No clinical claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from neogen.data_flywheel.isomer_resolution import IsomerResolution
from neogen.data_flywheel.provenance import content_hash


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class IsomerCounterfactual:
    parent_name: str
    isomer_a: str
    isomer_b: str
    cid_a: str | None
    cid_b: str | None
    inchikey_a: str | None
    inchikey_b: str | None
    same_connectivity: bool
    different_stereochemistry: bool
    legacy_2d_expected_collision: bool
    stereo_2d_expected_separation: bool
    conformer_3d_recommended: bool
    note: str
    source_a: str
    source_b: str
    retrieved_at: str
    content_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_counterfactual(
    parent_name: str,
    a: IsomerResolution,
    b: IsomerResolution,
    *,
    retrieved_at: str | None = None,
) -> IsomerCounterfactual:
    retrieved_at = retrieved_at or _now()
    same_conn = bool(
        a.inchikey_skeleton and b.inchikey_skeleton
        and a.inchikey_skeleton == b.inchikey_skeleton
    )
    diff_stereo = bool(
        same_conn and a.inchikey and b.inchikey and a.inchikey != b.inchikey
    )
    # 2D-without-chirality collides whenever connectivity is identical.
    legacy_collision = same_conn
    # Stereo-aware features separate only if there is a real, specified stereo diff.
    stereo_separation = bool(
        diff_stereo and (a.n_specified > 0 or b.n_specified > 0)
    )
    conformer_reco = max(a.n_stereocenters_total, b.n_stereocenters_total) >= 3

    if same_conn and not diff_stereo:
        note = (
            "Same connectivity AND same InChIKey: at least one query collapsed to a "
            "stereo-unspecified parent — stereo separation NOT achievable yet (gap)."
        )
    elif diff_stereo and not stereo_separation:
        note = "Stereo layer differs but no specified centers — separation uncertain."
    elif stereo_separation:
        note = "Enantiomer/diastereomer pair: legacy_2d collides, stereo_2d separates."
    else:
        note = "Different connectivity — not a stereo counterfactual."

    return IsomerCounterfactual(
        parent_name=parent_name,
        isomer_a=a.query_name, isomer_b=b.query_name,
        cid_a=a.pubchem_cid, cid_b=b.pubchem_cid,
        inchikey_a=a.inchikey, inchikey_b=b.inchikey,
        same_connectivity=same_conn, different_stereochemistry=diff_stereo,
        legacy_2d_expected_collision=legacy_collision,
        stereo_2d_expected_separation=stereo_separation,
        conformer_3d_recommended=conformer_reco,
        note=note, source_a=a.source_url, source_b=b.source_url,
        retrieved_at=retrieved_at,
        content_hash=content_hash(
            f"{parent_name}|{a.inchikey}|{b.inchikey}|{same_conn}|{diff_stereo}"
        ),
    )
