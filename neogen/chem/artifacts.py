"""Content-addressable filesystem storage for conformer artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem


DEFAULT_CONFORMER_ROOT = Path.home() / "NeoGen" / "data" / "conformers"


def cache_dir_for(
    stereoisomer_id: str,
    config_hash: str,
    root: Path | None = None,
) -> Path:
    return (root or DEFAULT_CONFORMER_ROOT) / stereoisomer_id / config_hash


def manifest_path(
    stereoisomer_id: str,
    config_hash: str,
    root: Path | None = None,
) -> Path:
    return cache_dir_for(stereoisomer_id, config_hash, root) / "manifest.json"


def has_manifest(
    stereoisomer_id: str,
    config_hash: str,
    root: Path | None = None,
) -> bool:
    return manifest_path(stereoisomer_id, config_hash, root).exists()


def read_manifest(
    stereoisomer_id: str,
    config_hash: str,
    root: Path | None = None,
) -> dict[str, Any]:
    path = manifest_path(stereoisomer_id, config_hash, root)
    return json.loads(path.read_text(encoding="utf-8"))


def write_conformer_artifacts(
    *,
    mol: Chem.Mol,
    manifest: dict[str, Any],
    stereoisomer_id: str,
    config_hash: str,
    root: Path | None = None,
) -> dict[str, Any]:
    directory = cache_dir_for(stereoisomer_id, config_hash, root)
    directory.mkdir(parents=True, exist_ok=True)
    enriched = dict(manifest)
    conformers = []
    for item in manifest["conformers"]:
        rank = int(item["rank"])
        path = directory / f"conformer_{rank}.sdf"
        block = Chem.MolToMolBlock(mol, confId=int(item["source_conf_id"]), includeStereo=True)
        path.write_text(block + "$$$$\n", encoding="utf-8")
        stored = dict(item)
        stored["coordinates_uri"] = str(path)
        stored["molblock_uri"] = str(path)
        stored.pop("source_conf_id", None)
        conformers.append(stored)
    enriched["conformers"] = conformers
    manifest_file = directory / "manifest.json"
    manifest_file.write_text(
        json.dumps(enriched, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return enriched


def cache_size_bytes(
    stereoisomer_id: str,
    config_hash: str,
    root: Path | None = None,
) -> int:
    directory = cache_dir_for(stereoisomer_id, config_hash, root)
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
