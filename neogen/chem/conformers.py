"""Deterministic ETKDGv3 conformer generation and cache orchestration."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolAlign

from neogen.chem import artifacts
from neogen.chem.geometry import ensemble_geometry_features
from neogen.chem.identity import Stereoisomer
from neogen.events.conformer_events import conformer_event
from neogen.events.molecule_events import EventRef
from neogen.events.store import append_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformerConfig:
    seed: int = 42
    method: str = "etkdgv3"
    forcefield: str = "mmff94s"
    max_conformers: int = 10
    prune_rmsd: float = 0.5
    energy_window_kcal_mol: float = 10.0
    num_candidates: int = 50

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None = None) -> "ConformerConfig":
        data = dict(value or {})
        if "energy_window" in data and "energy_window_kcal_mol" not in data:
            data["energy_window_kcal_mol"] = data.pop("energy_window")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: data[key] for key in data if key in allowed})

    def stable_dict(self) -> dict[str, Any]:
        return {
            "energy_window_kcal_mol": float(self.energy_window_kcal_mol),
            "forcefield": self.forcefield,
            "max_conformers": int(self.max_conformers),
            "method": self.method,
            "num_candidates": int(self.num_candidates),
            "prune_rmsd": float(self.prune_rmsd),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class ConformerRecord:
    conformer_id: str
    conformer_set_id: str
    rank: int
    energy_kcal_mol: float
    rmsd_cluster_id: int
    coordinates_uri: str
    molblock_uri: str
    coordinate_hash: str


@dataclass(frozen=True)
class ConformerSet:
    conformer_set_id: str
    stereoisomer_id: str
    method: str
    seed: int
    rdkit_version: str
    forcefield: str
    forcefield_version: str
    max_conformers: int
    prune_rmsd: float
    energy_window_kcal_mol: float
    config_hash: str
    conformers: tuple[ConformerRecord, ...]
    manifest_uri: str
    cache_hit: bool = False
    warnings: tuple[str, ...] = ()


class ConformerGenerationError(RuntimeError):
    """Raised when RDKit cannot generate a conformer set."""


def config_hash(config: ConformerConfig) -> str:
    encoded = json.dumps(config.stable_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def conformer_set_id_for(stereoisomer_id: str, config_hash_value: str) -> str:
    return hashlib.sha256(f"CONFSET|{stereoisomer_id}|{config_hash_value}".encode("utf-8")).hexdigest()[:32]


def conformer_id_for(conformer_set_id: str, rank: int, coordinate_hash: str) -> str:
    return hashlib.sha256(f"CONF|{conformer_set_id}|{rank}|{coordinate_hash}".encode("utf-8")).hexdigest()[:32]


def coordinate_hash(mol: Chem.Mol, conf_id: int) -> str:
    conf = mol.GetConformer(conf_id)
    atom_symbols = ",".join(atom.GetSymbol() for atom in mol.GetAtoms())
    atom_indices = ",".join(str(idx) for idx in sorted(range(mol.GetNumAtoms())))
    coords: list[str] = []
    for idx in sorted(range(mol.GetNumAtoms())):
        point = conf.GetAtomPosition(idx)
        coords.extend([f"{point.x:.4f}", f"{point.y:.4f}", f"{point.z:.4f}"])
    payload = "|".join([atom_symbols, atom_indices, ",".join(coords)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _energy(mol: Chem.Mol, conf_id: int, forcefield: str) -> tuple[float, str, list[str]]:
    warnings: list[str] = []
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if forcefield in {"mmff94s", "mmff94"} and props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is not None:
            return float(ff.CalcEnergy()), "mmff94s", warnings
    warnings.append(f"MMFF94s unavailable for confId={conf_id}; used UFF fallback")
    logger.warning(warnings[-1])
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    if ff is None:
        raise ConformerGenerationError("UFF forcefield unavailable")
    return float(ff.CalcEnergy()), "uff", warnings


def _optimize(mol: Chem.Mol, conf_id: int, forcefield: str) -> tuple[float, str, list[str]]:
    warnings: list[str] = []
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if forcefield in {"mmff94s", "mmff94"} and props is not None:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=500, mmffVariant="MMFF94s")
            energy, used, more = _energy(mol, conf_id, forcefield)
            return energy, used, warnings + more
    except Exception as exc:
        warnings.append(f"MMFF94s failed for confId={conf_id}: {exc}; used UFF fallback")
        logger.warning(warnings[-1])
    if forcefield != "none":
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
        energy, used, more = _energy(mol, conf_id, "uff")
        return energy, used, warnings + more
    return 0.0, "none", warnings


def _post_rmsd_prune(mol: Chem.Mol, ranked: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    for item in ranked:
        if all(
            rdMolAlign.GetBestRMS(mol, mol, prbId=int(item["source_conf_id"]), refId=int(other["source_conf_id"]))
            >= threshold
            for other in kept
        ):
            kept.append(item)
    return kept


def _manifest_to_conformer_set(manifest: dict[str, Any], *, cache_hit: bool) -> ConformerSet:
    conformers = tuple(ConformerRecord(**item) for item in manifest["conformers"])
    return ConformerSet(
        conformer_set_id=manifest["conformer_set_id"],
        stereoisomer_id=manifest["stereoisomer_id"],
        method=manifest["method"],
        seed=int(manifest["seed"]),
        rdkit_version=manifest["rdkit_version"],
        forcefield=manifest["forcefield"],
        forcefield_version=manifest["forcefield_version"],
        max_conformers=int(manifest["max_conformers"]),
        prune_rmsd=float(manifest["prune_rmsd"]),
        energy_window_kcal_mol=float(manifest["energy_window_kcal_mol"]),
        config_hash=manifest["config_hash"],
        conformers=conformers,
        manifest_uri=manifest["manifest_uri"],
        cache_hit=cache_hit,
        warnings=tuple(manifest.get("warnings", [])),
    )


def generate_conformer_set(
    isomer: Stereoisomer,
    config: ConformerConfig | dict[str, Any] | None = None,
    *,
    cache_root: Path | None = None,
    emit_events: bool = True,
) -> ConformerSet:
    cfg = config if isinstance(config, ConformerConfig) else ConformerConfig.from_mapping(config)
    cfg_hash = config_hash(cfg)
    set_id = conformer_set_id_for(isomer.stereoisomer_id, cfg_hash)
    start = time.perf_counter()
    requested = None
    if emit_events:
        requested = append_event(
            conformer_event(
                event_type="ConformerGenerationRequested",
                sequence_number=1,
                output_payload={"stereoisomer_id": isomer.stereoisomer_id, "config": cfg.stable_dict()},
                input_refs=[EventRef(type="stereoisomer", id=isomer.stereoisomer_id)],
                config_hash_value=cfg_hash,
                seed=cfg.seed,
            )
        )
    if artifacts.has_manifest(isomer.stereoisomer_id, cfg_hash, cache_root):
        manifest = artifacts.read_manifest(isomer.stereoisomer_id, cfg_hash, cache_root)
        return _manifest_to_conformer_set(manifest, cache_hit=True)
    try:
        mol = Chem.MolFromSmiles(isomer.isomeric_smiles)
        if mol is None:
            raise ConformerGenerationError(f"invalid isomeric_smiles for {isomer.stereoisomer_id}")
        mol = Chem.AddHs(mol, addCoords=False)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(cfg.seed)
        params.pruneRmsThresh = float(cfg.prune_rmsd)
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(cfg.num_candidates), params=params))
        if not cids:
            raise ConformerGenerationError("RDKit returned zero embedded conformers")
        raw: list[dict[str, Any]] = []
        warnings: list[str] = []
        forcefields: set[str] = set()
        for cid in cids:
            energy, used_ff, ff_warnings = _optimize(mol, int(cid), cfg.forcefield)
            coord_hash = coordinate_hash(mol, int(cid))
            warnings.extend(ff_warnings)
            forcefields.add(used_ff)
            raw.append({"source_conf_id": int(cid), "energy_kcal_mol": energy, "coordinate_hash": coord_hash})
        min_energy = min(float(item["energy_kcal_mol"]) for item in raw)
        filtered = [
            item for item in raw
            if float(item["energy_kcal_mol"]) <= min_energy + float(cfg.energy_window_kcal_mol)
        ]
        ranked = sorted(filtered, key=lambda item: (float(item["energy_kcal_mol"]), item["coordinate_hash"]))
        kept = _post_rmsd_prune(mol, ranked, float(cfg.prune_rmsd))[: int(cfg.max_conformers)]
        used_forcefield = "uff" if forcefields == {"uff"} else "mmff94s"
        records = []
        for rank, item in enumerate(kept):
            records.append(
                {
                    "conformer_id": conformer_id_for(set_id, rank, item["coordinate_hash"]),
                    "conformer_set_id": set_id,
                    "rank": rank,
                    "energy_kcal_mol": round(float(item["energy_kcal_mol"]), 8),
                    "rmsd_cluster_id": rank,
                    "coordinates_uri": "",
                    "molblock_uri": "",
                    "coordinate_hash": item["coordinate_hash"],
                    "source_conf_id": int(item["source_conf_id"]),
                }
            )
        manifest_uri = str(artifacts.manifest_path(isomer.stereoisomer_id, cfg_hash, cache_root))
        manifest = {
            "conformer_set_id": set_id,
            "stereoisomer_id": isomer.stereoisomer_id,
            "isomeric_smiles": isomer.isomeric_smiles,
            "method": cfg.method,
            "seed": cfg.seed,
            "rdkit_version": rdBase.rdkitVersion,
            "forcefield": used_forcefield,
            "forcefield_version": f"{used_forcefield}_rdkit_{rdBase.rdkitVersion}",
            "max_conformers": cfg.max_conformers,
            "prune_rmsd": cfg.prune_rmsd,
            "energy_window_kcal_mol": cfg.energy_window_kcal_mol,
            "config_hash": cfg_hash,
            "manifest_uri": manifest_uri,
            "warnings": sorted(set(warnings)),
            "conformers": records,
        }
        stored = artifacts.write_conformer_artifacts(
            mol=mol,
            manifest=manifest,
            stereoisomer_id=isomer.stereoisomer_id,
            config_hash=cfg_hash,
            root=cache_root,
        )
        result = _manifest_to_conformer_set(stored, cache_hit=False)
        if emit_events:
            generated = append_event(
                conformer_event(
                    event_type="ConformerSetGenerated",
                    sequence_number=2,
                    parent_event_id=getattr(requested, "event_id", None),
                    input_refs=[EventRef(type="stereoisomer", id=isomer.stereoisomer_id)],
                    output_refs=[EventRef(type="conformer_set", id=set_id)],
                    output_payload=stored,
                    config_hash_value=cfg_hash,
                    seed=cfg.seed,
                    warnings=result.warnings,
                    duration_ms=int((time.perf_counter() - start) * 1000),
                )
            )
            append_event(
                conformer_event(
                    event_type="ConformerSetStored",
                    sequence_number=3,
                    parent_event_id=generated.event_id,
                    input_refs=[EventRef(type="conformer_set", id=set_id)],
                    output_refs=[EventRef(type="conformer_set", id=set_id)],
                    output_payload={"manifest_uri": result.manifest_uri, "conformer_set_id": set_id},
                    config_hash_value=cfg_hash,
                    seed=cfg.seed,
                )
            )
        return result
    except Exception as exc:
        if emit_events:
            append_event(
                conformer_event(
                    event_type="ConformerGenerationFailed",
                    sequence_number=2,
                    parent_event_id=getattr(requested, "event_id", None),
                    input_refs=[EventRef(type="stereoisomer", id=isomer.stereoisomer_id)],
                    output_payload={"stereoisomer_id": isomer.stereoisomer_id, "reason": str(exc)},
                    config_hash_value=cfg_hash,
                    seed=cfg.seed,
                    status="failed",
                    warnings=[str(exc)],
                    duration_ms=int((time.perf_counter() - start) * 1000),
                )
            )
        raise


def load_conformer_mol(conformer: ConformerRecord) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(conformer.coordinates_uri, removeHs=False)
    mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        raise FileNotFoundError(f"could not load conformer SDF: {conformer.coordinates_uri}")
    return mol


def conformer_set_to_dict(conformer_set: ConformerSet) -> dict[str, Any]:
    data = asdict(conformer_set)
    data["conformers"] = [asdict(item) for item in conformer_set.conformers]
    data["warnings"] = list(conformer_set.warnings)
    return data
