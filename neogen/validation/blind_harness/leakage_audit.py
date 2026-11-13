"""Blind-cohort leakage auditor.

Given a *blind* cohort (records to be evaluated as if novel) and one or more
*reference* sources (training tables, known-drug lookups, gold-standard tables,
retrieval databases), determine whether each blind record is actually reachable
through a reference source by identity or near-identity.

Checks (a record is flagged leaked if ANY fires -- fail-closed):
  - exact normalized name overlap
  - canonical (non-isomeric) SMILES overlap
  - isomeric SMILES overlap
  - full InChIKey overlap
  - parent / skeleton block overlap (InChIKey first 14 chars)
  - near-duplicate structure (max Morgan-Tanimoto >= threshold)

This module reads only *identity* (name + structure). It never reads, exposes,
or tunes against the sealed answer values of a blind cohort. RDKit is required
for structural checks; if unavailable, structural checks are skipped and the run
is marked degraded (still fail-closed on name overlap).

CLI:
    python -m neogen.validation.blind_harness.leakage_audit \\
        --cohort data/validation/blind_holdout_50.json \\
        --references data/drugbank/parsed/expanded_known_drugs.json \\
                     data/clinical/gold_standard_pk.json \\
        --out leakage_audit.json [--fp-threshold 0.95]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:  # RDKit is pinned in requirements; degrade gracefully if absent.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    _RDKIT = True
except Exception:  # pragma: no cover - exercised only in RDKit-less envs
    _RDKIT = False

# Common field names across the repo's data files.
_NAME_FIELDS = ("name", "drug_name", "canonical_name", "drug")
_SMILES_FIELDS = ("smiles", "isomeric_smiles", "canonical_smiles", "SMILES")

DEFAULT_FP_THRESHOLD = 0.95
_FP_RADIUS = 2
_FP_BITS = 2048
_INCHIKEY_BLOCK = 14  # skeleton/connectivity block (parent identity)


def _norm_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(str(name).strip().lower().split()) or None


def canonical_smiles(smiles: Optional[str], isomeric: bool = False) -> Optional[str]:
    """RDKit canonical SMILES, or None if RDKit is absent / SMILES invalid."""
    if not _RDKIT or not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def inchikey_of(smiles: Optional[str]) -> Optional[str]:
    if not _RDKIT or not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol) or None
    except Exception:  # pragma: no cover - malformed molecule
        return None


def _fingerprint(smiles: Optional[str]):
    if not _RDKIT or not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, _FP_RADIUS, nBits=_FP_BITS)


def _content_hash(path: Path) -> str:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:32]
    except Exception:
        return "sha256:unavailable"


@dataclass(frozen=True)
class Record:
    """A minimal identity record (name + structure). No answer values."""

    name: Optional[str]
    smiles: Optional[str]
    source: str = "?"


def _pick(row: dict, fields: tuple[str, ...]) -> Optional[str]:
    for f in fields:
        if f in row and row[f] not in (None, "", "None"):
            return row[f]
    return None


def load_records(path: str | Path, source: Optional[str] = None) -> list[Record]:
    """Load identity records from a repo JSON file (list-of-dicts or a dict with
    a ``pairs``/``drugs``/``records`` list). Robust to the repo's varied schemas."""
    p = Path(path)
    data = json.loads(p.read_text())
    src = source or p.name
    rows: list[dict] = []
    if isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        for key in ("records", "drugs", "pairs", "cohort", "data"):
            if isinstance(data.get(key), list):
                rows = [r for r in data[key] if isinstance(r, dict)]
                break
    out: list[Record] = []
    for row in rows:
        name = _pick(row, _NAME_FIELDS)
        smiles = _pick(row, _SMILES_FIELDS)
        if name is None and smiles is None:
            continue
        out.append(Record(name=_norm_name(name), smiles=smiles, source=src))
    return out


@dataclass
class LeakageResult:
    blind_source: str
    reference_sources: list[str]
    fp_threshold: float
    rdkit_available: bool
    n_blind: int
    n_leaked: int
    leak_rate: float
    check_counts: dict[str, int]
    per_record: list[dict]
    input_hashes: dict[str, str]
    verdict: str  # BLIND_INTEGRITY_PASS | BLIND_INTEGRITY_FAIL | DEGRADED
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


class LeakageAuditor:
    """Builds identity indices over reference sources and audits a blind cohort."""

    def __init__(self, references: list[Record], fp_threshold: float = DEFAULT_FP_THRESHOLD):
        self.fp_threshold = fp_threshold
        self._names: set[str] = set()
        self._canon: set[str] = set()
        self._iso: set[str] = set()
        self._ikey: set[str] = set()
        self._parent: set[str] = set()
        self._fps: list = []
        for r in references:
            if r.name:
                self._names.add(r.name)
            c = canonical_smiles(r.smiles, isomeric=False)
            if c:
                self._canon.add(c)
            i = canonical_smiles(r.smiles, isomeric=True)
            if i:
                self._iso.add(i)
            k = inchikey_of(r.smiles)
            if k:
                self._ikey.add(k)
                self._parent.add(k[:_INCHIKEY_BLOCK])
            fp = _fingerprint(r.smiles)
            if fp is not None:
                self._fps.append(fp)

    def _max_tanimoto(self, smiles: Optional[str]) -> float:
        fp = _fingerprint(smiles)
        if fp is None or not self._fps:
            return 0.0
        return max(DataStructs.TanimotoSimilarity(fp, ref) for ref in self._fps)

    def audit_record(self, rec: Record) -> dict:
        canon = canonical_smiles(rec.smiles, isomeric=False)
        iso = canonical_smiles(rec.smiles, isomeric=True)
        ikey = inchikey_of(rec.smiles)
        near = self._max_tanimoto(rec.smiles) if _RDKIT else 0.0
        flags = {
            "name_overlap": bool(rec.name and rec.name in self._names),
            "canonical_smiles_overlap": bool(canon and canon in self._canon),
            "isomeric_smiles_overlap": bool(iso and iso in self._iso),
            "inchikey_overlap": bool(ikey and ikey in self._ikey),
            "parent_block_overlap": bool(ikey and ikey[:_INCHIKEY_BLOCK] in self._parent),
            "near_duplicate": bool(near >= self.fp_threshold),
        }
        return {
            "name": rec.name,
            "leaked": any(flags.values()),
            "max_tanimoto": round(near, 4),
            "flags": flags,
        }

    def audit(
        self,
        blind: list[Record],
        blind_source: str,
        reference_sources: list[str],
        input_hashes: dict[str, str],
    ) -> LeakageResult:
        per_record = [self.audit_record(r) for r in blind]
        n_leaked = sum(1 for r in per_record if r["leaked"])
        check_counts: dict[str, int] = {}
        for r in per_record:
            for k, v in r["flags"].items():
                check_counts[k] = check_counts.get(k, 0) + (1 if v else 0)
        n = len(per_record)
        rate = (n_leaked / n) if n else 0.0
        notes: list[str] = []
        if not _RDKIT:
            notes.append("RDKit unavailable: structural checks skipped (name-only, degraded).")
        # Fail-closed verdict.
        if not _RDKIT:
            verdict = "DEGRADED"
        elif n_leaked == 0:
            verdict = "BLIND_INTEGRITY_PASS"
        else:
            verdict = "BLIND_INTEGRITY_FAIL"
        return LeakageResult(
            blind_source=blind_source,
            reference_sources=reference_sources,
            fp_threshold=self.fp_threshold,
            rdkit_available=_RDKIT,
            n_blind=n,
            n_leaked=n_leaked,
            leak_rate=round(rate, 4),
            check_counts=check_counts,
            per_record=per_record,
            input_hashes=input_hashes,
            verdict=verdict,
            notes=notes,
        )


def run_audit(
    cohort_path: str | Path,
    reference_paths: list[str | Path],
    fp_threshold: float = DEFAULT_FP_THRESHOLD,
) -> LeakageResult:
    blind = load_records(cohort_path)
    refs: list[Record] = []
    for rp in reference_paths:
        refs.extend(load_records(rp))
    hashes = {str(cohort_path): _content_hash(Path(cohort_path))}
    for rp in reference_paths:
        hashes[str(rp)] = _content_hash(Path(rp))
    auditor = LeakageAuditor(refs, fp_threshold=fp_threshold)
    return auditor.audit(
        blind,
        blind_source=str(cohort_path),
        reference_sources=[str(p) for p in reference_paths],
        input_hashes=hashes,
    )


def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Blind-cohort leakage auditor (fail-closed).")
    ap.add_argument("--cohort", required=True, help="Blind cohort JSON (identity only used).")
    ap.add_argument("--references", nargs="+", required=True, help="Reference source JSON files.")
    ap.add_argument("--out", default=None, help="Write leakage_audit.json here.")
    ap.add_argument("--fp-threshold", type=float, default=DEFAULT_FP_THRESHOLD)
    args = ap.parse_args(argv)
    result = run_audit(args.cohort, args.references, fp_threshold=args.fp_threshold)
    payload = result.to_json()
    if args.out:
        Path(args.out).write_text(payload)
    # Concise summary to stdout (safe: no answer values).
    print(
        f"verdict={result.verdict} n_blind={result.n_blind} "
        f"n_leaked={result.n_leaked} leak_rate={result.leak_rate} "
        f"rdkit={result.rdkit_available}"
    )
    print("check_counts=" + json.dumps(result.check_counts, sort_keys=True))
    # Exit non-zero on integrity failure so CI / loops can gate on it.
    return 0 if result.verdict == "BLIND_INTEGRITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
