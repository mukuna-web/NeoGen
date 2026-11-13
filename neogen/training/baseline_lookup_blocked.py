"""Lookup-blocked Tier-1 evaluation for the Week 4 MVP gate."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from neogen.validation.cohort import Tier1Case, load_tier1_blind_holdout
from neogen.validation.tier1 import Tier1Harness, Tier1Report

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LookupBlockedReport:
    tier1_report: Tier1Report
    blocked_names: tuple[str, ...]
    evaluated_names: tuple[str, ...]

    @property
    def pct_passed(self) -> float:
        return self.tier1_report.pct_passed

    @property
    def gate_pass(self) -> bool:
        return self.pct_passed >= 56.0


def _normalise_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def load_lookup_names(repo_root: Path = REPO_ROOT) -> set[str]:
    names: set[str] = set()
    drugbank_path = repo_root / "data" / "drugbank" / "parsed" / "all_drugs.json"
    if drugbank_path.exists():
        for row in json.loads(drugbank_path.read_text()):
            if isinstance(row, dict) and row.get("name"):
                names.add(_normalise_name(str(row["name"])))
    opus_dir = repo_root / "data" / "opus_predictions"
    if opus_dir.exists():
        for path in sorted(opus_dir.glob("*.json")):
            try:
                rows = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        name = row.get("drug_name") or row.get("name")
                        if name:
                            names.add(_normalise_name(str(name)))
    return names


def filter_lookup_blocked_cases(
    cases: Iterable[Tier1Case],
    lookup_names: set[str] | None = None,
) -> tuple[tuple[Tier1Case, ...], tuple[str, ...]]:
    lookup_names = lookup_names if lookup_names is not None else load_lookup_names()
    kept: list[Tier1Case] = []
    blocked: list[str] = []
    for case in cases:
        if _normalise_name(case.name) in lookup_names:
            blocked.append(case.name)
        else:
            kept.append(case)
    return tuple(kept), tuple(sorted(blocked, key=str.lower))


@contextmanager
def _blocked_lookup_env():
    previous_block = os.environ.get("NEOGEN_BLOCK_HOLDOUT_LOOKUP")
    previous_blend = os.environ.get("NEOGEN_SIMILARITY_BLEND")
    os.environ["NEOGEN_BLOCK_HOLDOUT_LOOKUP"] = "1"
    os.environ["NEOGEN_SIMILARITY_BLEND"] = "force"
    try:
        yield
    finally:
        if previous_block is None:
            os.environ.pop("NEOGEN_BLOCK_HOLDOUT_LOOKUP", None)
        else:
            os.environ["NEOGEN_BLOCK_HOLDOUT_LOOKUP"] = previous_block
        if previous_blend is None:
            os.environ.pop("NEOGEN_SIMILARITY_BLEND", None)
        else:
            os.environ["NEOGEN_SIMILARITY_BLEND"] = previous_blend


def run_lookup_blocked_tier1(
    cases: tuple[Tier1Case, ...] | None = None,
    harness: Tier1Harness | None = None,
    lookup_names: set[str] | None = None,
) -> LookupBlockedReport:
    harness = harness or Tier1Harness()
    all_cases = cases if cases is not None else load_tier1_blind_holdout(literature_names=harness._literature_names)
    filtered, blocked = filter_lookup_blocked_cases(all_cases, lookup_names)
    with _blocked_lookup_env():
        report = harness.run(filtered)
    return LookupBlockedReport(
        tier1_report=report,
        blocked_names=blocked,
        evaluated_names=tuple(case.name for case in filtered),
    )
