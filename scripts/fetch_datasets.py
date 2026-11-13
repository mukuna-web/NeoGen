#!/usr/bin/env python3
"""Fetch the third-party ADME datasets this repository does not redistribute.

See DATA.md. These datasets are pulled from Therapeutics Data Commons via the
``PyTDC`` package, so each arrives under its own upstream licence and it is the
user's responsibility to honour those terms.

Nothing in this repository requires them. The full test suite and both
benchmark suites run without them, because the models in ``models/`` are
already trained. They are needed only to retrain from scratch.

Usage
-----
    pip install PyTDC
    python3 scripts/fetch_datasets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "tdc"

# (TDC group, dataset name) — the ADME endpoints the training code expects.
DATASETS: list[tuple[str, str]] = [
    ("ADME", "Caco2_Wang"),
    ("ADME", "Bioavailability_Ma"),
    ("ADME", "HIA_Hou"),
    ("ADME", "Lipophilicity_AstraZeneca"),
    ("ADME", "Solubility_AqSolDB"),
    ("ADME", "PPBR_AZ"),
    ("ADME", "VDss_Lombardo"),
    ("ADME", "Half_Life_Obach"),
    ("ADME", "Clearance_Hepatocyte_AZ"),
    ("ADME", "Clearance_Microsome_AZ"),
]


def main() -> int:
    try:
        from tdc.single_pred import ADME
    except ImportError:
        print(
            "PyTDC is not installed. Install it first:\n\n    pip install PyTDC\n",
            file=sys.stderr,
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for group, name in DATASETS:
        target = OUT_DIR / f"{name}.json"
        if target.exists():
            print(f"[skip] {name} — already present")
            continue
        try:
            data = ADME(name=name, path=str(OUT_DIR))
            frame = data.get_data()
            frame.to_json(target, orient="records")
            print(f"[ok]   {name} — {len(frame)} rows")
        except Exception as exc:  # pragma: no cover - network/upstream dependent
            failures += 1
            print(f"[fail] {name} — {exc}", file=sys.stderr)

    if failures:
        print(
            f"\n{failures} dataset(s) failed. Check your network connection and "
            "that the upstream dataset names are still current.",
            file=sys.stderr,
        )
        return 1

    print(f"\nDone. Datasets are in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
