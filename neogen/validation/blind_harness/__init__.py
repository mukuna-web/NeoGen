"""Leakage-resistant blind-evaluation harness.

First-class tooling to measure whether a "blind" evaluation cohort is actually
independent of the training / lookup / retrieval sources a model can reach at
inference time. Prior audits showed the existing ``blind_holdout_50`` cohort is
~100% lookup-leaked, yet leakage was only checkable via ad-hoc environment
flags. This package makes the leakage audit an explicit, reproducible artifact.

Design principles:
- **Fail closed.** If leakage cannot be ruled out, the cohort is reported unsafe.
- **No label tuning.** Nothing here reads or exposes sealed labels for tuning;
  it operates on identity (name / structure), never on the answer values.
- **Deterministic + hashed.** Inputs are content-hashed for replay.
"""

from .leakage_audit import (
    LeakageAuditor,
    LeakageResult,
    Record,
    canonical_smiles,
    inchikey_of,
    load_records,
)

__all__ = [
    "LeakageAuditor",
    "LeakageResult",
    "Record",
    "canonical_smiles",
    "inchikey_of",
    "load_records",
]
