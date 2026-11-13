"""Resource + tool availability check for the data-flywheel continuous loop.

Reports local compute (RDKit / torch / MPS), disk headroom, and (optionally)
VORTEXT API reachability so the loop can decide what is safe to run. No network
by default; no secrets logged.
"""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass, field

# Stop heavy/local-compute work if free disk drops below this (GB).
MIN_FREE_GB_FOR_COMPUTE = 3.0


@dataclass(frozen=True)
class ResourceStatus:
    free_gb: float
    rdkit: str | None
    torch: str | None
    mps_available: bool
    numpy: str | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def safe_for_local_compute(self) -> bool:
        return self.free_gb >= MIN_FREE_GB_FOR_COMPUTE

    @property
    def has_rdkit(self) -> bool:
        return self.rdkit is not None

    def to_dict(self) -> dict:
        return {
            "free_gb": round(self.free_gb, 2),
            "rdkit": self.rdkit,
            "torch": self.torch,
            "mps_available": self.mps_available,
            "numpy": self.numpy,
            "safe_for_local_compute": self.safe_for_local_compute,
            "notes": list(self.notes),
        }


def _ver(module: str) -> str | None:
    try:
        m = importlib.import_module(module)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return None


def _free_gb(path: str = "/") -> float:
    return shutil.disk_usage(path).free / (1024**3)


def check_resources() -> ResourceStatus:
    free = _free_gb()
    rdkit_v = _ver("rdkit")
    numpy_v = _ver("numpy")
    torch_v = _ver("torch")
    mps = False
    if torch_v:
        try:
            import torch

            mps = bool(torch.backends.mps.is_available())
        except Exception:
            mps = False
    notes: list[str] = []
    if free < MIN_FREE_GB_FOR_COMPUTE:
        notes.append(
            f"LOW DISK: {free:.1f} GB free (< {MIN_FREE_GB_FOR_COMPUTE} GB) — "
            "skip heavy/local-compute jobs."
        )
    if not rdkit_v:
        notes.append("RDKit not available — structure/stereo work limited.")
    return ResourceStatus(
        free_gb=free,
        rdkit=rdkit_v,
        torch=torch_v,
        mps_available=mps,
        numpy=numpy_v,
        notes=tuple(notes),
    )


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(check_resources().to_dict(), indent=2))
