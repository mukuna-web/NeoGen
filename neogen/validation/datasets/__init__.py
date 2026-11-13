"""Clinical reference datasets for V1 benchmarking.

Each module here exposes a ``build_*()`` factory returning a tuple of
:class:`BenchmarkCase` / :class:`DDIBenchmarkCase` instances. Datasets
are code, not YAML, so every row is type-checked at import time and
citations live next to the numbers.
"""
from neogen.validation.datasets.ddi_benchmark_v1 import build_ddi_benchmark_v1
from neogen.validation.datasets.pk_benchmark_v1 import build_pk_benchmark_v1

__all__ = ["build_ddi_benchmark_v1", "build_pk_benchmark_v1"]
