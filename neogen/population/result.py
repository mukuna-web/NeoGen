"""Result types for population Monte Carlo runs.

Immutable containers for per-patient and cohort-level outputs.
All statistics are computed lazily from the underlying list of
per-patient PK metrics. JSON serialization is lightweight —
only scalar summaries, not full concentration profiles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

from neogen.core.entities.patient import PatientProfile
from neogen.engine.simulation_result import PKMetrics


# ════════════════════════════════════════════════════════════════
# PER-PATIENT RESULT
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PatientSimulationResult:
    """One virtual patient's simulation outcome."""

    patient: PatientProfile
    drug_name: str
    dose_mg: float
    route: str
    pk_metrics: Optional[PKMetrics] = None
    cl_total_L_per_h: Optional[float] = None
    solver_success: bool = True
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.error == "" and self.pk_metrics is not None


# ════════════════════════════════════════════════════════════════
# COHORT-LEVEL SUMMARY STATS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PercentileSummary:
    """Standard non-parametric summary for a scalar PK metric."""

    n: int
    median: float
    p05: float
    p25: float
    p75: float
    p95: float
    geomean: float
    geomean_cv_pct: float  # geometric CV in %
    min_value: float
    max_value: float

    @classmethod
    def from_values(
        cls, values: Sequence[float]
    ) -> "PercentileSummary":
        vs = [v for v in values if v is not None and math.isfinite(v) and v > 0]
        n = len(vs)
        if n == 0:
            return cls(
                n=0,
                median=float("nan"),
                p05=float("nan"),
                p25=float("nan"),
                p75=float("nan"),
                p95=float("nan"),
                geomean=float("nan"),
                geomean_cv_pct=float("nan"),
                min_value=float("nan"),
                max_value=float("nan"),
            )
        sorted_v = sorted(vs)

        def pct(p: float) -> float:
            # Linear interpolation percentile (type-7 quantile).
            if n == 1:
                return sorted_v[0]
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            if f == c:
                return sorted_v[f]
            return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])

        log_vals = [math.log(v) for v in vs]
        mu_log = sum(log_vals) / n
        var_log = (
            sum((lv - mu_log) ** 2 for lv in log_vals) / n if n > 1 else 0.0
        )
        geomean = math.exp(mu_log)
        geomean_cv = (
            math.sqrt(math.exp(var_log) - 1.0) * 100.0 if var_log > 0 else 0.0
        )

        return cls(
            n=n,
            median=pct(0.5),
            p05=pct(0.05),
            p25=pct(0.25),
            p75=pct(0.75),
            p95=pct(0.95),
            geomean=geomean,
            geomean_cv_pct=geomean_cv,
            min_value=min(vs),
            max_value=max(vs),
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "median": self.median,
            "p05": self.p05,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "geomean": self.geomean,
            "geomean_cv_pct": self.geomean_cv_pct,
            "min": self.min_value,
            "max": self.max_value,
        }


# ════════════════════════════════════════════════════════════════
# POPULATION RESULT
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PopulationResult:
    """Composite Monte Carlo population result.

    Stores every per-patient result (so users can filter and
    re-aggregate) plus pre-computed percentile summaries for the
    headline metrics (Cmax, AUC, t½, CL).
    """

    drug_name: str
    dose_mg: float
    route: str
    n_requested: int
    seed: Optional[int]
    patient_results: Tuple[PatientSimulationResult, ...] = field(
        default_factory=tuple
    )
    runtime_s: float = 0.0
    label: str = ""

    # ── derived summaries ───────────────────────────────────────

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.patient_results if r.succeeded)

    @property
    def n_errored(self) -> int:
        return sum(1 for r in self.patient_results if not r.succeeded)

    def metric_values(self, name: str) -> Tuple[float, ...]:
        """Return a tuple of a single PK metric across successful patients.

        ``name`` must match a :class:`PKMetrics` field, or one of the
        convenience aliases ``cl`` (cl_total_L_per_h), ``auc``, ``cmax``,
        ``t_half``.
        """
        aliases = {
            "cl": lambda r: r.cl_total_L_per_h,
            "auc": lambda r: (
                r.pk_metrics.auc_0_inf_mg_h_per_L
                if r.pk_metrics is not None
                else None
            ),
            "cmax": lambda r: (
                r.pk_metrics.cmax_mg_per_L
                if r.pk_metrics is not None
                else None
            ),
            "t_half": lambda r: (
                r.pk_metrics.half_life_h
                if r.pk_metrics is not None
                else None
            ),
            "tmax": lambda r: (
                r.pk_metrics.tmax_h
                if r.pk_metrics is not None
                else None
            ),
            "vss": lambda r: (
                r.pk_metrics.vss_apparent_L
                if r.pk_metrics is not None
                else None
            ),
        }
        if name in aliases:
            getter = aliases[name]
        else:
            def _field_getter(r: PatientSimulationResult, _name=name):
                if r.pk_metrics is None:
                    return None
                return getattr(r.pk_metrics, _name, None)
            getter = _field_getter

        out: list[float] = []
        for r in self.patient_results:
            if not r.succeeded:
                continue
            v = getter(r)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            out.append(fv)
        return tuple(out)

    def summary_for(self, name: str) -> PercentileSummary:
        """Return percentile summary for one metric."""
        return PercentileSummary.from_values(self.metric_values(name))

    def headline_summaries(self) -> dict[str, PercentileSummary]:
        """Pre-computed summaries for the 5 PK metrics used in every report."""
        return {
            "cmax_mg_per_L": self.summary_for("cmax"),
            "auc_0_inf_mg_h_per_L": self.summary_for("auc"),
            "half_life_h": self.summary_for("t_half"),
            "cl_total_L_per_h": self.summary_for("cl"),
            "vss_apparent_L": self.summary_for("vss"),
        }

    # ── serialization ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        summaries = {
            k: v.to_dict() for k, v in self.headline_summaries().items()
        }
        return {
            "drug_name": self.drug_name,
            "dose_mg": self.dose_mg,
            "route": self.route,
            "n_requested": self.n_requested,
            "n_completed": self.n_completed,
            "n_errored": self.n_errored,
            "seed": self.seed,
            "label": self.label,
            "runtime_s": round(self.runtime_s, 2),
            "summaries": summaries,
        }

    def to_markdown(self) -> str:
        """Short Markdown summary — no per-patient details."""
        summaries = self.headline_summaries()
        lines = [
            f"# Population Simulation — {self.drug_name} "
            f"{self.dose_mg:g} mg {self.route}",
            "",
            f"- N requested: {self.n_requested}",
            f"- N completed: {self.n_completed}",
            f"- N errored:   {self.n_errored}",
            f"- Seed:        {self.seed}",
            f"- Runtime:     {self.runtime_s:.1f} s",
            "",
            "| Metric | Median | p05 | p95 | Geomean CV |",
            "|---|---|---|---|---|",
        ]
        for label, metric_key in (
            ("Cmax (mg/L)", "cmax_mg_per_L"),
            ("AUC_0_inf (mg·h/L)", "auc_0_inf_mg_h_per_L"),
            ("t½ (h)", "half_life_h"),
            ("CL_total (L/h)", "cl_total_L_per_h"),
            ("Vss (L)", "vss_apparent_L"),
        ):
            s = summaries[metric_key]
            if s.n == 0:
                lines.append(f"| {label} | - | - | - | - |")
                continue
            lines.append(
                f"| {label} | {s.median:.3f} | {s.p05:.3f} | "
                f"{s.p95:.3f} | {s.geomean_cv_pct:.1f}% |"
            )
        lines.append("")
        return "\n".join(lines)
