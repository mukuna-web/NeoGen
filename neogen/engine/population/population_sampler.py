"""Monte Carlo population sampling for the V1 engine.

This module turns the reference-male engine into a *virtual trial*
simulator. Given a base simulation (drug + schedule + physiology) and
a :class:`PopulationSpec`, it samples N patients from published
distributions and returns a :class:`PopulationResult` with per-patient
PK envelopes.

Scope for V1
------------

The sampling model is intentionally narrow. We only vary the axes the
P5 patient adapter actually consumes, because sampling a dimension we
can't model is dishonest:

* **CYP2D6 / CYP2C19 / CYP2C9 / CYP3A4** phenotypes — from published
  population allele frequencies.
* **Age** — from a clipped normal centered on the spec's mean.
* **Weight** — from a log-normal scaled off BMI ~ 25.
* **Sex** — Bernoulli.
* **Smoker** — Bernoulli.
* **Hepatic / renal impairment** — optional fixed overlay (not
  randomized by default — impairment is usually a patient *selection*
  criterion, not a stochastic property).

Everything is seeded with :class:`numpy.random.Generator` so results
are reproducible.

Published allele frequencies used below
---------------------------------------

*Caucasian populations, rounded to the nearest percent.*

* CYP2D6: PM 7%, IM 18%, NM 68%, UM 7%
  (Zanger & Schwab 2013; Gaedigk 2019)
* CYP2C19: PM 3%, IM 25%, NM 40%, UM 32%
  (Scott 2013; CPIC 2017)
* CYP2C9: PM 2%, IM 30%, NM 68%
  (Van Booven 2010)
* CYP3A4 PM is effectively ~0% in the population, so we keep it fixed
  at ``NORMAL`` by default. Users can override via ``PopulationSpec``.
* Smoking prevalence ~20% (CDC 2022 adult average).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neogen.core.entities.drug import Drug
from neogen.core.entities.patient import (
    CYPPhenotype,
    GeneticProfile,
    OrganFunction,
    PatientProfile,
    Sex,
)
from neogen.core.entities.physiology import PhysiologyModel
from neogen.engine.dose_scheduler import DoseSchedule
from neogen.engine.orchestrator_v2 import OrchestratorV2, SimulationResultV2


# ════════════════════════════════════════════════════════════════
# PUBLISHED DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════


_DEFAULT_PHENOTYPE_FREQS: Mapping[str, Mapping[CYPPhenotype, float]] = {
    "CYP2D6": {
        CYPPhenotype.POOR: 0.07,
        CYPPhenotype.INTERMEDIATE: 0.18,
        CYPPhenotype.NORMAL: 0.68,
        CYPPhenotype.ULTRA_RAPID: 0.07,
    },
    "CYP2C19": {
        CYPPhenotype.POOR: 0.03,
        CYPPhenotype.INTERMEDIATE: 0.25,
        CYPPhenotype.NORMAL: 0.40,
        CYPPhenotype.ULTRA_RAPID: 0.32,
    },
    "CYP2C9": {
        CYPPhenotype.POOR: 0.02,
        CYPPhenotype.INTERMEDIATE: 0.30,
        CYPPhenotype.NORMAL: 0.68,
        CYPPhenotype.ULTRA_RAPID: 0.0,
    },
    # CYP1A2 / CYP2B6 / CYP3A4 kept at NORMAL by default — clinically
    # significant polymorphisms exist but V1 doesn't yet carry per-allele
    # activity multipliers distinct from the 4-class phenotype bucket.
}


# ════════════════════════════════════════════════════════════════
# POPULATION SPEC
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PopulationSpec:
    """Sampling rules for a virtual trial.

    Defaults describe a healthy Caucasian adult cohort (30 ± 8 years,
    70 kg typical, 50/50 sex, 20% smokers, no organ impairment).
    Override any field to narrow to a specific study population.
    """

    n_patients: int = 100
    mean_age_years: float = 35.0
    age_sd_years: float = 10.0
    min_age_years: float = 18.0
    max_age_years: float = 85.0

    mean_weight_kg: float = 70.0
    weight_cv: float = 0.15  # log-normal coefficient of variation

    fraction_female: float = 0.50
    fraction_smoker: float = 0.20

    # Phenotype frequencies — default published Caucasian table, but a
    # caller can swap in an East Asian or African table, or restrict to
    # a specific genotype (e.g. CYP2D6 PM only, for a pharmacogenomic
    # sub-study).
    phenotype_freqs: Mapping[str, Mapping[CYPPhenotype, float]] = field(
        default_factory=lambda: {
            cyp: dict(freqs) for cyp, freqs in _DEFAULT_PHENOTYPE_FREQS.items()
        }
    )

    # Organ impairment overlay — applied to every sampled patient.
    # ``None`` means "don't touch the default OrganFunction (normal)".
    # To run a Child-Pugh B population, pass
    # ``organ_function_overlay=OrganFunction(hepatic=0.5)``.
    organ_function_overlay: Optional[OrganFunction] = None

    seed: Optional[int] = 0


# ════════════════════════════════════════════════════════════════
# POPULATION RESULT
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PopulationMetric:
    """One summarized metric across the population."""

    name: str
    median: float
    p05: float
    p95: float
    mean: float
    stdev: float

    @property
    def range_fold(self) -> float:
        """P95 / P5 — the fold span across 90% of the population."""
        if self.p05 <= 0:
            return float("inf")
        return self.p95 / self.p05


@dataclass(frozen=True)
class PopulationResult:
    """Immutable result of a Monte Carlo population simulation."""

    spec: PopulationSpec
    patients: Tuple[PatientProfile, ...]
    results: Tuple[SimulationResultV2, ...]
    cmax: PopulationMetric
    auc: PopulationMetric

    @property
    def n(self) -> int:
        return len(self.results)

    def to_summary_lines(self) -> Tuple[str, ...]:
        """Human-readable summary block."""
        lines = [
            f"== Population simulation (n={self.n}) ==",
            f"  Cmax median={self.cmax.median:.4f}  "
            f"P5={self.cmax.p05:.4f}  P95={self.cmax.p95:.4f}  "
            f"fold={self.cmax.range_fold:.1f}×",
            f"  AUC  median={self.auc.median:.4f}  "
            f"P5={self.auc.p05:.4f}  P95={self.auc.p95:.4f}  "
            f"fold={self.auc.range_fold:.1f}×",
        ]
        return tuple(lines)


# ════════════════════════════════════════════════════════════════
# SAMPLER
# ════════════════════════════════════════════════════════════════


class PopulationSampler:
    """Draws :class:`PatientProfile` instances from a :class:`PopulationSpec`.

    Stateless — calling ``sample(spec)`` with the same ``seed`` gives
    identical patients every time. This is the exact knob needed by
    reproducible validation runs.
    """

    def sample(self, spec: PopulationSpec) -> Tuple[PatientProfile, ...]:
        rng = np.random.default_rng(spec.seed)

        n = spec.n_patients
        ages = self._sample_ages(rng, spec, n)
        weights = self._sample_weights(rng, spec, n)
        sexes = self._sample_sexes(rng, spec, n)
        smokers = self._sample_smokers(rng, spec, n)

        patients: List[PatientProfile] = []
        for i in range(n):
            genetics = self._sample_genetics(rng, spec.phenotype_freqs)
            organ = spec.organ_function_overlay or OrganFunction()
            patients.append(
                PatientProfile(
                    weight_kg=float(weights[i]),
                    height_cm=176.0 if sexes[i] == Sex.MALE else 163.0,
                    age_years=float(ages[i]),
                    sex=sexes[i],
                    organ_function=organ,
                    genetics=genetics,
                    is_smoker=bool(smokers[i]),
                )
            )
        return tuple(patients)

    # ── per-axis samplers ─────────────────────────────────────────

    @staticmethod
    def _sample_ages(
        rng: np.random.Generator,
        spec: PopulationSpec,
        n: int,
    ) -> np.ndarray:
        ages = rng.normal(spec.mean_age_years, spec.age_sd_years, size=n)
        return np.clip(ages, spec.min_age_years, spec.max_age_years)

    @staticmethod
    def _sample_weights(
        rng: np.random.Generator,
        spec: PopulationSpec,
        n: int,
    ) -> np.ndarray:
        # Log-normal on weight keeps tails physiological.
        sigma = float(np.sqrt(np.log(1.0 + spec.weight_cv ** 2)))
        mu = float(np.log(spec.mean_weight_kg) - 0.5 * sigma ** 2)
        weights = rng.lognormal(mean=mu, sigma=sigma, size=n)
        return np.clip(weights, 40.0, 200.0)

    @staticmethod
    def _sample_sexes(
        rng: np.random.Generator,
        spec: PopulationSpec,
        n: int,
    ) -> Tuple[Sex, ...]:
        draws = rng.random(size=n)
        return tuple(
            Sex.FEMALE if d < spec.fraction_female else Sex.MALE for d in draws
        )

    @staticmethod
    def _sample_smokers(
        rng: np.random.Generator,
        spec: PopulationSpec,
        n: int,
    ) -> np.ndarray:
        return rng.random(size=n) < spec.fraction_smoker

    @staticmethod
    def _sample_genetics(
        rng: np.random.Generator,
        phenotype_freqs: Mapping[str, Mapping[CYPPhenotype, float]],
    ) -> GeneticProfile:
        """Independent draws per CYP from the published frequency table."""
        draws: dict[str, CYPPhenotype] = {}
        for cyp, table in phenotype_freqs.items():
            phenos = list(table.keys())
            probs = np.array([table[p] for p in phenos], dtype=float)
            # Normalize defensively in case caller supplied rounded freqs.
            probs = probs / probs.sum()
            idx = int(rng.choice(len(phenos), p=probs))
            draws[cyp] = phenos[idx]

        return GeneticProfile(
            cyp2d6=draws.get("CYP2D6", CYPPhenotype.NORMAL),
            cyp2c19=draws.get("CYP2C19", CYPPhenotype.NORMAL),
            cyp2c9=draws.get("CYP2C9", CYPPhenotype.NORMAL),
            cyp2b6=draws.get("CYP2B6", CYPPhenotype.NORMAL),
            cyp1a2=draws.get("CYP1A2", CYPPhenotype.NORMAL),
            cyp3a4=draws.get("CYP3A4", CYPPhenotype.NORMAL),
        )


# ════════════════════════════════════════════════════════════════
# POPULATION RUNNER
# ════════════════════════════════════════════════════════════════


class PopulationRunner:
    """Runs :class:`OrchestratorV2` across a sampled population.

    Usage::

        engine = OrchestratorV2()
        runner = PopulationRunner(engine)
        pop = runner.run(
            drug, schedule, physiology,
            spec=PopulationSpec(n_patients=200, seed=42),
        )
        for line in pop.to_summary_lines():
            print(line)
    """

    def __init__(
        self,
        engine: Optional[OrchestratorV2] = None,
        sampler: Optional[PopulationSampler] = None,
    ) -> None:
        self._engine = engine or OrchestratorV2()
        self._sampler = sampler or PopulationSampler()

    def run(
        self,
        drug: Drug,
        schedule: DoseSchedule,
        physiology: PhysiologyModel,
        spec: PopulationSpec,
    ) -> PopulationResult:
        patients = self._sampler.sample(spec)

        results: List[SimulationResultV2] = []
        for patient in patients:
            result = self._engine.run(
                drug, schedule, physiology, patient=patient
            )
            results.append(result)

        cmax_values = [
            r.pk_result.pk_metrics.cmax_mg_per_L for r in results
        ]
        auc_values = [
            r.pk_result.pk_metrics.auc_0_inf_mg_h_per_L for r in results
        ]

        return PopulationResult(
            spec=spec,
            patients=patients,
            results=tuple(results),
            cmax=_summarize("Cmax", cmax_values),
            auc=_summarize("AUC", auc_values),
        )


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════


def _summarize(name: str, values: Sequence[float]) -> PopulationMetric:
    arr = np.asarray(values, dtype=float)
    return PopulationMetric(
        name=name,
        median=float(np.median(arr)),
        p05=float(np.percentile(arr, 5)),
        p95=float(np.percentile(arr, 95)),
        mean=float(np.mean(arr)),
        stdev=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    )
