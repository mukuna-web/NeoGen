"""Virtual-patient Monte Carlo sampler.

Produces a :class:`VirtualPopulation` of :class:`PatientProfile`
instances sampled from the priors in
:class:`PopulationDistributions`. The sampler is deterministic
when seeded — reproducibility is essential for validation runs.

Only the Python standard library :mod:`random` module is used.
Numpy is deliberately avoided so the sampler stays importable in
minimal environments and keeps the seed API identical across
processes.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Tuple

from neogen.core.entities.patient import (
    CYPPhenotype,
    GeneticProfile,
    OrganFunction,
    PatientProfile,
    Sex,
)
from neogen.population.distributions import (
    PopulationDistributions,
    categorical_phenotype,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# VIRTUAL POPULATION CONTAINER
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VirtualPopulation:
    """A sampled cohort of :class:`PatientProfile` instances.

    Treated as an immutable tuple under the hood so slicing and
    iteration are cheap. Attach ``seed`` and ``distributions``
    for reproducibility metadata — useful when a population
    result is persisted and replayed later.
    """

    patients: Tuple[PatientProfile, ...]
    seed: Optional[int] = None
    label: str = ""

    def __iter__(self) -> Iterator[PatientProfile]:
        return iter(self.patients)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx):  # type: ignore[override]
        return self.patients[idx]

    @property
    def n(self) -> int:
        return len(self.patients)

    def demographic_summary(self) -> dict:
        """Return coarse demographic counts for reporting."""
        n = max(len(self.patients), 1)
        males = sum(1 for p in self.patients if p.sex == Sex.MALE)
        smokers = sum(1 for p in self.patients if p.is_smoker)
        cyp2d6_pm = sum(
            1
            for p in self.patients
            if p.genetics.cyp2d6 == CYPPhenotype.POOR
        )
        cyp2d6_um = sum(
            1
            for p in self.patients
            if p.genetics.cyp2d6 == CYPPhenotype.ULTRA_RAPID
        )
        return {
            "n": n,
            "male_fraction": males / n,
            "smoker_fraction": smokers / n,
            "mean_weight_kg": sum(p.weight_kg for p in self.patients) / n,
            "mean_age_years": sum(p.age_years for p in self.patients) / n,
            "cyp2d6_pm_fraction": cyp2d6_pm / n,
            "cyp2d6_um_fraction": cyp2d6_um / n,
        }


# ════════════════════════════════════════════════════════════════
# SAMPLER
# ════════════════════════════════════════════════════════════════


class PopulationSampler:
    """Sample virtual patients from a :class:`PopulationDistributions`.

    Usage::

        sampler = PopulationSampler()
        pop = sampler.sample(n=100, seed=42)
        for patient in pop:
            ...

    Pass a custom :class:`PopulationDistributions` to the
    constructor to model subpopulations.
    """

    def __init__(
        self,
        distributions: Optional[PopulationDistributions] = None,
    ) -> None:
        self._dist = distributions or PopulationDistributions()

    # ── public API ──────────────────────────────────────────────

    def sample(
        self,
        n: int,
        seed: Optional[int] = None,
        *,
        label: str = "",
    ) -> VirtualPopulation:
        """Draw ``n`` patients. If ``seed`` is given, the result is
        deterministic across runs and processes.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        rng = random.Random(seed)
        patients = tuple(self._sample_one(rng) for _ in range(n))
        logger.info(
            "Sampled %d virtual patients (seed=%s, label=%r)",
            n, seed, label,
        )
        return VirtualPopulation(
            patients=patients, seed=seed, label=label
        )

    def sample_iter(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> Iterator[PatientProfile]:
        """Lazy generator variant — yields one patient at a time."""
        rng = random.Random(seed)
        for _ in range(n):
            yield self._sample_one(rng)

    # ── internal ────────────────────────────────────────────────

    def _sample_one(self, rng: random.Random) -> PatientProfile:
        """Build a single PatientProfile from the distributions."""
        sex = (
            Sex.MALE
            if rng.random() < self._dist.sex_male_probability
            else Sex.FEMALE
        )

        weight = self._sample_weight(rng)
        height = self._sample_height(rng, sex)
        age = self._sample_age(rng)
        organ_function = self._sample_organ_function(rng)
        genetics = self._sample_genetics(rng)
        smoker = rng.random() < self._dist.smoker_probability

        return PatientProfile(
            weight_kg=round(weight, 1),
            height_cm=round(height, 1),
            age_years=round(age, 1),
            sex=sex,
            organ_function=organ_function,
            genetics=genetics,
            is_smoker=smoker,
            is_fasted=True,  # V1: always fasted, food effects deferred
        )

    def _sample_weight(self, rng: random.Random) -> float:
        w = self._dist.weight
        val = rng.lognormvariate(w.mu_log, w.sigma_log)
        return max(35.0, min(val, 200.0))

    def _sample_height(
        self, rng: random.Random, sex: Sex
    ) -> float:
        h = self._dist.height
        mean = h.mean_cm + (
            h.male_offset_cm if sex == Sex.MALE else -h.male_offset_cm
        )
        val = rng.gauss(mean, h.sd_cm)
        return max(140.0, min(val, 210.0))

    def _sample_age(self, rng: random.Random) -> float:
        a = self._dist.age
        return rng.uniform(a.min_years, a.max_years)

    def _sample_organ_function(
        self, rng: random.Random
    ) -> OrganFunction:
        o = self._dist.organ

        def truncated_normal(
            mean: float, sd: float, lo: float, hi: float
        ) -> float:
            # Rejection sampling — simple and adequate for wide bands.
            for _ in range(16):
                v = rng.gauss(mean, sd)
                if lo <= v <= hi:
                    return v
            return max(lo, min(mean, hi))

        hepatic = truncated_normal(
            o.hepatic_mean, o.hepatic_sd, o.hepatic_min, o.hepatic_max
        )
        renal = truncated_normal(
            o.renal_mean, o.renal_sd, o.renal_min, o.renal_max
        )
        # Cardiac and pulmonary tied to 1.0 in V1 — lifestyle and
        # organ-function variability is minor compared to weight,
        # genotype, hepatic, and renal.
        return OrganFunction(
            hepatic=round(hepatic, 3),
            renal=round(renal, 3),
            cardiac=1.0,
            pulmonary=1.0,
        )

    def _sample_genetics(self, rng: random.Random) -> GeneticProfile:
        return GeneticProfile(
            cyp2d6=categorical_phenotype(
                rng, self._dist.cyp2d6_frequencies
            ),
            cyp2c19=categorical_phenotype(
                rng, self._dist.cyp2c19_frequencies
            ),
            cyp2c9=categorical_phenotype(
                rng, self._dist.cyp2c9_frequencies
            ),
            cyp3a4=categorical_phenotype(
                rng, self._dist.cyp3a4_frequencies
            ),
            cyp1a2=categorical_phenotype(
                rng, self._dist.cyp1a2_frequencies
            ),
            cyp2b6=categorical_phenotype(
                rng, self._dist.cyp2b6_frequencies
            ),
        )
