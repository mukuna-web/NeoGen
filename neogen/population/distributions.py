"""Population distribution priors for Monte Carlo virtual-patient sampling.

Default distributions match the US/European CPIC frequency tables
for pharmacogenomic phenotypes and NHANES-based anthropometric
statistics. Users who need African, East Asian, or custom
populations build their own :class:`PopulationDistributions`.

References:
  - CPIC CYP2D6 consensus frequencies (Caudle 2020)
  - CPIC CYP2C19 consensus frequencies (Lima 2021)
  - NHANES III adult body weight distribution
  - Brown et al. (1997) physiological reference values
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

from neogen.core.entities.patient import CYPPhenotype


# ════════════════════════════════════════════════════════════════
# CYP PHENOTYPE FREQUENCIES (US / European reference population)
# ════════════════════════════════════════════════════════════════
#
# Values are proportions (sum ≈ 1.0). CPIC recommends these as the
# "typical" frequencies when the population-specific value is not
# known. Subpopulation-specific overrides can be supplied via the
# PopulationDistributions constructor.


CYP2D6_FREQUENCIES_US: Mapping[CYPPhenotype, float] = {
    CYPPhenotype.POOR: 0.06,
    CYPPhenotype.INTERMEDIATE: 0.09,
    CYPPhenotype.NORMAL: 0.82,
    CYPPhenotype.ULTRA_RAPID: 0.03,
}


CYP2C19_FREQUENCIES_US: Mapping[CYPPhenotype, float] = {
    CYPPhenotype.POOR: 0.03,
    CYPPhenotype.INTERMEDIATE: 0.26,
    CYPPhenotype.NORMAL: 0.66,
    CYPPhenotype.ULTRA_RAPID: 0.05,
}


CYP2C9_FREQUENCIES_US: Mapping[CYPPhenotype, float] = {
    CYPPhenotype.POOR: 0.03,
    CYPPhenotype.INTERMEDIATE: 0.35,
    CYPPhenotype.NORMAL: 0.62,
    CYPPhenotype.ULTRA_RAPID: 0.0,
}


# CYP3A4, CYP1A2, CYP2B6 show weaker phenotype stratification in
# reference populations; V1 treats them as Normal by default and
# lets downstream users override when clinically relevant.
CYP_DEFAULT_NORMAL: Mapping[CYPPhenotype, float] = {
    CYPPhenotype.POOR: 0.0,
    CYPPhenotype.INTERMEDIATE: 0.0,
    CYPPhenotype.NORMAL: 1.0,
    CYPPhenotype.ULTRA_RAPID: 0.0,
}


# ════════════════════════════════════════════════════════════════
# ANTHROPOMETRIC DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WeightDistribution:
    """Log-normal body-weight prior.

    Log-normal is chosen because body weights in adult populations
    are right-skewed (a few very heavy individuals drag the mean
    above the median). A CV of 15% matches NHANES III adult data.
    """

    mean_kg: float = 70.0
    cv: float = 0.15

    @property
    def sigma_log(self) -> float:
        # For log-normal: sigma_log ≈ sqrt(log(1 + CV^2))
        from math import log, sqrt

        return sqrt(log(1.0 + self.cv ** 2))

    @property
    def mu_log(self) -> float:
        from math import log

        return log(self.mean_kg) - 0.5 * self.sigma_log ** 2


@dataclass(frozen=True)
class HeightDistribution:
    """Normal body-height prior. Height is approximately Gaussian."""

    mean_cm: float = 170.0
    sd_cm: float = 9.0
    male_offset_cm: float = 6.0  # add to mean for males, subtract for females


@dataclass(frozen=True)
class AgeDistribution:
    """Uniform age prior within a specified range.

    Defaults to 20-60 y — the adult validation range the V1 blueprint
    explicitly scopes. Pediatric and elderly populations require
    additional allometry (P5.1+).
    """

    min_years: float = 20.0
    max_years: float = 60.0


@dataclass(frozen=True)
class OrganFunctionDistribution:
    """Truncated-normal organ function priors.

    Most virtual patients are organ-normal; a small minority have
    mild hepatic or renal impairment. The defaults sample function
    scalars from N(0.97, 0.10) truncated to [0.3, 1.0] because the
    underlying :class:`OrganFunction` entity validates each scalar
    to the closed interval [0, 1] — "function" is a fraction of
    normal, so values > 1 are disallowed by design.
    """

    hepatic_mean: float = 0.97
    hepatic_sd: float = 0.10
    hepatic_min: float = 0.3
    hepatic_max: float = 1.0

    renal_mean: float = 0.97
    renal_sd: float = 0.10
    renal_min: float = 0.3
    renal_max: float = 1.0


# ════════════════════════════════════════════════════════════════
# TOP-LEVEL DISTRIBUTION BUNDLE
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PopulationDistributions:
    """Immutable bundle of all priors used by :class:`PopulationSampler`.

    Default values approximate a US adult reference population. Build
    a custom instance to model other populations:

        dist = PopulationDistributions(
            age=AgeDistribution(min_years=65, max_years=85),
            weight=WeightDistribution(mean_kg=75, cv=0.20),
            cyp2d6_frequencies=CUSTOM_EAST_ASIAN_2D6,
        )
    """

    # Anthropometry
    weight: WeightDistribution = field(default_factory=WeightDistribution)
    height: HeightDistribution = field(default_factory=HeightDistribution)
    age: AgeDistribution = field(default_factory=AgeDistribution)
    sex_male_probability: float = 0.5

    # Pharmacogenomics
    cyp2d6_frequencies: Mapping[CYPPhenotype, float] = field(
        default_factory=lambda: dict(CYP2D6_FREQUENCIES_US)
    )
    cyp2c19_frequencies: Mapping[CYPPhenotype, float] = field(
        default_factory=lambda: dict(CYP2C19_FREQUENCIES_US)
    )
    cyp2c9_frequencies: Mapping[CYPPhenotype, float] = field(
        default_factory=lambda: dict(CYP2C9_FREQUENCIES_US)
    )
    cyp3a4_frequencies: Mapping[CYPPhenotype, float] = field(
        default_factory=lambda: dict(CYP_DEFAULT_NORMAL)
    )
    cyp1a2_frequencies: Mapping[CYPPhenotype, float] = field(
        default_factory=lambda: dict(CYP_DEFAULT_NORMAL)
    )
    cyp2b6_frequencies: Mapping[CYPPhenotype, float] = field(
        default_factory=lambda: dict(CYP_DEFAULT_NORMAL)
    )

    # Organ function
    organ: OrganFunctionDistribution = field(
        default_factory=OrganFunctionDistribution
    )

    # Lifestyle
    smoker_probability: float = 0.15

    # ── validation ──────────────────────────────────────────────

    def __post_init__(self) -> None:
        self._validate_frequencies(
            "cyp2d6_frequencies", self.cyp2d6_frequencies
        )
        self._validate_frequencies(
            "cyp2c19_frequencies", self.cyp2c19_frequencies
        )
        self._validate_frequencies(
            "cyp2c9_frequencies", self.cyp2c9_frequencies
        )
        self._validate_frequencies(
            "cyp3a4_frequencies", self.cyp3a4_frequencies
        )
        self._validate_frequencies(
            "cyp1a2_frequencies", self.cyp1a2_frequencies
        )
        self._validate_frequencies(
            "cyp2b6_frequencies", self.cyp2b6_frequencies
        )
        if not 0.0 <= self.sex_male_probability <= 1.0:
            raise ValueError(
                f"sex_male_probability must be in [0, 1], "
                f"got {self.sex_male_probability}"
            )
        if not 0.0 <= self.smoker_probability <= 1.0:
            raise ValueError(
                f"smoker_probability must be in [0, 1], "
                f"got {self.smoker_probability}"
            )

    @staticmethod
    def _validate_frequencies(
        name: str, freqs: Mapping[CYPPhenotype, float]
    ) -> None:
        total = sum(freqs.values())
        if abs(total - 1.0) > 1.0e-3:
            raise ValueError(
                f"{name} must sum to 1.0 (got {total:.4f})"
            )
        for pheno, f in freqs.items():
            if f < 0.0:
                raise ValueError(
                    f"{name}[{pheno.name}] must be ≥ 0 (got {f})"
                )


# ════════════════════════════════════════════════════════════════
# CATEGORICAL SAMPLING HELPER
# ════════════════════════════════════════════════════════════════


def categorical_phenotype(
    rng, frequencies: Mapping[CYPPhenotype, float]
) -> CYPPhenotype:
    """Sample a :class:`CYPPhenotype` from a frequency mapping.

    rng is any object exposing ``.random()`` (e.g. random.Random,
    numpy Generator is not accepted directly — pass a Python
    random.Random for reproducibility).
    """
    u = rng.random()
    cumulative = 0.0
    last_pheno: CYPPhenotype = CYPPhenotype.NORMAL
    for pheno, freq in frequencies.items():
        cumulative += freq
        last_pheno = pheno
        if u < cumulative:
            return pheno
    return last_pheno


def bounded_phenotype_set() -> Tuple[CYPPhenotype, ...]:
    """Return the canonical ordering of phenotypes for deterministic iteration."""
    return (
        CYPPhenotype.POOR,
        CYPPhenotype.INTERMEDIATE,
        CYPPhenotype.NORMAL,
        CYPPhenotype.ULTRA_RAPID,
    )
