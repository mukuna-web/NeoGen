"""NeoGen V1 Population Module — virtual-patient Monte Carlo.

Public API
----------

    from neogen.population import PopulationSampler, PopulationRunner

    # Sample 100 virtual patients with a fixed seed
    pop = PopulationSampler().sample(n=100, seed=42)

    # Run the V1 engine on every patient
    result = PopulationRunner().simulate(
        drug_name="midazolam",
        dose_mg=5.0,
        route="iv_bolus",
        population=pop,
    )

    print(result.to_markdown())

Components
----------

  * **PopulationDistributions** — priors for weight, height, age,
    sex, CYP phenotypes, and organ function. Defaults to a US
    adult reference population.
  * **PopulationSampler** — draws :class:`VirtualPopulation`
    instances deterministically when seeded.
  * **scale_physiology_to_patient / scale_drug_to_genotype** —
    pure functions that produce per-patient copies of
    :class:`PhysiologyModel` and :class:`Drug`, ready to pass
    into the unchanged V1 engine.
  * **PopulationRunner** — Monte Carlo harness that wires sampler,
    scaling, and the V1 orchestrator into a single call.
  * **PopulationResult** — per-patient results plus percentile
    summaries (5th / median / 95th / geomean CV).

Design principles
-----------------

  * Zero modifications to :mod:`neogen.engine` or
    :mod:`neogen.core`. All scaling happens before the engine is
    called; scaled entities are produced via
    :func:`dataclasses.replace` and :mod:`copy.deepcopy` so
    originals are never mutated.
  * Seed in → deterministic result out. Reproducibility across
    processes is required for validation work.
  * All distributions configurable. Subpopulation-specific
    allele frequencies, custom age ranges, custom organ-function
    priors all flow through a single
    :class:`PopulationDistributions` dataclass.
"""

from neogen.population.distributions import (
    AgeDistribution,
    CYP_DEFAULT_NORMAL,
    CYP2C19_FREQUENCIES_US,
    CYP2C9_FREQUENCIES_US,
    CYP2D6_FREQUENCIES_US,
    HeightDistribution,
    OrganFunctionDistribution,
    PopulationDistributions,
    WeightDistribution,
    bounded_phenotype_set,
    categorical_phenotype,
)
from neogen.population.result import (
    PatientSimulationResult,
    PercentileSummary,
    PopulationResult,
)
from neogen.population.runner import PopulationRunner
from neogen.population.sampler import PopulationSampler, VirtualPopulation
from neogen.population.scaling import (
    effective_gfr_ml_per_min,
    genotype_clint_multiplier,
    scale_drug_to_genotype,
    scale_for_patient,
    scale_physiology_to_patient,
)

__all__ = [
    # Distributions
    "PopulationDistributions",
    "WeightDistribution",
    "HeightDistribution",
    "AgeDistribution",
    "OrganFunctionDistribution",
    "CYP2D6_FREQUENCIES_US",
    "CYP2C19_FREQUENCIES_US",
    "CYP2C9_FREQUENCIES_US",
    "CYP_DEFAULT_NORMAL",
    "categorical_phenotype",
    "bounded_phenotype_set",
    # Sampler
    "PopulationSampler",
    "VirtualPopulation",
    # Scaling
    "scale_drug_to_genotype",
    "scale_physiology_to_patient",
    "scale_for_patient",
    "genotype_clint_multiplier",
    "effective_gfr_ml_per_min",
    # Runner + results
    "PopulationRunner",
    "PopulationResult",
    "PatientSimulationResult",
    "PercentileSummary",
]
