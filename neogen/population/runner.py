"""PopulationRunner — Monte Carlo orchestration over a virtual cohort.

Takes a base drug + schedule + a :class:`VirtualPopulation`, scales
the drug and physiology per patient, calls the existing
:class:`SimulationOrchestrator` for each, and aggregates results
into a :class:`PopulationResult`.

Parallelization
---------------
Parallel execution is **available as an opt-in** via
``n_workers=N`` or ``n_workers='auto'``. The default is
``n_workers=1`` (serial) because empirical benchmarks on short
fast-PK simulations (midazolam, 100 patients) showed threading is
actually slower than serial: ~1.35 s serial vs ~1.9 s with 4
threads. The reason is that ``scipy.solve_ivp`` releases the GIL
during the BDF C code, but the surrounding Python setup (Drug
building, Kp / clearance computation, dose schedule assembly)
dominates for fast drugs and stays GIL-serialized. Threading
creates net contention with no integration-level payoff.

For longer / slower simulations (drugs with 20+ hour half-lives,
denser time grids, or 500+ patient cohorts) threading can give a
real 1.5-2× speedup on a 4-core machine — empirical testing
recommended per-cohort. Use ``n_workers='auto'`` to try it.

Determinism is preserved under any ``n_workers`` value: results
are returned in the same order as ``population.patients`` and
the seed semantics are set before dispatch, so
``PopulationResult.patient_results[i]`` always corresponds to
``population.patients[i]`` regardless of which thread finishes
first. Verified by a smoke test: 100/100 patient results are
byte-identical between serial and 4-thread runs.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union

from neogen.core.entities.drug import Drug
from neogen.core.entities.formulation import DoseSchedule
from neogen.core.entities.physiology import PhysiologyModel
from neogen.core.constants.reference_physiology import build_reference_male
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.orchestrator import SimulationOrchestrator
from neogen.population.result import (
    PatientSimulationResult,
    PopulationResult,
)
from neogen.population.sampler import VirtualPopulation
from neogen.population.scaling import (
    scale_drug_to_genotype,
    scale_physiology_to_patient,
)

logger = logging.getLogger(__name__)


WorkerCount = Union[int, str]
"""Type alias: ``int`` for explicit count, or ``'auto'`` for
``min(8, os.cpu_count())``."""


class PopulationRunner:
    """Run a V1 simulation across every patient in a virtual population.

    Usage::

        sampler = PopulationSampler()
        pop = sampler.sample(n=100, seed=42)

        runner = PopulationRunner()
        result = runner.simulate(
            drug_name="midazolam",
            dose_mg=5.0,
            route="iv_bolus",
            population=pop,
        )
        print(result.to_markdown())
    """

    def __init__(
        self,
        orchestrator: Optional[SimulationOrchestrator] = None,
        factory: Optional[DrugFactory] = None,
        base_physiology: Optional[PhysiologyModel] = None,
        n_workers: WorkerCount = 1,
    ) -> None:
        self._orch = orchestrator or SimulationOrchestrator()
        self._factory = factory or DrugFactory()
        self._base_physiology = base_physiology or build_reference_male()
        self._scheduler = DoseScheduler()
        self._n_workers = _resolve_n_workers(n_workers)

    # ── public API ──────────────────────────────────────────────

    def simulate(
        self,
        drug_name: str,
        dose_mg: float,
        route: str,
        population: VirtualPopulation,
        *,
        label: str = "",
    ) -> PopulationResult:
        """Run the V1 engine over every patient in ``population``.

        route ∈ {'iv_bolus', 'oral', 'iv_infusion'}.

        Patients are dispatched to ``n_workers`` threads (default
        auto) with scipy releasing the GIL during ODE integration.
        Results are returned in the same order as
        ``population.patients`` regardless of completion order.
        """
        base_drug = self._factory.create(drug_name)
        schedule = self._build_schedule(drug_name, dose_mg, route)

        t0 = time.time()

        patient_results: List[PatientSimulationResult]
        if self._n_workers <= 1 or len(population) <= 1:
            # Serial path — no thread overhead for single-patient runs.
            patient_results = [
                self._simulate_one(base_drug, schedule, patient)
                for patient in population
            ]
        else:
            patient_results = self._simulate_parallel(
                base_drug, schedule, population
            )

        runtime = time.time() - t0

        result = PopulationResult(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            n_requested=len(population),
            seed=population.seed,
            patient_results=tuple(patient_results),
            runtime_s=runtime,
            label=label,
        )
        logger.info(
            "Population run %s: %d/%d completed in %.1f s (%d workers)",
            drug_name,
            result.n_completed,
            result.n_requested,
            runtime,
            self._n_workers,
        )
        return result

    def _simulate_parallel(
        self,
        base_drug: Drug,
        schedule: DoseSchedule,
        population: VirtualPopulation,
    ) -> List[PatientSimulationResult]:
        """Thread-pooled patient dispatch preserving input order."""
        n = len(population)
        out: List[Optional[PatientSimulationResult]] = [None] * n
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._n_workers
        ) as pool:
            future_to_idx = {
                pool.submit(
                    self._simulate_one, base_drug, schedule, patient
                ): idx
                for idx, patient in enumerate(population)
            }
            for fut in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    out[idx] = fut.result()
                except Exception as exc:  # pragma: no cover — defensive
                    logger.exception(
                        "Worker raised on patient idx %d: %s", idx, exc
                    )
                    out[idx] = PatientSimulationResult(
                        patient=population.patients[idx],
                        drug_name=base_drug.name,
                        dose_mg=schedule.events[0].dose_mg if schedule.events else 0.0,
                        route=schedule.events[0].route.name if schedule.events else "",
                        error=f"worker: {exc}",
                    )
        # Replace any Nones with a defensive error record (should not happen).
        return [
            r
            if r is not None
            else PatientSimulationResult(
                patient=population.patients[i],
                drug_name=base_drug.name,
                dose_mg=schedule.events[0].dose_mg if schedule.events else 0.0,
                route=schedule.events[0].route.name if schedule.events else "",
                error="worker: no result",
            )
            for i, r in enumerate(out)
        ]

    # ── internal ────────────────────────────────────────────────

    def _build_schedule(
        self, drug_name: str, dose_mg: float, route: str
    ) -> DoseSchedule:
        route_lc = route.lower()
        if route_lc in ("iv_bolus", "iv"):
            return self._scheduler.iv_bolus(drug_name, dose_mg)
        if route_lc == "oral":
            return self._scheduler.single_oral(drug_name, dose_mg)
        if route_lc in ("iv_infusion", "infusion"):
            return self._scheduler.iv_infusion(drug_name, dose_mg)
        raise ValueError(
            f"Unsupported route for population simulation: {route!r}. "
            f"Supported: iv_bolus, oral, iv_infusion."
        )

    def _simulate_one(
        self,
        base_drug: Drug,
        schedule: DoseSchedule,
        patient,
    ) -> PatientSimulationResult:
        """Run one patient's simulation and catch any engine errors."""
        try:
            scaled_drug = scale_drug_to_genotype(base_drug, patient)
            scaled_phys = scale_physiology_to_patient(
                self._base_physiology, patient
            )
        except Exception as exc:
            logger.warning(
                "Scaling failed for patient (%s, %s): %s",
                base_drug.name, patient.sex.value, exc,
            )
            return PatientSimulationResult(
                patient=patient,
                drug_name=base_drug.name,
                dose_mg=schedule.events[0].dose_mg if schedule.events else 0.0,
                route=schedule.events[0].route.name if schedule.events else "",
                error=f"scaling: {exc}",
            )

        try:
            result = self._orch.run(scaled_drug, schedule, scaled_phys)
        except Exception as exc:
            logger.warning(
                "Engine run failed for patient (%s, %s): %s",
                base_drug.name, patient.sex.value, exc,
            )
            return PatientSimulationResult(
                patient=patient,
                drug_name=base_drug.name,
                dose_mg=schedule.events[0].dose_mg if schedule.events else 0.0,
                route=schedule.events[0].route.name if schedule.events else "",
                error=f"engine: {exc}",
            )

        return PatientSimulationResult(
            patient=patient,
            drug_name=result.drug_name,
            dose_mg=result.dose_mg,
            route=result.route,
            pk_metrics=result.pk_metrics,
            cl_total_L_per_h=result.cl_total_L_per_h,
            solver_success=result.solver_success,
            error="",
        )


# ════════════════════════════════════════════════════════════════
# WORKER-COUNT RESOLUTION
# ════════════════════════════════════════════════════════════════


def _resolve_n_workers(n_workers: WorkerCount) -> int:
    """Resolve the ``n_workers`` parameter to a concrete integer.

    ``'auto'`` maps to ``min(8, os.cpu_count() or 1)``. Explicit
    integers are validated to be ≥ 1. Anything else raises
    ``ValueError``.
    """
    if isinstance(n_workers, str):
        if n_workers.lower() != "auto":
            raise ValueError(
                f"n_workers must be an int or 'auto', got {n_workers!r}"
            )
        cpus = os.cpu_count() or 1
        return max(1, min(8, cpus))
    try:
        n = int(n_workers)
    except (TypeError, ValueError):
        raise ValueError(
            f"n_workers must be int or 'auto', got {n_workers!r}"
        )
    if n < 1:
        raise ValueError(f"n_workers must be ≥ 1, got {n}")
    return n
