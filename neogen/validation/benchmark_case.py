"""Benchmark case definitions — the V1 validation contract.

A :class:`BenchmarkCase` is the minimum information needed to run the
NeoGen engine against a published clinical result and score the
prediction. It is deliberately narrow:

* Drug name (resolved by :class:`DrugFactory`)
* Dose + route (IV bolus, IV infusion, oral — resolved by
  :class:`DoseScheduler`)
* Optional :class:`PatientProfile` for patient-specific cases (elderly,
  PM, CKD, etc.). ``None`` means a healthy 35-year-old reference male.
* Observed clinical reference values for Cmax, AUC, and/or t½ plus a
  citation string.
* An optional tolerance fold (default 2.0) — the classic PBPK
  acceptance criterion.

Design rules
------------

1. **Frozen dataclasses only.** Every field is immutable.
2. **Partial references are allowed.** A case may list Cmax only, or
   AUC only — the scoring layer skips any metric whose observed value
   is ``None``.
3. **Source is mandatory.** Every observed value must be traceable to
   a published paper or drug label. The `source` field is a free-form
   citation ("Olkkola 1994") — the datasets carry the fuller bibliography.
4. **No physics lives here.** This is a contract module only. All
   simulation is handled by :class:`BenchmarkRunner`.

DDI vs single-drug
------------------

Single-drug benchmarks use :class:`BenchmarkCase`. DDI benchmarks use
:class:`DDIBenchmarkCase`, which adds a list of
:class:`PerpetratorExposure` entries and an **observed AUC fold-change**
(AUCR) rather than absolute Cmax / AUC values. This mirrors how the
clinical DDI literature actually reports results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from neogen.core.entities.patient import PatientProfile
from neogen.engine.ddi.ddi_orchestrator import PerpetratorExposure


# ════════════════════════════════════════════════════════════════
# SINGLE-DRUG BENCHMARK CASE
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BenchmarkCase:
    """One clinical reference row for a single-drug PK benchmark.

    Attributes
    ----------
    name
        Human-readable identifier for the case. Appears in the
        scorecard and in error messages. Recommended format:
        ``"{drug}-{route}-{dose_mg}mg[-{condition}]"``, e.g.
        ``"midazolam-iv-5mg"``, ``"midazolam-iv-5mg-child_pugh_c"``.
    drug_name
        Passed directly to :meth:`DrugFactory.create`. Must resolve
        against the literature DB or carry an explicit SMILES override.
    dose_mg
        Total dose in milligrams. For multi-dose schedules use
        ``schedule_kind="multi"`` and supply ``dosing_interval_h``.
    route
        ``"IV"`` or ``"ORAL"``. Case-insensitive; stored canonicalized.
    schedule_kind
        ``"single"`` (default) or ``"multi"``. Single-dose cases use
        an IV bolus or a single oral dose; multi-dose cases repeat
        the same dose every ``dosing_interval_h`` hours until
        ``total_duration_h``.
    dosing_interval_h / total_duration_h
        Only consulted when ``schedule_kind == "multi"``.
    patient
        Optional :class:`PatientProfile` for patient-specific cases.
        ``None`` runs against the reference male.
    observed_cmax_mg_per_L / observed_auc_mg_h_per_L / observed_half_life_h
        Published reference values. ``None`` means "not scored for
        this metric". A case with all three ``None`` is legal but
        useless — the scorecard will skip it.
    tolerance_fold
        Fold tolerance for "within" counting. PBPK convention is 2.0.
    source
        Citation string. Required.
    notes
        Free-form commentary (e.g. "high-extraction, buffered by
        mixing-pool ceiling"). Rendered in the scorecard.
    """

    name: str
    drug_name: str
    dose_mg: float
    route: str  # "IV" or "ORAL"
    source: str

    # Optional — schedule
    schedule_kind: str = "single"  # "single" or "multi"
    dosing_interval_h: Optional[float] = None
    total_duration_h: Optional[float] = None

    # Optional — patient
    patient: Optional[PatientProfile] = None

    # Optional — observed clinical values
    observed_cmax_mg_per_L: Optional[float] = None
    observed_auc_mg_h_per_L: Optional[float] = None
    observed_half_life_h: Optional[float] = None

    # Scoring
    tolerance_fold: float = 2.0

    # Metadata
    notes: str = ""
    role: str = "victim"  # "victim" or "perpetrator" — perpetrators excluded from T1 PK scoring

    def __post_init__(self) -> None:
        if self.tolerance_fold < 1.0:
            raise ValueError(
                f"tolerance_fold must be ≥ 1.0, got {self.tolerance_fold}"
            )
        canon = self.route.upper()
        if canon not in ("IV", "ORAL"):
            raise ValueError(
                f"route must be 'IV' or 'ORAL', got {self.route!r}"
            )
        # Re-set via object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "route", canon)

        if self.schedule_kind not in ("single", "multi"):
            raise ValueError(
                f"schedule_kind must be 'single' or 'multi', "
                f"got {self.schedule_kind!r}"
            )
        if self.schedule_kind == "multi":
            if self.dosing_interval_h is None or self.total_duration_h is None:
                raise ValueError(
                    "multi-dose cases require both dosing_interval_h "
                    "and total_duration_h"
                )

        if not self.source.strip():
            raise ValueError("source citation is required")

    @property
    def has_any_reference(self) -> bool:
        """True if the case has at least one scorable observed metric."""
        return any(
            v is not None
            for v in (
                self.observed_cmax_mg_per_L,
                self.observed_auc_mg_h_per_L,
                self.observed_half_life_h,
            )
        )


# ════════════════════════════════════════════════════════════════
# DDI BENCHMARK CASE
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DDIBenchmarkCase:
    """One clinical reference row for a victim + perpetrator DDI pair.

    The DDI literature reports **AUC fold-change** (AUCR = AUC_ddi /
    AUC_mono) rather than absolute values, because inter-individual
    variability in baseline clearance is huge. The scorecard compares
    the engine's predicted AUCR against the observed AUCR.

    Attributes
    ----------
    name
        Free-form identifier, e.g. ``"ketoconazole+midazolam-iv"``.
    victim_drug_name / perpetrator_drug_name
        Passed to :meth:`DrugFactory.create`.
    victim_dose_mg / victim_route
        Dose of the victim for the scored run.
    perpetrator_steady_state_conc_um
        Unbound liver-site steady-state concentration of the
        perpetrator (μM). Used by the :class:`DDIOrchestrator` when
        building the CLint(t) provider.
    perpetrator_duration_on_h
        How long the perpetrator is on board (hours). For a reversible
        inhibitor at steady state, 72 h is plenty to hit equilibrium.
    include_enzyme_pool
        Whether the Step-9 enzyme-pool evolver is engaged. Leave
        ``False`` for purely reversible inhibitors; set ``True`` for
        MBI / induction cases that need time-dependent enzyme abundance.
    observed_aucr
        Published AUC ratio (AUC_ddi / AUC_mono).
    tolerance_fold
        Fold tolerance for "within" counting.
    source / notes
        Same semantics as :class:`BenchmarkCase`.
    """

    name: str
    victim_drug_name: str
    perpetrator_drug_name: str
    victim_dose_mg: float
    victim_route: str  # "IV" or "ORAL"
    perpetrator_steady_state_conc_um: float
    observed_aucr: float
    source: str

    perpetrator_duration_on_h: float = 72.0
    include_enzyme_pool: bool = False

    # Clinical DDI protocols almost always dose the perpetrator for days
    # *before* giving the probe drug so that induction and MBI reach
    # steady state. 168 h (7 days) is enough to plateau both: rifampicin
    # CYP3A4 induction (t½ ≈ 35 h) is at ~95 % of max by day 7, and
    # clarithromycin-style MBI has fully turned over the CYP pool by
    # day 5. Set to 0 if the reference protocol explicitly uses a single
    # simultaneous co-dose.
    pretreatment_h: float = 168.0

    # Gut-wall DDI channel — luminal perpetrator concentration (µM).
    # When set and the victim is ORAL, the benchmark runner constructs
    # a GutWallMBIExposure and passes it alongside the hepatic
    # PerpetratorExposure. This enables gut-wall MBI (clarithromycin,
    # erythromycin) and gut-wall induction (rifampicin) effects on fg.
    # Typical range: 100–10,000 µM (gut lumen is a small volume and
    # the whole oral perpetrator dose dissolves into it).
    perpetrator_luminal_conc_um: Optional[float] = None

    patient: Optional[PatientProfile] = None
    tolerance_fold: float = 2.0

    notes: str = ""

    def __post_init__(self) -> None:
        if self.observed_aucr <= 0:
            raise ValueError(
                f"observed_aucr must be > 0, got {self.observed_aucr}"
            )
        if self.tolerance_fold < 1.0:
            raise ValueError(
                f"tolerance_fold must be ≥ 1.0, got {self.tolerance_fold}"
            )
        canon = self.victim_route.upper()
        if canon not in ("IV", "ORAL"):
            raise ValueError(
                f"victim_route must be 'IV' or 'ORAL', got {self.victim_route!r}"
            )
        object.__setattr__(self, "victim_route", canon)
        if not self.source.strip():
            raise ValueError("source citation is required")
