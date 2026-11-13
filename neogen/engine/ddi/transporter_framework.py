"""Multi-Transporter DDI Framework — Phase 6.1 Advanced Capabilities.

Extends the V1.2 OATP1B1 architecture (``transporter_ddi.py``) to a
registry-based framework supporting multiple transporters. This allows
systematic evaluation of transporter-mediated DDIs across:

* **OATP1B1** — hepatic sinusoidal uptake (statins, repaglinide)
* **OCT2** — renal proximal tubule secretion (metformin, cisplatin)
* **BCRP** — gut/liver efflux (rosuvastatin, sulfasalazine)

Architecture
------------
Each transporter is described by a ``TransporterDDIConfig`` that maps
Drug entity attributes to DDI evaluation logic. The registry pattern
allows new transporters to be added without modifying core engine code.

The framework provides:

1. ``TransporterDDIConfig`` — per-transporter configuration
2. ``TRANSPORTER_REGISTRY`` — all supported transporters
3. ``evaluate_transporter_ddi()`` — generic IC50-based DDI evaluation
4. ``evaluate_all_transporter_ddis()`` — batch evaluation across registry

The actual OATP1B1 DDI logic remains in ``transporter_ddi.py``. This
module provides the *framework* for extending to other transporters
when clinical validation data becomes available.

References: Giacomini 2010 (ITC whitepaper), Hillgren 2013,
FDA DDI Guidance 2020 (transporter section).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from neogen.core.entities.drug import Drug


# ════════════════════════════════════════════════════════════════
# TRANSPORTER CONFIGURATION
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TransporterDDIConfig:
    """Configuration for a single transporter DDI evaluation.

    Attributes
    ----------
    transporter_name
        Human-readable transporter identifier (e.g. "OATP1B1").
    substrate_attr
        Drug.transporters attribute name to check substrate status
        (e.g. "is_oatp1b1_substrate").
    ic50_attr
        Drug.transporters attribute name for the IC50 value
        (e.g. "oatp1b1_ic50").
    default_uptake_factor
        Baseline amplification factor for the transporter. For
        uptake transporters (OATP1B1), this amplifies CLint. For
        efflux transporters (BCRP, P-gp), this reduces effective
        permeability. Value of 1.0 means no baseline effect.
    compartment
        Primary site of action: "hepatic", "renal", "intestinal".
    direction
        Transport direction: "uptake" or "efflux".
    """

    transporter_name: str
    substrate_attr: str
    ic50_attr: str
    default_uptake_factor: float
    compartment: str = "hepatic"
    direction: str = "uptake"


# ════════════════════════════════════════════════════════════════
# TRANSPORTER REGISTRY
# ════════════════════════════════════════════════════════════════

TRANSPORTER_REGISTRY: List[TransporterDDIConfig] = [
    TransporterDDIConfig(
        transporter_name="OATP1B1",
        substrate_attr="is_oatp1b1_substrate",
        ic50_attr="oatp1b1_ic50",
        default_uptake_factor=15.0,
        compartment="hepatic",
        direction="uptake",
    ),
    TransporterDDIConfig(
        transporter_name="OCT2",
        substrate_attr="is_oct2_substrate",
        ic50_attr="oct2_ic50",
        default_uptake_factor=1.0,
        compartment="renal",
        direction="uptake",
    ),
    TransporterDDIConfig(
        transporter_name="BCRP",
        substrate_attr="is_bcrp_substrate",
        ic50_attr="bcrp_ic50",
        default_uptake_factor=1.0,
        compartment="intestinal",
        direction="efflux",
    ),
]


# ════════════════════════════════════════════════════════════════
# EVALUATION RESULT
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TransporterDDIResult:
    """Result of a single transporter DDI evaluation.

    Attributes
    ----------
    transporter_name
        Which transporter was evaluated.
    victim_is_substrate
        Whether the victim drug is a substrate.
    inhibitor_ic50_uM
        The perpetrator's IC50 for this transporter (uM), or None.
    inhibitor_conc_uM
        Unbound perpetrator concentration at the site (uM).
    fold_change
        Predicted fold-change in the relevant PK parameter.
        For uptake inhibition: AUC ratio. For efflux inhibition:
        absorption ratio. 1.0 = no effect.
    """

    transporter_name: str
    victim_is_substrate: bool
    inhibitor_ic50_uM: Optional[float]
    inhibitor_conc_uM: Optional[float]
    fold_change: float


# ════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ════════════════════════════════════════════════════════════════


def evaluate_transporter_ddi(
    config: TransporterDDIConfig,
    victim: Drug,
    perpetrator: Drug,
    perpetrator_conc_uM: float,
) -> TransporterDDIResult:
    """Evaluate a single transporter-mediated DDI.

    Uses the standard static IC50 model::

        R = 1 + [I]_u / IC50

    For uptake transporters (OATP1B1, OCT2), inhibition reduces
    transporter-mediated clearance, increasing victim AUC by factor R.

    For efflux transporters (BCRP), inhibition increases absorption
    by reducing efflux, also described by factor R applied to effective
    permeability.

    Parameters
    ----------
    config
        Transporter configuration from the registry.
    victim
        The substrate drug.
    perpetrator
        The inhibitor drug.
    perpetrator_conc_uM
        Unbound perpetrator concentration at the transporter site (uM).

    Returns
    -------
    TransporterDDIResult
        Contains the fold-change prediction.
    """
    victim_transporters = victim.transporters
    perp_transporters = perpetrator.transporters

    # Check if victim is a substrate
    is_substrate = getattr(victim_transporters, config.substrate_attr, False)

    # Get perpetrator IC50
    ic50 = getattr(perp_transporters, config.ic50_attr, None)

    if not is_substrate or ic50 is None or ic50 <= 0:
        return TransporterDDIResult(
            transporter_name=config.transporter_name,
            victim_is_substrate=is_substrate,
            inhibitor_ic50_uM=ic50,
            inhibitor_conc_uM=perpetrator_conc_uM,
            fold_change=1.0,
        )

    # Static IC50 model: R = 1 + [I]u / IC50
    r_value = 1.0 + perpetrator_conc_uM / ic50

    return TransporterDDIResult(
        transporter_name=config.transporter_name,
        victim_is_substrate=is_substrate,
        inhibitor_ic50_uM=ic50,
        inhibitor_conc_uM=perpetrator_conc_uM,
        fold_change=r_value,
    )


def evaluate_all_transporter_ddis(
    victim: Drug,
    perpetrator: Drug,
    perpetrator_conc_uM: float,
    registry: Sequence[TransporterDDIConfig] = TRANSPORTER_REGISTRY,
) -> Dict[str, TransporterDDIResult]:
    """Evaluate DDIs across all registered transporters.

    Parameters
    ----------
    victim
        The substrate drug.
    perpetrator
        The inhibitor drug.
    perpetrator_conc_uM
        Unbound perpetrator concentration (uM).
    registry
        Transporter configurations to evaluate. Defaults to the
        full ``TRANSPORTER_REGISTRY``.

    Returns
    -------
    dict
        Mapping transporter name to TransporterDDIResult.
    """
    results: Dict[str, TransporterDDIResult] = {}
    for config in registry:
        results[config.transporter_name] = evaluate_transporter_ddi(
            config, victim, perpetrator, perpetrator_conc_uM
        )
    return results


def get_transporter_config(name: str) -> Optional[TransporterDDIConfig]:
    """Look up a transporter configuration by name.

    Parameters
    ----------
    name
        Transporter name (case-insensitive).

    Returns
    -------
    TransporterDDIConfig or None
    """
    name_upper = name.upper()
    for config in TRANSPORTER_REGISTRY:
        if config.transporter_name.upper() == name_upper:
            return config
    return None
