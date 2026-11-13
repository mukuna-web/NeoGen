"""Concentration-time profile validator — placeholder.

Validates simulated C-t profiles against digitized clinical data.
Currently a placeholder with the interface defined but no clinical
profile data loaded.

When clinical profile data becomes available (digitized from
published figures), this module will:

1. Load digitized C-t curves as (time_h, conc_mg_per_L) arrays.
2. Align simulated and observed time grids via interpolation.
3. Compute goodness-of-fit metrics:
   - Visual predictive check (VPC): fraction of observed points
     within the 5th-95th percentile prediction interval
   - Mean absolute error (MAE) on log-transformed concentrations
   - Root mean squared error (RMSE)
   - Coefficient of determination (R^2)
4. Return a per-drug ProfileValidationResult.

Data format
-----------
Clinical profiles should be stored as CSV files in
``data/validation/ct_profiles/`` with columns:
  - drug_name (str)
  - dose_mg (float)
  - route (str: "IV" or "ORAL")
  - time_h (float)
  - conc_mg_per_L (float)
  - source (str: citation)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileValidationResult:
    """Validation result for a single C-t profile comparison.

    Attributes
    ----------
    drug_name
        Name of the drug.
    dose_mg
        Dose in mg.
    route
        "IV" or "ORAL".
    mae_log
        Mean absolute error on log10-transformed concentrations.
    rmse
        Root mean squared error on linear concentrations.
    r_squared
        Coefficient of determination on log10 concentrations.
    fraction_within_2fold
        Fraction of observed time points where the simulated
        concentration is within 2-fold of observed.
    n_timepoints
        Number of compared time points.
    """

    drug_name: str
    dose_mg: float
    route: str
    mae_log: Optional[float] = None
    rmse: Optional[float] = None
    r_squared: Optional[float] = None
    fraction_within_2fold: Optional[float] = None
    n_timepoints: int = 0


def validate_profile(
    drug_name: str,
    dose_mg: float,
    route: str,
    simulated_time_h: Sequence[float],
    simulated_conc: Sequence[float],
    observed_time_h: Optional[Sequence[float]] = None,
    observed_conc: Optional[Sequence[float]] = None,
) -> ProfileValidationResult:
    """Compare simulated C-t profile against observed clinical data.

    Parameters
    ----------
    drug_name
        Drug identifier.
    dose_mg
        Dose in mg.
    route
        "IV" or "ORAL".
    simulated_time_h
        Simulated time grid (hours).
    simulated_conc
        Simulated plasma concentrations (mg/L).
    observed_time_h
        Observed (digitized) time points. None if no data available.
    observed_conc
        Observed concentrations at those time points. None if no data.

    Returns
    -------
    ProfileValidationResult
        With metrics populated if observed data is available, otherwise
        with all metrics as None.
    """
    if observed_time_h is None or observed_conc is None:
        logger.info(
            "No observed C-t profile data for %s %s %.0f mg",
            drug_name, route, dose_mg,
        )
        return ProfileValidationResult(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
        )

    if len(observed_time_h) != len(observed_conc):
        raise ValueError(
            "observed_time_h and observed_conc must have the same length"
        )

    sim_t = np.asarray(simulated_time_h, dtype=float)
    sim_c = np.asarray(simulated_conc, dtype=float)
    obs_t = np.asarray(observed_time_h, dtype=float)
    obs_c = np.asarray(observed_conc, dtype=float)

    # Interpolate simulated onto observed time grid
    sim_interp = np.interp(obs_t, sim_t, sim_c)

    # Filter out zero/negative concentrations for log metrics
    mask = (obs_c > 0) & (sim_interp > 0)
    n_valid = int(np.sum(mask))

    if n_valid == 0:
        return ProfileValidationResult(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            n_timepoints=0,
        )

    obs_valid = obs_c[mask]
    sim_valid = sim_interp[mask]

    # MAE on log10
    log_obs = np.log10(obs_valid)
    log_sim = np.log10(sim_valid)
    mae_log = float(np.mean(np.abs(log_sim - log_obs)))

    # RMSE on linear scale
    rmse = float(np.sqrt(np.mean((sim_valid - obs_valid) ** 2)))

    # R^2 on log10
    ss_res = float(np.sum((log_sim - log_obs) ** 2))
    ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Fraction within 2-fold
    ratios = sim_valid / obs_valid
    within_2fold = float(np.mean((ratios >= 0.5) & (ratios <= 2.0)))

    return ProfileValidationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        mae_log=round(mae_log, 4),
        rmse=round(rmse, 4),
        r_squared=round(r_squared, 4),
        fraction_within_2fold=round(within_2fold, 4),
        n_timepoints=n_valid,
    )


def load_clinical_profiles(
    data_dir: Optional[str] = None,
) -> Dict[str, List[Tuple[Sequence[float], Sequence[float]]]]:
    """Load digitized clinical C-t profiles from CSV files.

    Parameters
    ----------
    data_dir
        Path to the directory containing CSV profile files.
        Default: ``data/validation/ct_profiles/``.

    Returns
    -------
    dict
        Mapping drug_name -> list of (time_h, conc_mg_per_L) tuples.
        Currently returns empty dict (placeholder — no data files yet).
    """
    logger.info(
        "Clinical C-t profile loading is a placeholder. "
        "No digitized profile data available yet."
    )
    return {}
