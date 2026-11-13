"""
Multi-Dose Steady-State Simulation
====================================
Simulates repeated drug dosing over multiple days.
Shows accumulation, steady-state, trough/peak levels.

Real patients don't take one dose — they take drugs 3x/day for years.
This module superimposes multiple single-dose PBPK simulations with time offsets.

Key concepts:
- Accumulation ratio: Css / C_single_dose (depends on dosing interval vs half-life)
- Time to steady state: ~5 × half-life
- Trough (Cmin): lowest concentration just before next dose
- Peak (Cmax,ss): highest concentration at steady state
- Peak-to-trough ratio: Cmax,ss / Cmin,ss (should be <4 for most drugs)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DosingRegimen:
    """Defines how a patient takes a drug."""
    dose_mg: float = 100.0           # Dose per administration
    interval_h: float = 8.0          # Hours between doses (q8h = every 8 hours)
    n_doses: int = 15                # Total number of doses
    route: str = "oral"              # oral, iv
    loading_dose_mg: float = 0.0     # Optional first-dose loading dose

    @property
    def total_duration_h(self) -> float:
        return self.interval_h * self.n_doses

    @property
    def frequency_label(self) -> str:
        labels = {
            4: "QID (every 4h)",
            6: "QID (every 6h)",
            8: "TID (every 8h)",
            12: "BID (every 12h)",
            24: "QD (once daily)",
            48: "Every 48h",
            168: "Weekly",
        }
        return labels.get(int(self.interval_h), f"Every {self.interval_h}h")


# ════════════════════════════════════════════════════════════════
# COMMON REGIMENS DATABASE
# ════════════════════════════════════════════════════════════════

COMMON_REGIMENS = {
    "aspirin_analgesic": DosingRegimen(dose_mg=500, interval_h=6, n_doses=20, route="oral"),
    "aspirin_antiplatelet": DosingRegimen(dose_mg=81, interval_h=24, n_doses=14, route="oral"),
    "caffeine_moderate": DosingRegimen(dose_mg=200, interval_h=8, n_doses=9, route="oral"),
    "ibuprofen_standard": DosingRegimen(dose_mg=400, interval_h=6, n_doses=16, route="oral"),
    "paracetamol_standard": DosingRegimen(dose_mg=1000, interval_h=6, n_doses=16, route="oral"),
    "metformin_standard": DosingRegimen(dose_mg=500, interval_h=12, n_doses=14, route="oral"),
    "metformin_max": DosingRegimen(dose_mg=1000, interval_h=8, n_doses=21, route="oral"),
    "amoxicillin_standard": DosingRegimen(dose_mg=500, interval_h=8, n_doses=21, route="oral"),
    "warfarin_standard": DosingRegimen(dose_mg=5, interval_h=24, n_doses=14, route="oral"),
    "omeprazole_standard": DosingRegimen(dose_mg=20, interval_h=24, n_doses=14, route="oral"),
    "atorvastatin_standard": DosingRegimen(dose_mg=20, interval_h=24, n_doses=14, route="oral"),
    "lisinopril_standard": DosingRegimen(dose_mg=10, interval_h=24, n_doses=14, route="oral"),
    "metoprolol_standard": DosingRegimen(dose_mg=50, interval_h=12, n_doses=14, route="oral"),
}


# ════════════════════════════════════════════════════════════════
# MULTI-DOSE SUPERPOSITION ENGINE
# ════════════════════════════════════════════════════════════════

def simulate_multidose(
    single_dose_solution: dict,
    regimen: DosingRegimen,
    half_life_h: float = 4.0,
) -> dict:
    """
    Simulate multiple doses by superimposing shifted single-dose PBPK curves.

    This uses the principle of superposition — valid for linear PK systems.
    Each new dose adds a fresh copy of the single-dose concentration curve,
    shifted in time by the dosing interval.

    Args:
        single_dose_solution: Output from PBPKModel.solve() for ONE dose
        regimen: DosingRegimen specifying doses, intervals, count
        half_life_h: Drug half-life (for steady-state calculations)

    Returns:
        Dict with multi-dose time-concentration data and metrics
    """
    # Extract single-dose data
    t_single = single_dose_solution["time_h"]
    c_single = single_dose_solution["plasma_conc"]

    # Concentration keys to superimpose
    conc_keys = [k for k in single_dose_solution if k.endswith("_conc")]

    # Create high-resolution time array covering all doses + elimination
    total_time = regimen.total_duration_h + 5 * half_life_h
    dt = 0.05  # 3-minute resolution
    t_multi = np.arange(0, total_time, dt)
    n_points = len(t_multi)

    # Initialize concentration arrays for all compartments
    multi_conc = {key: np.zeros(n_points) for key in conc_keys}

    # Superimpose each dose
    dose_times = []
    for i in range(regimen.n_doses):
        t_dose = i * regimen.interval_h
        dose_times.append(t_dose)

        # Dose scaling (loading dose for first dose if specified)
        if i == 0 and regimen.loading_dose_mg > 0:
            dose_scale = regimen.loading_dose_mg / regimen.dose_mg
        else:
            dose_scale = 1.0

        # Add shifted single-dose curve to each compartment
        for key in conc_keys:
            c_data = single_dose_solution.get(key, np.zeros_like(t_single))
            if c_data is None or len(c_data) == 0:
                continue
            # Interpolate the single-dose curve at shifted time points
            t_shifted = t_multi - t_dose
            c_interp = np.interp(t_shifted, t_single, c_data, left=0, right=0)
            multi_conc[key] += c_interp * dose_scale

    # ── Compute steady-state metrics ──
    plasma = multi_conc.get("plasma_conc", np.zeros(n_points))

    # Find the last complete dosing interval
    last_dose_time = dose_times[-1] if dose_times else 0
    second_last_dose = dose_times[-2] if len(dose_times) > 1 else 0

    # Steady-state metrics (from last dosing interval)
    ss_mask = (t_multi >= second_last_dose) & (t_multi < last_dose_time + regimen.interval_h)
    if np.any(ss_mask):
        plasma_ss = plasma[ss_mask]
        cmax_ss = float(np.max(plasma_ss))
        cmin_ss = float(np.min(plasma_ss))
        cavg_ss = float(np.mean(plasma_ss))
        try:
            auc_ss = float(np.trapezoid(plasma_ss, t_multi[ss_mask]))
        except AttributeError:
            # NumPy < 2.0 compatibility
            auc_ss = float(np.trapz(plasma_ss, t_multi[ss_mask]))
    else:
        cmax_ss = float(np.max(plasma)) if len(plasma) > 0 else 0
        cmin_ss = float(np.min(plasma)) if len(plasma) > 0 else 0
        cavg_ss = float(np.mean(plasma)) if len(plasma) > 0 else 0
        auc_ss = 0

    # First dose metrics
    first_interval_mask = t_multi < regimen.interval_h
    if np.any(first_interval_mask):
        cmax_first = float(np.max(plasma[first_interval_mask]))
    else:
        cmax_first = 0

    # Accumulation ratio
    accumulation_ratio = cmax_ss / cmax_first if cmax_first > 0 else 1.0

    # Peak-to-trough ratio
    peak_trough_ratio = cmax_ss / cmin_ss if cmin_ss > 0 else float('inf')

    # Time to steady state (when Cmax reaches 90% of Cmax_ss)
    t_ss = 0
    for i in range(len(dose_times)):
        interval_mask = (t_multi >= dose_times[i]) & (t_multi < dose_times[i] + regimen.interval_h)
        if np.any(interval_mask):
            cmax_i = float(np.max(plasma[interval_mask]))
            if cmax_i >= 0.90 * cmax_ss:
                t_ss = dose_times[i]
                break

    # Theoretical steady-state time (5 half-lives)
    t_ss_theoretical = 5 * half_life_h

    # Downsample for output (keep ~500 points max for UI performance)
    step = max(1, len(t_multi) // 500)
    t_out = t_multi[::step]
    conc_out = {key: vals[::step] for key, vals in multi_conc.items()}

    # Build time course dicts for UI
    time_course = []
    plasma_out = conc_out.get("plasma_conc", np.zeros(len(t_out)))
    liver_out = conc_out.get("liver_conc", np.zeros(len(t_out)))
    kidney_out = conc_out.get("kidney_conc", np.zeros(len(t_out)))
    gut_out = conc_out.get("gut_conc", np.zeros(len(t_out)))
    brain_out = conc_out.get("brain_conc", np.zeros(len(t_out)))

    for i in range(len(t_out)):
        time_course.append({
            "time_h": round(float(t_out[i]), 3),
            "plasma_conc": round(float(plasma_out[i]), 4),
            "liver_conc": round(float(liver_out[i]), 4),
            "kidney_conc": round(float(kidney_out[i]), 4),
            "gut_conc": round(float(gut_out[i]), 4),
            "brain_conc": round(float(brain_out[i]), 4),
        })

    return {
        "time_course": time_course,
        "dose_times": dose_times,
        "metrics": {
            "cmax_ss": round(cmax_ss, 4),
            "cmin_ss": round(cmin_ss, 4),
            "cavg_ss": round(cavg_ss, 4),
            "auc_ss": round(auc_ss, 4),
            "cmax_first_dose": round(cmax_first, 4),
            "accumulation_ratio": round(accumulation_ratio, 2),
            "peak_trough_ratio": round(peak_trough_ratio, 2),
            "time_to_steady_state_h": round(t_ss, 1),
            "theoretical_tss_h": round(t_ss_theoretical, 1),
            "half_life_h": round(half_life_h, 2),
            "dosing_interval_h": regimen.interval_h,
            "n_doses": regimen.n_doses,
            "total_duration_h": round(regimen.total_duration_h, 1),
            "frequency": regimen.frequency_label,
        },
        "steady_state_achieved": t_ss > 0 and t_ss < regimen.total_duration_h,
        "warnings": _generate_warnings(cmax_ss, cmin_ss, peak_trough_ratio, accumulation_ratio),
    }


def _generate_warnings(cmax_ss, cmin_ss, ptr, accum_ratio) -> list[str]:
    """Generate clinical warnings based on steady-state metrics."""
    warnings = []
    if accum_ratio > 3.0:
        warnings.append(f"High accumulation ratio ({accum_ratio:.1f}x) — risk of toxicity with repeated dosing")
    if ptr > 10.0:
        warnings.append(f"Wide peak-trough fluctuation ({ptr:.1f}x) — consider more frequent dosing or extended-release formulation")
    if cmin_ss <= 0.001:
        warnings.append("Trough levels near zero — drug may be sub-therapeutic before next dose")
    return warnings
