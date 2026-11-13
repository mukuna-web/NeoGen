"""
Real-Time Cardiac Action Potential Waveform
=============================================
Generates actual cardiac AP waveform data that can be displayed as a
real-time ECG-like trace on the 3D visualization.

The cardiac action potential has 5 phases:
  Phase 4: Resting (-85 mV) — IK1 maintains resting potential
  Phase 0: Rapid depolarization (-85 → +40 mV in 1ms) — INa opens
  Phase 1: Early repolarization (+40 → +20 mV) — INa closes, Ito opens briefly
  Phase 2: Plateau (+20 mV for ~200ms) — ICaL inward balanced by IKr outward
  Phase 3: Repolarization (+20 → -85 mV) — IKr/IKs outward, ICaL closes
  Phase 4: Return to rest

Drug effects on the waveform:
  - IKr block (hERG) → Phase 3 prolonged → APD increases → QT prolongation
  - ICaL block → Phase 2 shortened → APD decreases
  - INa block → Phase 0 slower (reduced upstroke velocity)
  - INaL block → Phase 2 shortened (reduced plateau)

This generates waveform points that can be plotted as a real-time trace.
"""
from __future__ import annotations

import math
import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_ap_waveform(
    heart_rate_bpm: float = 72,
    ikr_block_pct: float = 0,
    ical_block_pct: float = 0,
    ina_block_pct: float = 0,
    inal_block_pct: float = 0,
    iks_block_pct: float = 0,
    n_beats: int = 3,
    points_per_beat: int = 200,
) -> dict:
    """
    Generate cardiac action potential waveform with drug effects.

    Returns arrays of time (ms) and voltage (mV) for plotting.
    """
    # Beat duration
    rr_interval_ms = 60000 / heart_rate_bpm  # ms per beat
    apd_base = 300  # Base APD90 in ms (at 72 bpm)

    # Drug effects on APD
    # IKr block prolongs Phase 3 → longer APD
    apd_ikr = ikr_block_pct * 2.0  # ms prolongation per % block
    # ICaL block shortens Phase 2 → shorter APD
    apd_ical = -ical_block_pct * 1.0
    # INaL block shortens Phase 2
    apd_inal = -inal_block_pct * 0.5
    # IKs block prolongs Phase 3 (less than IKr)
    apd_iks = iks_block_pct * 0.5

    apd_modified = apd_base + apd_ikr + apd_ical + apd_inal + apd_iks
    apd_modified = max(150, min(600, apd_modified))

    # Phase durations (ms)
    phase0_dur = max(1, 2 * (1 + ina_block_pct / 100))  # Slower upstroke with INa block
    phase1_dur = 10
    phase2_dur = apd_modified * 0.5  # Plateau
    phase3_dur = apd_modified * 0.3  # Repolarization
    phase4_dur = rr_interval_ms - apd_modified  # Diastole

    # Voltage values (mV)
    v_rest = -85
    v_peak = 40 * (1 - ina_block_pct / 200)  # Reduced peak with INa block
    v_notch = 20  # Phase 1 notch
    v_plateau = 10 + 15 * (1 - ical_block_pct / 100)  # Lower plateau with ICaL block

    # Generate waveform
    all_time = []
    all_voltage = []
    current_time = 0

    for beat in range(n_beats):
        t_beat = np.linspace(0, 1, points_per_beat)

        for i, t in enumerate(t_beat):
            # Normalized time within the beat (0 to 1)
            abs_time = t * rr_interval_ms  # ms within this beat

            if abs_time < phase0_dur:
                # Phase 0: rapid depolarization
                frac = abs_time / phase0_dur
                v = v_rest + (v_peak - v_rest) * frac

            elif abs_time < phase0_dur + phase1_dur:
                # Phase 1: early repolarization (notch)
                frac = (abs_time - phase0_dur) / phase1_dur
                v = v_peak + (v_notch - v_peak) * frac

            elif abs_time < phase0_dur + phase1_dur + phase2_dur:
                # Phase 2: plateau
                frac = (abs_time - phase0_dur - phase1_dur) / phase2_dur
                # Slight downslope during plateau
                v = v_plateau - frac * 5

            elif abs_time < phase0_dur + phase1_dur + phase2_dur + phase3_dur:
                # Phase 3: repolarization
                frac = (abs_time - phase0_dur - phase1_dur - phase2_dur) / phase3_dur
                v_start = v_plateau - 5
                v = v_start + (v_rest - v_start) * (1 - math.exp(-3 * frac))

            else:
                # Phase 4: resting
                v = v_rest

            all_time.append(current_time + abs_time)
            all_voltage.append(v)

        current_time += rr_interval_ms

    # Calculate QT interval
    qt_ms = phase0_dur + phase1_dur + phase2_dur + phase3_dur
    # Bazett's correction: QTc = QT / sqrt(RR in seconds)
    rr_sec = rr_interval_ms / 1000
    qtc_ms = qt_ms / math.sqrt(rr_sec) if rr_sec > 0 else qt_ms

    return {
        "waveform": {
            "time_ms": [round(t, 1) for t in all_time],
            "voltage_mv": [round(v, 1) for v in all_voltage],
            "n_beats": n_beats,
            "n_points": len(all_time),
        },
        "parameters": {
            "heart_rate_bpm": heart_rate_bpm,
            "rr_interval_ms": round(rr_interval_ms, 1),
            "apd90_ms": round(apd_modified, 1),
            "apd90_baseline_ms": apd_base,
            "apd90_change_ms": round(apd_modified - apd_base, 1),
            "qt_ms": round(qt_ms, 1),
            "qtc_ms": round(qtc_ms, 1),
            "upstroke_velocity": round(100 * (1 - ina_block_pct / 100), 1),
            "plateau_voltage_mv": round(v_plateau, 1),
        },
        "drug_effects": {
            "ikr_block": ikr_block_pct,
            "ical_block": ical_block_pct,
            "ina_block": ina_block_pct,
            "inal_block": inal_block_pct,
            "iks_block": iks_block_pct,
        },
        "phases": {
            "phase0_ms": round(phase0_dur, 1),
            "phase1_ms": round(phase1_dur, 1),
            "phase2_ms": round(phase2_dur, 1),
            "phase3_ms": round(phase3_dur, 1),
            "phase4_ms": round(phase4_dur, 1),
        },
        "risk_assessment": {
            "qtc_normal": qtc_ms < 440,
            "qtc_borderline": 440 <= qtc_ms < 500,
            "qtc_prolonged": qtc_ms >= 500,
            "tdp_risk": (
                "HIGH — QTc ≥ 500ms" if qtc_ms >= 500
                else "MODERATE — QTc 470-500ms" if qtc_ms >= 470
                else "BORDERLINE — QTc 440-470ms" if qtc_ms >= 440
                else "NORMAL — QTc < 440ms"
            ),
        },
    }
