"""Validation metrics — pure functions.

Standard fold-error / within-n-fold metrics used by all four
validation tiers. The functions are intentionally small and do
not depend on any engine or cohort types, so they can be unit
tested in isolation.

FDA/EMA PBPK qualification uses the 2-fold band as the de facto
acceptance criterion: a prediction is "within 2-fold" if the
ratio predicted/observed lies in [0.5, 2.0]. GMFE (geometric
mean fold error) is the standard aggregate metric for a cohort.

Reference:
  Abduljalil K et al. CPT Pharmacometrics Syst Pharmacol 2014; 3:e121
  Sager JE et al. Drug Metab Dispos 2015; 43:1823-1837
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence


def fold_error(predicted: Optional[float], observed: Optional[float]) -> float:
    """Return predicted / observed.

    Returns NaN when either side is None, non-positive, or when
    observed is zero. A fold error of 1.0 means perfect prediction;
    2.0 means the prediction is double the observation; 0.5 means
    half.
    """
    if predicted is None or observed is None:
        return float("nan")
    try:
        p = float(predicted)
        o = float(observed)
    except (TypeError, ValueError):
        return float("nan")
    if o <= 0 or p <= 0:
        return float("nan")
    return p / o


def within_n_fold(
    predicted: Optional[float],
    observed: Optional[float],
    n: float = 2.0,
) -> bool:
    """True if predicted/observed lies in [1/n, n]."""
    fe = fold_error(predicted, observed)
    if math.isnan(fe):
        return False
    return (1.0 / n) <= fe <= n


def absolute_fold_error(
    predicted: Optional[float], observed: Optional[float]
) -> float:
    """Return max(pred/obs, obs/pred) — always ≥ 1.0.

    Useful for aggregation because it collapses 0.25× and 4× into
    the same "4-fold error" magnitude.
    """
    fe = fold_error(predicted, observed)
    if math.isnan(fe) or fe <= 0:
        return float("nan")
    return max(fe, 1.0 / fe)


def percent_error(
    predicted: Optional[float], observed: Optional[float]
) -> float:
    """Return 100 × (predicted − observed) / observed. NaN if invalid."""
    if predicted is None or observed is None:
        return float("nan")
    try:
        o = float(observed)
        if o == 0:
            return float("nan")
        return 100.0 * (float(predicted) - o) / o
    except (TypeError, ValueError):
        return float("nan")


def within_pct(
    predicted: Optional[float],
    observed: Optional[float],
    pct: float = 20.0,
) -> bool:
    """True if |percent_error| ≤ pct."""
    pe = percent_error(predicted, observed)
    if math.isnan(pe):
        return False
    return abs(pe) <= pct


def geometric_mean_fold_error(
    predicted: Sequence[Optional[float]],
    observed: Sequence[Optional[float]],
) -> float:
    """GMFE = 10 ** mean(|log10(pred/obs)|).

    A GMFE of 1.5 means the average prediction is off by a factor
    of 1.5 (either over or under). A GMFE of 1.0 is perfect.
    Invalid entries (NaN, non-positive) are skipped.
    """
    if len(predicted) != len(observed):
        raise ValueError(
            f"predicted and observed must match length "
            f"({len(predicted)} vs {len(observed)})"
        )
    abs_logs = []
    for p, o in zip(predicted, observed):
        fe = fold_error(p, o)
        if math.isnan(fe) or fe <= 0:
            continue
        abs_logs.append(abs(math.log10(fe)))
    if not abs_logs:
        return float("nan")
    return 10.0 ** (sum(abs_logs) / len(abs_logs))


def average_fold_error(
    predicted: Sequence[Optional[float]],
    observed: Sequence[Optional[float]],
) -> float:
    """AFE = 10 ** mean(log10(pred/obs)) — signed aggregate.

    AFE > 1 means the cohort is systematically over-predicted.
    AFE < 1 means it is systematically under-predicted.
    Complementary to GMFE, which reports magnitude only.
    """
    if len(predicted) != len(observed):
        raise ValueError("predicted/observed length mismatch")
    logs = []
    for p, o in zip(predicted, observed):
        fe = fold_error(p, o)
        if math.isnan(fe) or fe <= 0:
            continue
        logs.append(math.log10(fe))
    if not logs:
        return float("nan")
    return 10.0 ** (sum(logs) / len(logs))


def percent_within_fold(
    predicted: Sequence[Optional[float]],
    observed: Sequence[Optional[float]],
    n: float = 2.0,
) -> float:
    """Return percentage (0-100) of cases within n-fold."""
    if len(predicted) != len(observed) or not predicted:
        return 0.0
    passed = sum(
        1
        for p, o in zip(predicted, observed)
        if within_n_fold(p, o, n)
    )
    return 100.0 * passed / len(predicted)


def rmse_log10(
    predicted: Sequence[Optional[float]],
    observed: Sequence[Optional[float]],
) -> float:
    """Root-mean-square error in log10 space.

    Less sensitive to outliers than GMFE when there are a few
    failures; still scale-invariant.
    """
    if len(predicted) != len(observed):
        raise ValueError("predicted/observed length mismatch")
    sq = []
    for p, o in zip(predicted, observed):
        fe = fold_error(p, o)
        if math.isnan(fe) or fe <= 0:
            continue
        sq.append(math.log10(fe) ** 2)
    if not sq:
        return float("nan")
    return math.sqrt(sum(sq) / len(sq))


def summarise(
    predicted: Sequence[Optional[float]],
    observed: Sequence[Optional[float]],
) -> dict:
    """Return a dict of all standard aggregate metrics for a cohort."""
    valid_pairs = [
        (p, o)
        for p, o in zip(predicted, observed)
        if not math.isnan(fold_error(p, o))
    ]
    return {
        "n_total": len(predicted),
        "n_valid": len(valid_pairs),
        "pct_within_2_fold": percent_within_fold(predicted, observed, 2.0),
        "pct_within_3_fold": percent_within_fold(predicted, observed, 3.0),
        "gmfe": geometric_mean_fold_error(predicted, observed),
        "afe": average_fold_error(predicted, observed),
        "rmse_log10": rmse_log10(predicted, observed),
    }
