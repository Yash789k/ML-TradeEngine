"""Classification and return statistics used by the review.

Pure functions with explicit inputs. Definitions match the verified research-v2
tables where an equivalent exists (macro F1 over the three labels, balanced
accuracy over classes that occur, natural-log loss, multiclass Brier score as the
sum of squared errors over classes, Sharpe ratio with a zero risk-free rate and
252 sessions per year). The date-block bootstrap and Holm adjustment are imported
from ``research_v2.diagnostics`` rather than reimplemented.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import kurtosis, norm, skew

from research_v2.diagnostics import block_effect, block_indices, holm  # noqa: F401 (re-exported)

CLASSES = ["DOWN", "FLAT", "UP"]
EULER_GAMMA = 0.5772156649015329


def _validate(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if p.ndim != 2 or p.shape != (len(y), 3) or not len(y):
        raise ValueError("Probabilities must be (n, 3) with one row per label")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Invalid probabilities")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Unmatured or unknown labels cannot be scored")
    return y, p


def classification_report(y, p) -> dict:
    """Accuracy, balanced accuracy, macro F1, log loss, Brier, per-class report, confusion."""
    y, p = _validate(y, p)
    predicted = p.argmax(axis=1)
    confusion = np.zeros((3, 3), int)
    np.add.at(confusion, (y, predicted), 1)
    support, predicted_support, hits = confusion.sum(axis=1), confusion.sum(axis=0), np.diag(confusion)
    precision = np.divide(hits, predicted_support, out=np.zeros(3), where=predicted_support > 0)
    recall = np.divide(hits, support, out=np.zeros(3), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(3), where=denominator > 0)
    return dict(
        n=int(len(y)),
        accuracy=float(np.mean(predicted == y)),
        balanced_accuracy=float(recall[support > 0].mean()),
        macro_f1=float(f1.mean()),
        log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        brier=float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
        classes=[
            dict(
                class_name=CLASSES[c],
                precision=float(precision[c]),
                recall=float(recall[c]),
                f1=float(f1[c]),
                support=int(support[c]),
                predicted=int(predicted_support[c]),
            )
            for c in range(3)
        ],
        confusion=confusion.tolist(),
    )


def row_log_loss(y, p) -> np.ndarray:
    y, p = _validate(y, p)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


# --------------------------------------------------------------------------
# Sharpe ratio and its uncertainty
# --------------------------------------------------------------------------

def sharpe_ratio(returns, periods: int = 252) -> float:
    """Annualised mean over sample standard deviation, zero risk-free rate."""
    r = np.asarray(returns, float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("At least two finite returns required")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 1e-15 else float("nan")


def sharpe_block_interval(returns, block: int = 10, reps: int = 2000, periods: int = 252):
    """Percentile interval for the annualised Sharpe ratio from circular date blocks."""
    r = np.asarray(returns, float)
    samples = r[block_indices(len(r), block, reps)]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(sd > 1e-15, samples.mean(axis=1) / sd * math.sqrt(periods), np.nan)
    values = values[np.isfinite(values)]
    if len(values) < reps // 2:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def probabilistic_sharpe_ratio(returns, benchmark_per_period: float = 0.0) -> float:
    """P(true Sharpe > benchmark) allowing for skewness and kurtosis.

    Bailey and López de Prado (2012). Sharpe ratios are per period, not annualised.
    """
    r = np.asarray(returns, float)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or not sd > 1e-15:
        return float("nan")
    sr = r.mean() / sd
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    variance_term = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    if variance_term <= 0:
        return float("nan")
    return float(norm.cdf((sr - benchmark_per_period) * math.sqrt(n - 1) / math.sqrt(variance_term)))


def expected_maximum_sharpe(n_trials: int, variance_of_trial_sharpes: float) -> float:
    """Expected best per-period Sharpe among ``n_trials`` strategies with no true skill."""
    if n_trials < 2 or not variance_of_trial_sharpes >= 0:
        raise ValueError("At least two trials and a non-negative variance required")
    return float(
        math.sqrt(variance_of_trial_sharpes)
        * (
            (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e))
        )
    )


def deflated_sharpe_ratio(returns, n_trials: int, variance_of_trial_sharpes: float) -> float:
    """Probabilistic Sharpe ratio against the expected best of ``n_trials`` skill-less trials.

    Bailey and López de Prado (2014). ``variance_of_trial_sharpes`` is the variance
    of the per-period Sharpe ratios across the trials that were actually run.
    """
    return probabilistic_sharpe_ratio(returns, expected_maximum_sharpe(n_trials, variance_of_trial_sharpes))


def return_report(returns, periods: int = 252, block: int = 10) -> dict:
    """Sharpe ratio with the uncertainty needed to decide whether it means anything."""
    r = np.asarray(returns, float)
    n = len(r)
    sharpe = sharpe_ratio(r, periods)
    per_period = sharpe / math.sqrt(periods) if np.isfinite(sharpe) else float("nan")
    low, high = sharpe_block_interval(r, block=min(block, max(2, n // 4)), periods=periods) if n >= 20 else (float("nan"), float("nan"))
    # Large-sample standard error for independent returns (Lo, 2002), annualised.
    standard_error = (
        math.sqrt((1 + 0.5 * per_period**2) / n) * math.sqrt(periods) if np.isfinite(per_period) else float("nan")
    )
    return dict(
        sessions=int(n),
        years=float(n / periods),
        total_return=float(np.prod(1 + r) - 1),
        mean_daily_return=float(r.mean()),
        daily_volatility=float(r.std(ddof=1)),
        sharpe_zero_rf=sharpe,
        sharpe_ci_low=low,
        sharpe_ci_high=high,
        sharpe_standard_error_iid=standard_error,
        probabilistic_sharpe_ratio=probabilistic_sharpe_ratio(r),
        interpretable=bool(np.isfinite(low) and np.isfinite(high) and n >= periods and (low > 0 or high < 0)),
        interpretation=sharpe_interpretation(n, sharpe, low, high, periods),
    )


def sharpe_interpretation(n, sharpe, low, high, periods: int = 252) -> str:
    if not np.isfinite(sharpe):
        return "Undefined: the return series has no variation."
    if n < 63:
        return (
            f"{n} sessions is far too few to interpret a Sharpe ratio: one quarter of daily returns "
            "cannot separate skill from noise."
        )
    if not (np.isfinite(low) and np.isfinite(high)):
        return "No interval could be estimated; treat the point value as descriptive only."
    if low <= 0 <= high:
        return (
            f"The 95% interval [{low:.2f}, {high:.2f}] includes zero, so this sample does not "
            "establish a positive risk-adjusted return."
        )
    direction = "above" if low > 0 else "below"
    caveat = "" if n >= periods else " The sample is shorter than one year."
    return f"The 95% interval [{low:.2f}, {high:.2f}] lies {direction} zero in this sample.{caveat}"
