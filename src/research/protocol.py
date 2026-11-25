"""Predeclared evidence rules and dependent-return uncertainty."""

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.research.ranker import t_stat_returns


@dataclass(frozen=True)
class Protocol:
    training_end: str = "2023-12-29"
    validation_start: str = "2024-01-02"
    validation_end: str = "2024-12-31"
    evaluation_start: str = "2025-01-02"
    prospective_holdout_start: str = "2026-09-29"
    horizon: int = 5
    threshold: float = 0.01
    commission: float = 0.001
    slippage: float = 0.0005
    risk_free: float = 0.05
    bootstrap_block: int = 10
    bootstrap_samples: int = 1000
    seed: int = 42
    min_closed_trades: int = 20
    significance: float = 0.05

    def as_dict(self):
        return asdict(self)


def evidence(returns, benchmark, family_size, protocol=Protocol()):
    if not returns.index.equals(benchmark.index):
        raise ValueError("Evidence requires matching observation periods")
    active = returns - benchmark
    t = t_stat_returns(active)
    p = float(norm.sf(t)) if active.std() else 1.0
    n = len(active)
    rng = np.random.default_rng(protocol.seed)
    x = active.to_numpy()
    samples = []
    for _ in range(protocol.bootstrap_samples):
        starts = rng.integers(0, n, int(np.ceil(n / protocol.bootstrap_block)))
        idx = ((starts[:, None] + np.arange(protocol.bootstrap_block)) % n).ravel()[:n]
        samples.append(x[idx].mean() * 252)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(
        active_mean_annual=float(active.mean() * 252),
        active_hac_t=t,
        one_sided_p=p,
        bonferroni_p=min(1.0, p * family_size),
        family_size=family_size,
        active_mean_ci_low=float(lo),
        active_mean_ci_high=float(hi),
        positive_evidence=bool(lo > 0 and p * family_size < protocol.significance),
    )
