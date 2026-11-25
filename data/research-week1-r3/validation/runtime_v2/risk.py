"""Fixed portfolio policy for both rolling research and local daily decisions."""

import numpy as np
import pandas as pd
from runtime_v2.account import Limits
from runtime_v2.model import features

GROWTH = {
    s: "growth"
    for s in ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "QQQ", "TSLA", "XLK"]
}
CANDIDATES = ("trend", "ml", "adaptive_hybrid")


def limits(cost_multiplier=1):
    return Limits(
        max_gross=0.8,
        max_position=0.1,
        max_turnover=0.4,
        commission=0.001 * cost_multiplier,
        slippage=0.0005 * cost_multiplier,
        groups=GROWTH,
        group_limits={"growth": 0.35},
    )


def constrain_weights(weights, history):
    w = weights.clip(lower=0, upper=0.1).astype(float)
    g = [s for s in w.index if s in GROWTH]
    if g and w[g].sum() > 0.35:
        w.loc[g] *= 0.35 / w[g].sum()
    if w.sum() > 0.8:
        w *= 0.8 / w.sum()
    cov = (
        history.tail(63).cov().reindex(index=w.index, columns=w.index).to_numpy() * 252
    )
    if not np.isfinite(cov).all():
        raise ValueError("Insufficient finite history for risk calibration")
    cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov))
    vol = float(np.sqrt(max(0, w.to_numpy() @ cov @ w.to_numpy())))
    if vol > 0.12:
        w *= 0.12 / vol
    return w


