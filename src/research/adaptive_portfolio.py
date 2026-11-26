"""Fixed portfolio policy for both rolling research and local daily decisions."""

import numpy as np
import pandas as pd
from src.execution.account import Limits
from src.models.adaptive import features

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


def target_panels(data, predictions):
    symbols = sorted(data)
    index = predictions[symbols[0]].index
    if any(not predictions[s].index.equals(index) for s in symbols):
        raise ValueError("Predictions must cover the same sessions")
    full_prices = pd.DataFrame({s: data[s].Close for s in symbols})
    rets = full_prices.pct_change(fill_method=None)
    trend = pd.DataFrame(
        {
            s: (
                (features(data[s], data["SPY"])["average_200"] > 0)
                & (features(data[s], data["SPY"])["momentum_12_1"] > 0)
            ).astype(float)
            for s in symbols
        }
    ).loc[index]
    ml = pd.DataFrame(
        {s: (predictions[s].expected_return > 0.003).astype(float) for s in symbols}
    )
    gate = pd.DataFrame({s: predictions[s].gate.astype(float) for s in symbols})
    raw = dict(
        trend=trend,
        ml=ml,
        adaptive_hybrid=(1 - 0.5 * gate) * trend + 0.5 * gate * ml,
        equal_weight=pd.DataFrame(1.0, index=index, columns=symbols),
        cash=pd.DataFrame(0.0, index=index, columns=symbols),
    )
    panels = {}
    for name, intent in raw.items():
        target = pd.DataFrame(index=index, columns=symbols, dtype=float)
        w = pd.Series(0.0, index=symbols)
        for i, date in enumerate(index):
            if i == 0 or date.weekday() == 4:
                w = constrain_weights(
                    intent.loc[date] * (0.8 / len(symbols)), rets.loc[:date]
                )
            target.loc[date] = w
        panels[name] = target
    return full_prices.loc[index], panels
