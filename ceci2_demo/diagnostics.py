"""Secondary diagnostics computed from the same forecasts and capital path."""

import numpy as np
import pandas as pd

from .evaluate import replay


def diagnostics(predictions, ledger, output):
    portfolio = ledger.set_index(["decision", "model"]).sort_index()
    market_state = predictions.groupby("decision").regime.first()
    portfolio = portfolio.join(market_state, on="decision").reset_index()

    def risk(r):
        return pd.Series(
            {
                "sessions": len(r),
                "annualized_volatility": r.std() * np.sqrt(252),
                "mean_daily_return": r.mean(),
                "conditional_sharpe": r.mean() / r.std() * np.sqrt(252)
                if r.std()
                else np.nan,
                "expected_shortfall_95": -r.nsmallest(
                    max(1, int(np.ceil(0.05 * len(r))))
                ).mean(),
            }
        )

    portfolio.groupby(["model", "regime"]).net_return.apply(risk).unstack().to_csv(
        output / "regimes.csv"
    )
    costs = []
    for name in ("HMM gate", "Uniform vote", "Equal weight", "SPY buy and hold"):
        is_model = name in ("HMM gate", "Uniform vote")
        group = predictions[predictions.model == (name if is_model else "HMM gate")]
        for bps in (0, 5, 10, 20):
            run = replay(group, name, cost=bps / 10000)
            costs.append(
                {
                    "model": name,
                    "bps_each_side": bps,
                    "net_return": run.equity.iloc[-1] / 10000 - 1,
                    "fees_usd": run.fees.sum(),
                }
            )
    pd.DataFrame(costs).to_csv(output / "costs.csv", index=False)
