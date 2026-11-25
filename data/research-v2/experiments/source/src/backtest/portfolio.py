"""Executable portfolio replay through the local paper event handler."""

import pandas as pd
from src.execution.account import Account, Limits, ExecutionSession


def replay_portfolio(
    prices,
    targets,
    initial_capital=10000.0,
    limits=None,
    state_path=None,
    rebalance_dates=None,
):
    if not prices.index.equals(targets.index) or set(prices.columns) != set(
        targets.columns
    ):
        raise ValueError("Price and target panels require matching coverage")
    if (
        not prices.index.is_monotonic_increasing
        or not prices.index.is_unique
        or prices.isna().any().any()
        or targets.isna().any().any()
    ):
        raise ValueError("Incomplete or unordered replay panel")
    account = Account(
        list(prices.columns), initial_capital, limits or Limits(), state_path
    )
    session = ExecutionSession(account)
    rebalance_dates = set(prices.index if rebalance_dates is None else rebalance_dates)
    rows = []
    for date in prices.index:
        result = session.on_close(
            date,
            prices.loc[date].to_dict(),
            targets.loc[date].to_dict() if date in rebalance_dates else None,
        )
        rows.append(
            dict(
                date=date,
                **{k: v for k, v in result.items() if k != "positions"},
                **{f"quantity_{s}": q for s, q in result["positions"].items()},
            )
        )
    return pd.DataFrame(rows).set_index("date"), account
