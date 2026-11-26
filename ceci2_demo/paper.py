"""Inspect the historical paper account at a dated simulated opening."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import START


def inspect(study, asof):
    orders = pd.read_csv(
        study / "dynamic_orders.csv.gz", parse_dates=["signal_date", "execution_date"]
    )
    ledger = pd.read_csv(study / "portfolio.csv.gz", parse_dates=["decision", "mark"])
    ledger = ledger[ledger.model == "HMM gate dynamic"]
    asof = pd.Timestamp(asof)
    if asof < ledger.decision.min() or asof > ledger.mark.max():
        raise ValueError("Choose a session within the historical study")
    prior = orders[orders.execution_date <= asof]
    holdings = prior.groupby("symbol").units.sum()
    cash = START - (prior.units * prior.price + prior.fee).sum()
    print(f"Simulated opening on {asof.date()}\nCash after fills: ${cash:,.2f}")
    print(holdings[holdings.abs() > 1e-7].rename("adjusted units").to_string())
    print(prior[prior.execution_date == asof].to_string(index=False))
    if asof == ledger.mark.max() and not np.isclose(cash, ledger.equity.iloc[-1]):
        raise ValueError("Final cash differs from the account ledger")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asof")
    parser.add_argument(
        "--study", type=Path, default=Path("data/ceci2-demo/five-session")
    )
    args = parser.parse_args()
    inspect(args.study, args.asof)
