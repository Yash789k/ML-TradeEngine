"""Registered sandbox portfolio comparison through the pilot ledger.

Question (fixed in ``data/review-v1/registration.json`` before any result):
does scaling the weekly targets by one minus the HMM turbulence probability
improve the risk-adjusted result of its parent portfolio after costs?

Arms: cash, equal weight (benchmark), trend (the pilot shadow), and the two
regime-scaled versions. Every arm runs through ``runtime_v2.account`` with the
pilot limits and costs, so results are comparable with each other and with the
sealed September accounts.

Two periods, reported separately and never pooled:

* ``history``       2 January 2024 to 25 September 2026. Previously inspected,
                    so exploratory. The only period with inference.
* ``post_release``  29 September 2026 to the latest completed session. Simulated
                    replay on a fresh download. Too short to interpret a Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from review_v1 import registration, regime_report, replay
from review_v1.paths import ADAPTIVE_RELEASE, OUT, SIMULATED, rel, write_json
from review_v1.registration import BLOCK, HISTORY, PERIODS, REPS, SANDBOX_ARMS, SANDBOX_CONTRASTS, SYMBOLS
from review_v1.stats import (
    block_effect, block_indices, deflated_sharpe_ratio, expected_maximum_sharpe, holm, return_report, sharpe_ratio,
)

DIR = OUT / "sandbox"
EXP = ADAPTIVE_RELEASE / "experiment"
CAPITAL = 10000.0
ARM_TEXT = {
    "cash": "Cash (no positions)",
    "equal_weight": "Equal weight, all 22 instruments (benchmark)",
    "trend": "Trend rule (pilot shadow)",
    "trend_regime": "Trend rule × (1 − P(turbulent))",
    "equal_weight_regime": "Equal weight × (1 − P(turbulent))",
}


def run_account(prices: pd.DataFrame, targets: pd.DataFrame, rebalance, capital: float = CAPITAL):
    """Run one arm through the pilot ledger. ``rebalance`` is a boolean per session."""
    from runtime_v2.account import Account, ExecutionSession
    from runtime_v2.risk import limits

    if not prices.index.equals(targets.index) or list(prices.columns) != list(targets.columns):
        raise ValueError("Price and target panels must share sessions and instruments")
    if prices.isna().any().any() or targets.isna().any().any() or (prices <= 0).any().any() or (targets < 0).any().any():
        raise ValueError("Incomplete or invalid replay panel")
    rebalance = np.asarray(rebalance, bool)
    account = Account(list(prices.columns), capital, limits())
    session = ExecutionSession(account)
    rows = []
    for i, date in enumerate(prices.index):
        result = session.on_close(date, prices.iloc[i].to_dict(), targets.iloc[i].to_dict() if rebalance[i] else None)
        rows.append(dict(date=date, **{k: v for k, v in result.items() if k != "positions"}, **{f"quantity_{s}": q for s, q in result["positions"].items()}))
    return pd.DataFrame(rows).set_index("date"), account.events


def daily_returns(history: pd.DataFrame, initial: float = CAPITAL) -> pd.Series:
    return history.equity.pct_change().fillna(history.equity.iloc[0] / initial - 1)


def summarise(history: pd.DataFrame, events: list, initial: float = CAPITAL) -> dict:
    """Ledger metrics (same function as the research-v2 tables) plus Sharpe uncertainty."""
    from research_v2.diagnostics import portfolio_metrics

    ledger = portfolio_metrics(history, events, initial)
    returns = daily_returns(history, initial)
    report = return_report(returns.to_numpy(), block=BLOCK)
    keep = [
        "sessions", "terminal_equity", "net_return", "cagr", "sharpe_zero_rf", "sortino_zero_mar", "max_drawdown",
        "max_drawdown_duration_sessions", "mean_gross_exposure", "max_gross_exposure", "combined_fees_usd",
        "turnover_initial_capital", "fills", "sell_fills", "net_realized_sell_pnl_usd", "sell_fill_win_rate",
        "profit_factor_sell_fills", "terminal_positions", "rejections", "max_reconciliation_error",
    ]
    out = {k: ledger[k] for k in keep}
    out.update(
        pnl_usd=float(history.equity.iloc[-1] - initial),
        orders=int(sum(e["kind"] == "order" for e in events)),
        sharpe_ci_low=report["sharpe_ci_low"], sharpe_ci_high=report["sharpe_ci_high"],
        sharpe_standard_error_iid=report["sharpe_standard_error_iid"],
        probabilistic_sharpe_ratio=report["probabilistic_sharpe_ratio"],
        sharpe_interpretation=report["interpretation"],
        annualised_volatility=float(returns.std(ddof=1) * np.sqrt(252)),
    )
    return out


def events_between(events: list, start, end) -> list:
    return [e for e in events if e.get("asof") and pd.Timestamp(start) <= pd.Timestamp(e["asof"]).tz_localize(None) <= pd.Timestamp(end)]


def sharpe_difference(a, b, block: int = BLOCK, reps: int = REPS) -> dict:
    """Paired date-block bootstrap of the annualised Sharpe difference a − b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Paired daily returns required")
    index = block_indices(len(a), block, reps)

    def ratio(x):
        sd = x.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 1e-15, x.mean(axis=1) / sd * np.sqrt(252), np.nan)

    boot = ratio(a[index]) - ratio(b[index])
    boot = boot[np.isfinite(boot)]
    effect = sharpe_ratio(a) - sharpe_ratio(b)
    low, high = np.quantile(boot, [0.025, 0.975])
    p = (1 + np.count_nonzero(np.abs(boot - effect) >= abs(effect) - 1e-15)) / (len(boot) + 1)
    return dict(effect=float(effect), ci_low=float(low), ci_high=float(high), p_two_sided=float(p), replicates=int(len(boot)), block_length=block)


def spy_reference(prices: pd.DataFrame) -> dict:
    """Buy-and-hold SPY, fully invested, no costs. Reference only: different capital at risk."""
    returns = prices.SPY.pct_change().fillna(0.0)
    equity = (1 + returns).cumprod()
    report = return_report(returns.to_numpy(), block=BLOCK)
    return dict(
        net_return=float(equity.iloc[-1] - 1), sharpe_zero_rf=report["sharpe_zero_rf"], sharpe_ci_low=report["sharpe_ci_low"],
        sharpe_ci_high=report["sharpe_ci_high"], max_drawdown=float((equity / equity.cummax() - 1).min()),
        annualised_volatility=float(returns.std(ddof=1) * np.sqrt(252)),
        note="100% invested and uncosted, so not matched to the 80%-capped, costed arms.",
    )


def curves(histories: dict, p_turbulent: pd.Series) -> pd.DataFrame:
    rows = []
    for arm, history in histories.items():
        peak = np.maximum.accumulate(np.r_[CAPITAL, history.equity.to_numpy()])[1:]
        frame = pd.DataFrame(
            dict(date=regime_report.naive(history.index).strftime("%Y-%m-%d"), arm=arm, equity=history.equity.round(4).to_numpy(),
                 drawdown=(history.equity.to_numpy() / peak - 1).round(6), gross_exposure=history.gross_exposure.round(6).to_numpy(),
                 p_turbulent=p_turbulent.round(6).to_numpy())
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def history_panels():
    prices = pd.read_parquet(EXP / "prices.parquet")
    index = regime_report.naive(prices.index)
    posterior = regime_report.load_posterior("history")
    p_turbulent = posterior.p_turbulent.reindex(index)
    if p_turbulent.isna().any():
        raise ValueError("Regime probability missing on a history session")
    p_turbulent.index = prices.index
    targets = dict(
        cash=pd.DataFrame(0.0, index=prices.index, columns=prices.columns),
        equal_weight=pd.read_parquet(EXP / "equal_weight_targets.parquet"),
        trend=pd.read_parquet(EXP / "trend_targets.parquet"),
    )
    targets["trend_regime"] = targets["trend"].mul(1 - p_turbulent, axis=0)
    targets["equal_weight_regime"] = targets["equal_weight"].mul(1 - p_turbulent, axis=0)
    # Amendment A1: the calendar the sealed targets were built with.
    rebalance = (prices.index.weekday == 4) | (np.arange(len(prices)) == 0)
    return prices, {a: targets[a][list(prices.columns)].astype(float) for a in SANDBOX_ARMS}, rebalance, p_turbulent


def post_release_panels():
    """Fresh prices, replayed trend targets and causal regime probabilities."""
    from runtime_v2.risk import constrain_weights

    result = replay.run()
    frames = result["frames"]
    full = pd.DataFrame({s: frames[s].Close for s in SYMBOLS})
    days = result["days"]
    index = pd.DatetimeIndex([pd.Timestamp(d["session"], tz="UTC") for d in days])
    prices = full.loc[index]
    trend = pd.DataFrame([d["shadow_weights"] for d in days], index=index)[list(prices.columns)]
    rebalance = np.array([d["rebalance"] for d in days], bool)
    returns = full.pct_change(fill_method=None)
    equal = pd.DataFrame(
        [constrain_weights(pd.Series(0.8 / len(SYMBOLS), index=prices.columns), returns.loc[:date]) for date in index], index=index
    )
    posterior = regime_report.load_posterior("recent")
    p_turbulent = posterior.p_turbulent.reindex(regime_report.naive(index))
    if p_turbulent.isna().any():
        raise ValueError("Regime probability missing on a post-release session")
    p_turbulent.index = index
    targets = dict(
        cash=pd.DataFrame(0.0, index=index, columns=prices.columns), equal_weight=equal, trend=trend,
        trend_regime=trend.mul(1 - p_turbulent, axis=0), equal_weight_regime=equal.mul(1 - p_turbulent, axis=0),
    )
    return prices, targets, rebalance, p_turbulent, result


def parity_history(histories: dict) -> dict:
    """The harness must reproduce the sealed September accounts exactly."""
    rows = {}
    for arm in ["trend", "equal_weight", "cash"]:
        sealed = pd.read_parquet(EXP / f"{arm}_account.parquet")
        rows[arm] = dict(
            max_abs_equity_difference_usd=float((sealed.equity - histories[arm].equity).abs().max()),
            max_abs_fee_difference_usd=float((sealed.fees - histories[arm].fees).abs().max()),
            sealed_terminal_equity=float(sealed.equity.iloc[-1]),
        )
    return dict(arms=rows, passed=bool(all(r["max_abs_equity_difference_usd"] < 1e-6 for r in rows.values())),
                note="Same targets, prices, calendar and ledger as the sealed adaptive release.")


def parity_post(histories: dict, result: dict) -> dict:
    """The harness trend arm must equal the shadow account inside ``policy.replay``."""
    replayed = np.array([d["shadow"]["equity"] for d in result["days"]])
    difference = float(np.abs(replayed - histories["trend"].equity.to_numpy()).max())
    return dict(max_abs_equity_difference_usd=difference, passed=bool(difference < 1e-6),
                note="runtime_v2.policy.replay shadow account versus this harness on the same sessions.")


def deflation(returns: dict, sealed_returns: pd.DataFrame) -> dict:
    """Deflated Sharpe ratio using the trials listed in the registration."""
    trial = dict(
        trend=returns["trend"], equal_weight=returns["equal_weight"], trend_regime=returns["trend_regime"],
        equal_weight_regime=returns["equal_weight_regime"], ml=sealed_returns["ml"].to_numpy(), adaptive_hybrid=sealed_returns["adaptive_hybrid"].to_numpy(),
    )
    per_period = {k: sharpe_ratio(v) / np.sqrt(252) for k, v in trial.items()}
    variance = float(np.var(list(per_period.values()), ddof=1))
    threshold = expected_maximum_sharpe(len(trial), variance)
    return dict(
        trials=sorted(trial), n_trials=len(trial), variance_of_per_period_sharpes=variance,
        expected_best_sharpe_with_no_skill_annualised=float(threshold * np.sqrt(252)),
        deflated_sharpe_ratio={k: deflated_sharpe_ratio(v, len(trial), variance) for k, v in trial.items()},
        note=(
            "Probability that the true Sharpe ratio exceeds the best value expected by luck among the counted trials. "
            "Earlier prototype backtests and the block-reset study portfolios are not counted, so these values are upper bounds."
        ),
    )


def contrasts(returns: dict) -> pd.DataFrame:
    rows = []
    for arm, parent in SANDBOX_CONTRASTS:
        sharpe = sharpe_difference(returns[arm], returns[parent])
        mean = block_effect(returns[arm] - returns[parent], BLOCK, REPS)
        rows.append(
            dict(
                arm=arm, parent=parent, sessions=len(returns[arm]), block_length=BLOCK,
                sharpe_arm=sharpe_ratio(returns[arm]), sharpe_parent=sharpe_ratio(returns[parent]),
                sharpe_difference=sharpe["effect"], sharpe_ci_low=sharpe["ci_low"], sharpe_ci_high=sharpe["ci_high"], sharpe_p=sharpe["p_two_sided"],
                annualised_return_difference=mean["effect"] * 252, return_ci_low=mean["ci_low"] * 252, return_ci_high=mean["ci_high"] * 252, return_p=mean["p_two_sided"],
            )
        )
    frame = pd.DataFrame(rows)
    frame["sharpe_holm_p"] = holm(frame.sharpe_p.to_numpy())
    frame["verdict"] = np.where((frame.sharpe_difference > 0) & (frame.sharpe_holm_p < 0.05), "supported on this history", "not established")
    return frame


def versus_benchmark(returns: dict) -> pd.DataFrame:
    """Each invested arm against costed equal weight (descriptive, same ledger)."""
    rows = []
    for arm in ["trend", "trend_regime", "equal_weight_regime"]:
        sharpe = sharpe_difference(returns[arm], returns["equal_weight"])
        mean = block_effect(returns[arm] - returns["equal_weight"], BLOCK, REPS)
        rows.append(
            dict(arm=arm, benchmark="equal_weight", sharpe_difference=sharpe["effect"], sharpe_ci_low=sharpe["ci_low"], sharpe_ci_high=sharpe["ci_high"],
                 annualised_return_difference=mean["effect"] * 252, return_ci_low=mean["ci_low"] * 252, return_ci_high=mean["ci_high"] * 252)
        )
    return pd.DataFrame(rows)


def build() -> dict:
    reg = registration.require()
    DIR.mkdir(parents=True, exist_ok=True)

    # ---- history ---------------------------------------------------------
    prices, targets, rebalance, p_turbulent = history_panels()
    histories, events = {}, {}
    for arm in SANDBOX_ARMS:
        histories[arm], events[arm] = run_account(prices, targets[arm], rebalance)
    returns = {arm: daily_returns(histories[arm]).to_numpy() for arm in SANDBOX_ARMS}
    metric_rows = [dict(period="history", arm=arm, label=ARM_TEXT[arm], **summarise(histories[arm], events[arm])) for arm in SANDBOX_ARMS]
    for period, (start, end) in PERIODS.items():
        for arm in SANDBOX_ARMS:
            history = histories[arm].loc[start:end]
            before = histories[arm].loc[histories[arm].index < history.index[0]].equity
            initial = float(before.iloc[-1]) if len(before) else CAPITAL
            metric_rows.append(dict(period=period, arm=arm, label=ARM_TEXT[arm], **summarise(history, events_between(events[arm], start, end), initial)))
    contrast_frame = contrasts(returns)
    benchmark_frame = versus_benchmark(returns)
    sealed_returns = pd.read_parquet(EXP / "returns.parquet")
    curves(histories, p_turbulent).to_csv(DIR / "curves_history.csv", index=False)

    # ---- post-release (simulated) ---------------------------------------
    post_prices, post_targets, post_rebalance, post_p, result = post_release_panels()
    post_histories, post_events = {}, {}
    for arm in SANDBOX_ARMS:
        post_histories[arm], post_events[arm] = run_account(post_prices, post_targets[arm], post_rebalance)
    for arm in SANDBOX_ARMS:
        metric_rows.append(dict(period="post_release", arm=arm, label=ARM_TEXT[arm], **summarise(post_histories[arm], post_events[arm])))
    curves(post_histories, post_p).to_csv(DIR / "curves_post_release.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(DIR / "metrics.csv", index=False)
    contrast_frame.to_csv(DIR / "contrasts.csv", index=False)
    benchmark_frame.to_csv(DIR / "versus_benchmark.csv", index=False)

    post_index = regime_report.naive(post_prices.index)
    summary = dict(
        registration_sha256=reg["sha256"],
        amendments=[a["id"] for a in reg["amendments"]],
        question=reg["registration"]["sandbox"]["question"],
        decision_rule=reg["registration"]["sandbox"]["decision_rule"],
        arms=ARM_TEXT,
        assumptions=dict(
            capital_usd=CAPITAL,
            data="Yahoo Finance public chart v8 daily bars; dividend/split-adjusted OHLC (current vintage at download)",
            fills="at the next observed close after the decision; fractional units; long only",
            costs="10 bp commission + 5 bp slippage on traded notional, both sides (15 bp per side)",
            limits="80% gross, 10% per instrument, 35% growth group, 12% annualised volatility target, 40% turnover per rebalance, 3% cash buffer, 20% drawdown stop",
            cash_return="zero (no interest on idle cash)",
            sharpe="annualised, zero risk-free rate, 252 sessions",
            benchmark="equal weight of the same 22 instruments through the same ledger and costs; cash; SPY buy-and-hold as an unmatched reference",
        ),
        history=dict(
            status="exploratory: previously inspected",
            first=HISTORY[0], last=HISTORY[1], sessions=int(len(prices)), rebalances=int(rebalance.sum()),
            parity=parity_history(histories),
            spy_reference=spy_reference(prices),
            deflated=deflation(returns, sealed_returns),
            turbulent_share=float((p_turbulent >= 0.5).mean()),
        ),
        post_release=dict(
            status="simulated replay on a fresh download; descriptive only; not an untouched holdout",
            provenance=SIMULATED,
            first=str(post_index[0].date()), last=str(post_index[-1].date()), sessions=int(len(post_prices)), rebalances=int(post_rebalance.sum()),
            snapshot=rel(result["snapshot"]), retrieved_at=result["meta"]["retrieved_at"],
            parity=parity_post(post_histories, result),
            spy_reference=spy_reference(post_prices),
            turbulent_share=float((post_p >= 0.5).mean()),
        ),
    )
    write_json(DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    out = build()
    pd.set_option("display.width", 250)
    m = pd.read_csv(DIR / "metrics.csv")
    cols = ["period", "arm", "sessions", "net_return", "pnl_usd", "sharpe_zero_rf", "sharpe_ci_low", "sharpe_ci_high", "max_drawdown", "fills", "combined_fees_usd", "mean_gross_exposure", "probabilistic_sharpe_ratio"]
    print(m.loc[m.period.isin(["history", "post_release"]), cols].round(4).to_string())
    print(pd.read_csv(DIR / "contrasts.csv").round(4).to_string())
    print(pd.read_csv(DIR / "versus_benchmark.csv").round(4).to_string())
    print(out["history"]["parity"], out["post_release"]["parity"])
    print(out["history"]["deflated"])
    print(out["history"]["spy_reference"], out["post_release"]["spy_reference"])
