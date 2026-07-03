"""
Phase 06D — Regime-Gated Meta-Strategy

The paper's core contribution: at each bar, allocate capital to the zoo
strategy with the best *historical* Sharpe ratio in the *current* HMM
regime state, using only information available up to that bar.

Walk-forward selection (no look-ahead in the allocator)
-------------------------------------------------------
For ticker T at day t:
  1. Read regime state s = hmm_regime[t-1]  (state known at close of t-1)
  2. Rank all strategies by Sharpe of their returns on days d < t
     where hmm_regime[d] == s  (expanding window, min 60 in-regime obs)
  3. Hold the top-ranked strategy for day t; earn its day-t return
  4. During burn-in (insufficient in-regime history) fall back to the
     equal-weight average of all strategies.

Selection is re-evaluated when the regime changes or every
`rebalance_every` bars, whichever comes first — this keeps turnover
realistic instead of flipping strategy daily.

Portfolio = equal-weight across tickers (daily mean of per-ticker returns).

Known limitation (documented in the paper): the HMM itself was fitted on
the full series in Phase 02, so regime labels carry mild look-ahead.
The *allocator* is strictly walk-forward.

Inputs
------
  data/research/{ticker}/{strategy}/equity_curve.parquet   (Phase 06A)
  data/features/{ticker}_features.parquet  → hmm_regime    (Phase 02)

Outputs
-------
  data/research/meta/meta_returns.parquet      per-ticker + portfolio daily returns
  data/research/meta/meta_allocations.parquet  which strategy held per ticker/day
  data/research/meta/meta_summary.json         metrics vs benchmarks
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_RESEARCH_ROOT = _PROJECT_ROOT / "data" / "research"
_FEATURES_ROOT = _PROJECT_ROOT / "data" / "features"
_META_ROOT     = _RESEARCH_ROOT / "meta"

_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_strategy_returns(ticker: str) -> pd.DataFrame:
    """
    Load daily returns for every zoo strategy of `ticker`.

    Returns a DataFrame indexed by date, one column per strategy name.
    """
    tdir = _RESEARCH_ROOT / ticker
    if not tdir.exists():
        raise FileNotFoundError(f"No zoo results for {ticker} — run research.py zoo first.")

    cols: dict[str, pd.Series] = {}
    for sdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
        eq_path = sdir / "equity_curve.parquet"
        if not eq_path.exists():
            continue
        eq = pd.read_parquet(eq_path)
        series = eq.iloc[:, 0]
        series.index = pd.to_datetime(series.index, utc=True)
        cols[sdir.name] = series.pct_change().fillna(0.0)

    if not cols:
        raise FileNotFoundError(f"No equity curves under {tdir}")
    return pd.DataFrame(cols).sort_index()


def load_regime(ticker: str) -> pd.Series:
    """Load the HMM regime series for `ticker` (values -1/0/1/2)."""
    path = _FEATURES_ROOT / f"{ticker}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No feature cache for {ticker}")
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True)
    if "hmm_regime" not in df.columns:
        raise KeyError(f"{ticker}: features missing 'hmm_regime'")
    return df["hmm_regime"].astype(int)


# ---------------------------------------------------------------------------
# Core allocator
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, rf_daily: float) -> float:
    """Simple annualised Sharpe on a return array (0 if degenerate)."""
    if len(returns) < 2:
        return -np.inf
    sd = returns.std(ddof=1)
    if sd < 1e-12:
        return -np.inf
    return float((returns.mean() - rf_daily) / sd * np.sqrt(_TRADING_DAYS))


class MetaStrategy:
    """
    Regime-gated walk-forward strategy selector for a single ticker.

    Parameters
    ----------
    min_regime_obs  : minimum in-regime observations before trusting the
                      ranking (burn-in fallback: equal-weight all strategies)
    rebalance_every : re-evaluate selection at most every N bars even if
                      the regime hasn't changed (default 21 ≈ monthly)
    risk_free       : annual risk-free rate for Sharpe ranking
    switch_cost     : expected cost per strategy switch. Strategy equity
                      curves already embed their own trade costs; the switch
                      cost covers the forced liquidate-A / enter-B round trip.
                      Since zoo strategies are flat a large share of the time,
                      the expected cost is roughly half a full round trip
                      (default 15 bps).
    confirm_bars    : regime change must persist this many bars before the
                      allocator reacts — filters HMM whipsaw (default 3)
    min_hold        : minimum bars to hold a selection before switching
                      again (default 10)
    """

    def __init__(
        self,
        min_regime_obs:  int   = 60,
        rebalance_every: int   = 21,
        risk_free:       float = 0.05,
        switch_cost:     float = 0.0015,
        confirm_bars:    int   = 3,
        min_hold:        int   = 10,
    ) -> None:
        self.min_regime_obs  = min_regime_obs
        self.rebalance_every = rebalance_every
        self.rf_daily        = (1 + risk_free) ** (1 / _TRADING_DAYS) - 1
        self.switch_cost     = switch_cost
        self.confirm_bars    = confirm_bars
        self.min_hold        = min_hold

    # ------------------------------------------------------------------

    def run_ticker(
        self,
        strat_returns: pd.DataFrame,
        regime: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Walk-forward meta-strategy for one ticker.

        Parameters
        ----------
        strat_returns : DataFrame (date × strategy) of daily returns
        regime        : Series of HMM state per date

        Returns
        -------
        (meta_returns, allocation) — daily return Series and the name of the
        strategy held each day ("EW" during burn-in).
        """
        # Align on common dates
        common = strat_returns.index.intersection(regime.index)
        R = strat_returns.loc[common]
        g = regime.loc[common].values
        n, k = R.shape
        ret_mat = R.values
        strat_names = list(R.columns)

        meta_ret  = np.zeros(n)
        alloc     = np.empty(n, dtype=object)
        current   = None            # currently held strategy index (None = EW)
        since_eval   = 10**9        # bars since last ranking evaluation
        since_switch = 10**9        # bars since last actual switch
        confirmed_state = -1        # regime confirmed after `confirm_bars` stability

        for t in range(n):
            # ── Confirm regime: last `confirm_bars` known states identical ──
            if t >= self.confirm_bars:
                window = g[t - self.confirm_bars:t]
                if (window == window[0]).all() and window[0] >= 0:
                    new_state = int(window[0])
                else:
                    new_state = confirmed_state       # unstable → keep old
            else:
                new_state = -1

            regime_changed  = new_state != confirmed_state and new_state >= 0
            confirmed_state = new_state if new_state >= 0 else confirmed_state

            need_reselect = (
                current is None
                or since_eval >= self.rebalance_every
                or (regime_changed and since_switch >= self.min_hold)
            )

            if need_reselect and t > 0 and confirmed_state >= 0:
                since_eval = 0
                # In-regime historical mask strictly before t
                past_mask = (g[:t] == confirmed_state)
                if past_mask.sum() >= self.min_regime_obs:
                    sharpes = np.array([
                        _sharpe(ret_mat[:t][past_mask, j], self.rf_daily)
                        for j in range(k)
                    ])
                    best = int(np.nanargmax(sharpes))
                    if best != current and since_switch >= self.min_hold:
                        if current is not None:
                            # pay switching cost on transition day
                            meta_ret[t] -= self.switch_cost
                        current      = best
                        since_switch = 0
                # else: keep whatever we had (possibly None → EW)

            if current is None:
                meta_ret[t] += ret_mat[t].mean()      # equal-weight burn-in
                alloc[t]     = "EW"
            else:
                meta_ret[t] += ret_mat[t, current]
                alloc[t]     = strat_names[current]

            since_eval   += 1
            since_switch += 1

        return (
            pd.Series(meta_ret, index=common, name="meta"),
            pd.Series(alloc,    index=common, name="allocation"),
        )

    # ------------------------------------------------------------------

    def run_universe(
        self,
        tickers: list[str],
        save: bool = True,
    ) -> dict:
        """
        Run the meta-strategy on every ticker and build the equal-weight
        portfolio. Returns a results dict; persists artifacts when save=True.
        """
        from src.backtest.metrics import compute_metrics
        from src.research.ranker import alpha_beta, information_ratio, t_stat_returns

        per_ticker_ret:   dict[str, pd.Series] = {}
        per_ticker_alloc: dict[str, pd.Series] = {}

        for ticker in tickers:
            try:
                R = load_strategy_returns(ticker)
                g = load_regime(ticker)
                ret, alloc = self.run_ticker(R, g)
                per_ticker_ret[ticker]   = ret
                per_ticker_alloc[ticker] = alloc
                log.info("  [Meta] %s  Sharpe=%.3f  CAGR=%.1f%%",
                         ticker,
                         _sharpe(ret.values, self.rf_daily),
                         (float((1 + ret).prod()) ** (_TRADING_DAYS / max(len(ret), 1)) - 1) * 100)
            except Exception as exc:
                log.warning("  [Meta] %s failed: %s", ticker, exc)

        if not per_ticker_ret:
            raise RuntimeError("Meta-strategy produced no results for any ticker.")

        # Restrict to dates where every ticker reports — feature caches can be
        # refreshed asynchronously (live engine only refreshes its own universe),
        # leaving trailing days with partial coverage.
        ret_df   = pd.DataFrame(per_ticker_ret).sort_index().dropna()
        alloc_df = pd.DataFrame(per_ticker_alloc).sort_index().loc[ret_df.index]
        portfolio = ret_df.mean(axis=1).rename("portfolio")

        # ── Benchchmarks ────────────────────────────────────────────────
        spy_bh: Optional[pd.Series] = None
        try:
            spy_eq = pd.read_parquet(_PROJECT_ROOT / "data" / "backtest" / "SPY" / "equity_curves.parquet")
            spy_bh = spy_eq["buy_hold"].pct_change().fillna(0.0)
            spy_bh.index = pd.to_datetime(spy_bh.index, utc=True)
            spy_bh = spy_bh.reindex(portfolio.index).fillna(0.0)
        except Exception as exc:
            log.warning("Could not load SPY benchmark: %s", exc)

        # Best single zoo strategy portfolio (same strategy on all tickers, EW)
        best_single: dict[str, pd.Series] = {}
        strat_names = load_strategy_returns(tickers[0]).columns
        for s in strat_names:
            legs = []
            for t in tickers:
                try:
                    r = load_strategy_returns(t)
                    if s in r.columns:
                        legs.append(r[s])
                except Exception:
                    pass
            if legs:
                best_single[s] = pd.concat(legs, axis=1).mean(axis=1).reindex(portfolio.index).fillna(0.0)

        # ── Metrics ─────────────────────────────────────────────────────
        results: dict = {"tickers": list(per_ticker_ret.keys())}

        m = compute_metrics(portfolio)
        m["t_stat"] = t_stat_returns(portfolio)
        if spy_bh is not None:
            alpha, beta = alpha_beta(portfolio, spy_bh)
            m["alpha"], m["beta"] = alpha, beta
            m["info_ratio"] = information_ratio(portfolio, spy_bh)
        results["meta_portfolio"] = m

        singles = {}
        for s, r in best_single.items():
            sm = compute_metrics(r)
            sm["t_stat"] = t_stat_returns(r)
            singles[s] = sm
        results["single_strategy_portfolios"] = singles

        if spy_bh is not None:
            results["spy_buy_hold"] = compute_metrics(spy_bh)

        # Allocation distribution (how often each strategy is selected)
        flat = alloc_df.values.ravel()
        flat = flat[pd.notna(flat)]
        counts = pd.Series(flat).value_counts(normalize=True)
        results["allocation_pct"] = {k: round(float(v), 4) for k, v in counts.items()}

        # ── Persist ─────────────────────────────────────────────────────
        if save:
            _META_ROOT.mkdir(parents=True, exist_ok=True)
            out = ret_df.copy()
            out["portfolio"] = portfolio
            out.to_parquet(_META_ROOT / "meta_returns.parquet")
            alloc_df.to_parquet(_META_ROOT / "meta_allocations.parquet")
            (_META_ROOT / "meta_summary.json").write_text(
                json.dumps(results, indent=2, default=str)
            )
            log.info("[Meta] artifacts saved → %s", _META_ROOT)

        return results
