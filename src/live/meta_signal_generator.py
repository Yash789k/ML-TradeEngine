"""Experimental causal meta-strategy signal generation for local simulation.

Research/accounting repairs invalidate earlier performance claims. Selection is
based on past virtual strategy outcomes; shared execution turns current signals
into constrained target holdings. The EW warm-up fallback is cash in this path.
No broker performance, profitability or prospective validation is established.
"""

from __future__ import annotations

import logging
from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.live.signal_generator import LiveSignal

log = logging.getLogger(__name__)

_ATR_MULTIPLIER_DEFAULT = 2.0   # matches Phase 05 default


# ---------------------------------------------------------------------------
# Core: fetch + compute the strategy-returns matrix used by the allocator
# ---------------------------------------------------------------------------

def _fetch_market_data(ticker: str, force_refresh: bool) -> tuple:
    """Fetch OHLCV + SPY + macro, mirroring SignalGenerator's data path."""
    from src.data.loader import DataLoader

    loader = DataLoader(max_age_hours=0.5 if force_refresh else 12.0)
    ohlcv  = loader.load_equity(ticker, force_refresh=force_refresh)

    spy_df = macro = None
    try:
        spy_df = loader.load_equity("SPY", force_refresh=force_refresh)
    except Exception as exc:
        log.debug("Could not load SPY for %s: %s", ticker, exc)
    try:
        macro = loader.load_macro(force_refresh=force_refresh)
    except Exception as exc:
        log.debug("Could not load macro for %s: %s", ticker, exc)

    return ohlcv, spy_df, macro


def compute_strategy_returns(
    ohlcv: pd.DataFrame,
    spy_df: Optional[pd.DataFrame],
    macro: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Run every zoo strategy's full-history signal → daily-return series.

    Returns a DataFrame (date × strategy_name) of daily returns, identical
    in spirit to src.research.meta_strategy.load_strategy_returns() but
    computed live instead of read from cached backtest parquet.
    """
    from src.backtest.simulator import run_simulation
    from src.research.strategies import ALL_STRATEGIES
    from src.research.strategies.carry_proxy import CarryProxyStrategy
    from src.research.strategies.pairs_arb import PairsArbStrategy

    cols: dict[str, pd.Series] = {}
    for strategy in ALL_STRATEGIES:
        pair_df = spy_df if isinstance(strategy, PairsArbStrategy) else None
        mac     = macro  if isinstance(strategy, CarryProxyStrategy) else None
        try:
            signal_df = strategy.build_signal_df(ohlcv, macro=mac, pair_df=pair_df)
            result    = run_simulation(
                signal_df, mode="long_only" if strategy.long_only else "long_short"
            )
            cols[strategy.name] = result.daily_returns
        except Exception as exc:
            log.warning("Strategy %s failed on live data: %s", strategy.name, exc)

    if not cols:
        raise RuntimeError("No zoo strategy produced returns — cannot allocate.")
    return pd.DataFrame(cols).sort_index()


def _strategy_by_name(name: str):
    from src.research.strategies import ALL_STRATEGIES
    for s in ALL_STRATEGIES:
        if s.name == name:
            return s
    return None


def _today_direction(strategy, ohlcv: pd.DataFrame, spy_df, macro) -> int:
    """Raw signal from `strategy` on the latest bar, encoded {2,1,0}."""
    from src.research.strategies.carry_proxy import CarryProxyStrategy
    from src.research.strategies.pairs_arb import PairsArbStrategy

    pair_df = spy_df if isinstance(strategy, PairsArbStrategy) else None
    mac     = macro  if isinstance(strategy, CarryProxyStrategy) else None
    sig     = strategy.generate_signals(ohlcv, macro=mac, pair_df=pair_df)
    raw     = int(sig.fillna(0).iloc[-1])

    if strategy.long_only:
        return 2 if raw == 1 else 1          # -1 collapses to flat, same as _encode
    return 2 if raw == 1 else (0 if raw == -1 else 1)


# ---------------------------------------------------------------------------
# MetaSignalGenerator
# ---------------------------------------------------------------------------

class MetaSignalGenerator:
    """
    Generate a daily live signal from the regime-gated meta-strategy.

    Parameters
    ----------
    atr_multiplier : stop-loss distance in units of ATR-14 (default 2.0;
                     overridden per-bar when policy_engine is supplied)
    policy_engine  : optional src.risk.policy_engine.PolicyEngine for a
                     dynamic (regime/vol-adaptive) stop distance
    min_regime_obs, rebalance_every, confirm_bars, min_hold, switch_cost :
                     forwarded to MetaStrategy — keep these identical to the
                     Phase 06D backtest config so live selection matches the
                     validated research, unless deliberately re-tuning.
    """

    def __init__(
        self,
        atr_multiplier:  float = _ATR_MULTIPLIER_DEFAULT,
        policy_engine=None,
        min_regime_obs:  int   = 60,
        rebalance_every: int   = 21,
        confirm_bars:    int   = 3,
        min_hold:        int   = 10,
        switch_cost:     float = 0.0015,
    ) -> None:
        self.atr_multiplier  = atr_multiplier
        self.policy_engine   = policy_engine
        self._meta_kwargs = dict(
            min_regime_obs=min_regime_obs, rebalance_every=rebalance_every,
            confirm_bars=confirm_bars, min_hold=min_hold, switch_cost=switch_cost,
        )

    # ------------------------------------------------------------------

    def generate(self, ticker: str, force_refresh: bool = True) -> LiveSignal:
        from src.features.pipeline import FeatureEngineer
        from src.research.meta_strategy import MetaStrategy

        log.info("[MetaSignalGenerator] %s — fetching live data …", ticker)
        ohlcv, spy_df, macro = _fetch_market_data(ticker, force_refresh)

        fe = FeatureEngineer(max_age_hours=0.5 if force_refresh else 12.0)
        feat_df = fe.compute_features(
            df=ohlcv, asset_slug=ticker, spy_df=spy_df, macro_df=macro, force_refresh=force_refresh,
        )

        strat_returns = compute_strategy_returns(ohlcv, spy_df, macro)
        regime = (
            feat_df["hmm_regime"]
            .reindex(strat_returns.index).ffill().fillna(-1).astype(int)
        )

        meta = MetaStrategy(**self._meta_kwargs)
        _, alloc = meta.run_ticker(strat_returns, regime)
        selected = str(alloc.iloc[-1])

        if selected == "EW":
            # Burn-in fallback — see module docstring "Known simplification".
            direction = 1
            log.info("[MetaSignalGenerator] %s → EW burn-in — no live position taken", ticker)
        else:
            strategy = _strategy_by_name(selected)
            if strategy is None:
                log.warning("[MetaSignalGenerator] %s: selected strategy %r not found — FLAT",
                            ticker, selected)
                direction = 1
            else:
                direction = _today_direction(strategy, ohlcv, spy_df, macro)

        # ── Stop distance: dynamic policy or static default ──────────────
        atr_multiplier = self.atr_multiplier
        size_cap, eligible = 1.0, True
        if self.policy_engine is not None:
            policy = self.policy_engine.decide_latest(feat_df)
            atr_multiplier, size_cap, eligible = policy.atr_multiplier, policy.size_cap, policy.eligible

        last_row   = feat_df.dropna(subset=["atr_14"]).iloc[-1]
        atr_val    = float(last_row.get("atr_14", 0.0))
        last_close = float(feat_df["Close"].iloc[-1])
        stop_loss  = last_close - atr_multiplier * atr_val
        signal_date = feat_df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)

        signal = LiveSignal(size_cap=size_cap, eligible=eligible,
            ticker      = ticker,
            date        = signal_date,
            direction   = direction,
            confidence  = 1.0 if selected != "EW" else 0.0,   # not a probability here
            p_up        = 1.0 if direction == 2 else 0.0,
            p_flat      = 1.0 if direction == 1 else 0.0,
            p_down      = 1.0 if direction == 0 else 0.0,
            # Equal-weight sizing target, consistent with the Phase 06D
            # backtest (EW portfolio across tickers) — NOT a Kelly fraction.
            # LiveEngine._compute_qty divides this further by len(tickers)
            # when strategy_mode="meta".
            kelly_frac  = 1.0 if direction == 2 else 0.0,
            atr         = round(atr_val,    4),
            stop_loss   = round(stop_loss,  4),
            close       = round(last_close, 4),
            strategy_mode     = "meta",
            selected_strategy = selected,
        )

        log.info(
            "[MetaSignalGenerator] %s → %s  selected=%s  close=%.2f  stop=%.2f",
            ticker, signal.label, selected, last_close, stop_loss,
        )
        return signal

    def generate_batch(
        self, tickers: list[str], force_refresh: bool = True,
    ) -> dict[str, LiveSignal]:
        """Generate signals for multiple tickers. Skips tickers that fail."""
        results: dict[str, LiveSignal] = {}
        self.errors = {}
        for ticker in tickers:
            try:
                results[ticker] = self.generate(ticker, force_refresh=force_refresh)
            except Exception as exc:
                log.error("[MetaSignalGenerator] %s failed: %s", ticker, exc)
                self.errors[ticker] = str(exc)
        return results
