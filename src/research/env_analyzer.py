"""
Phase 06C — Environment Characterisation

Four analyses that characterise the algorithmic trading environment and
validate (or challenge) the research paper's claims:

  1. regime_breakdown      — per-HMM-state Sharpe/return for every strategy
  2. cost_sensitivity      — edge survival across commission/slippage levels
  3. signal_decay          — rolling 90-day Sharpe to detect strategy decay
  4. factor_attribution    — Fama-French 3-factor OLS decomposition

All results are saved to data/research/env/ as Parquet files.

Design
------
Analyses 1, 3, and 4 load from saved artifacts (equity_curve.parquet)
so no re-simulation is needed.  Analysis 2 (cost sensitivity) re-runs
the simulator at each cost level — strategies are fast (< 1s) so this
is acceptable on a subset of tickers.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
from scipy import stats

from src.backtest.metrics import compute_metrics
from src.features.regime import add_hmm_regime

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_RESEARCH_ROOT = _PROJECT_ROOT / "data" / "research"
_PARQUET_ROOT  = _PROJECT_ROOT / "data" / "parquet"
_ENV_ROOT      = _RESEARCH_ROOT / "env"
_TRADING_DAYS  = 252

_STRATEGY_NAMES = [
    "Momentum_12_1", "Mean_Reversion", "EMA_Crossover", "Turtle_Breakout",
    "Pairs_StatArb", "Carry_Proxy", "Vol_Breakout", "Alpha_Trends",
]

_REGIME_LABELS = {0: "bear", 1: "ranging", 2: "bull"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regime_sharpe(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    """Sharpe over a (non-contiguous) set of days."""
    clean = returns.dropna().replace([np.inf, -np.inf], 0)
    if len(clean) < 5 or clean.std(ddof=1) == 0:
        return 0.0
    excess = clean - risk_free_daily
    return float(excess.mean() / clean.std(ddof=1) * np.sqrt(_TRADING_DAYS))


def _regime_ann_return(returns: pd.Series) -> float:
    """Mean daily return × 252 (proxy CAGR for non-contiguous periods)."""
    clean = returns.dropna().replace([np.inf, -np.inf], 0)
    return float(clean.mean() * _TRADING_DAYS) if len(clean) > 0 else 0.0


def _load_equity_returns(
    research_dir: Path, ticker: str, strategy_name: str
) -> Optional[pd.Series]:
    ec_path = research_dir / ticker / strategy_name / "equity_curve.parquet"
    if not ec_path.exists():
        return None
    df = pd.read_parquet(ec_path)
    return df["equity"].pct_change().fillna(0).rename("returns")


def _compute_ticker_regime(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Fit HMM on ticker OHLCV and return daily regime Series (0/1/2).

    Tries full covariance first; falls back to diagonal if the covariance
    matrix is near-singular (common on synthetic / low-variance data).
    """
    close      = ohlcv["Adj_Close"].ffill() if "Adj_Close" in ohlcv.columns else ohlcv["Close"].ffill()
    log_ret    = np.log(close / close.shift(1))
    real_vol   = log_ret.rolling(21).std()
    hmm_input  = pd.DataFrame(
        {"log_return": log_ret, "realized_vol_21": real_vol}
    ).dropna()

    if len(hmm_input) < 60:
        return pd.Series(dtype=int)

    # Try full covariance first, fall back to diagonal if singular
    for cov_type in ("full", "diag"):
        try:
            hmm_df, _ = add_hmm_regime(
                hmm_input.copy(),
                n_states=3,
                random_state=42,
                covariance_type=cov_type,
            )
            return hmm_df["hmm_regime"].reindex(ohlcv.index).ffill()
        except Exception as exc:
            log.debug("HMM (%s cov) failed: %s — trying next", cov_type, exc)

    log.warning("HMM fit failed for all covariance types on this ticker")
    return pd.Series(dtype=int)


# ---------------------------------------------------------------------------
# EnvAnalyzer
# ---------------------------------------------------------------------------

class EnvAnalyzer:
    """
    Phase 06C — Environment Characterisation.

    Parameters
    ----------
    research_dir : path to data/research/  (default: auto-detected)
    parquet_dir  : path to data/parquet/   (default: auto-detected)
    """

    def __init__(
        self,
        research_dir: Optional[Path] = None,
        parquet_dir: Optional[Path] = None,
    ) -> None:
        self.research_dir = Path(research_dir) if research_dir else _RESEARCH_ROOT
        self.parquet_dir  = Path(parquet_dir)  if parquet_dir  else _PARQUET_ROOT
        self.env_dir      = self.research_dir / "env"
        self.env_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # 1. Regime Breakdown
    # ======================================================================

    def regime_breakdown(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        strategy_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute per-HMM-state performance for every (ticker, strategy) pair.

        For each regime state (bear=0, ranging=1, bull=2), computes:
          - n_days      : number of trading days in that state
          - pct_days    : fraction of total days
          - ann_return  : mean daily return × 252
          - sharpe      : annualised Sharpe over regime days

        Returns a DataFrame indexed by (ticker, strategy, regime).
        """
        if strategy_names is None:
            strategy_names = _STRATEGY_NAMES

        rows: list[dict] = []
        tickers = list(ohlcv_dict.keys())
        total   = len(tickers)

        for idx, ticker in enumerate(tickers):
            log.info("  [regime_breakdown] %s (%d/%d) …", ticker, idx + 1, total)
            ohlcv   = ohlcv_dict[ticker]
            regimes = _compute_ticker_regime(ohlcv)

            if regimes.empty:
                log.warning("    No regime data for %s — skipping", ticker)
                continue

            for strategy_name in strategy_names:
                strategy_returns = _load_equity_returns(
                    self.research_dir, ticker, strategy_name
                )
                if strategy_returns is None:
                    continue

                # Align on common index
                merged = pd.concat(
                    [strategy_returns.rename("r"), regimes.rename("regime")],
                    axis=1,
                ).dropna()

                if len(merged) < 30:
                    continue

                total_days = len(merged)
                for regime_val, regime_label in _REGIME_LABELS.items():
                    mask   = merged["regime"] == regime_val
                    subset = merged.loc[mask, "r"]
                    n      = len(subset)
                    if n < 5:
                        continue
                    rows.append({
                        "ticker":      ticker,
                        "strategy":    strategy_name,
                        "regime":      regime_label,
                        "n_days":      n,
                        "pct_days":    round(n / total_days, 3),
                        "ann_return":  round(_regime_ann_return(subset), 4),
                        "sharpe":      round(_regime_sharpe(subset), 3),
                    })

        if not rows:
            log.warning("[EnvAnalyzer] regime_breakdown: no data collected — "
                        "check that HMM converged and equity_curve.parquet files exist.")
            df = pd.DataFrame(
                columns=["ticker", "strategy", "regime",
                         "n_days", "pct_days", "ann_return", "sharpe"]
            ).set_index(["ticker", "strategy", "regime"])
        else:
            df = pd.DataFrame(rows).set_index(["ticker", "strategy", "regime"])

        out = self.env_dir / "regime_breakdown.parquet"
        df.to_parquet(out)
        log.info("[EnvAnalyzer] regime_breakdown saved → %s", out)
        return df

    # ======================================================================
    # 2. Cost Sensitivity
    # ======================================================================

    def cost_sensitivity(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        strategies: list,
        spy_df: Optional[pd.DataFrame] = None,
        macro: Optional[pd.DataFrame] = None,
        commission_levels: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """
        Re-run strategies at multiple commission levels to quantify break-even.

        commission_levels : list of per-side commission fractions to test.
          Default: [0, 0.0005, 0.001, 0.002, 0.005, 0.010]
          (0 bps → 5 bps → 10 bps → 20 bps → 50 bps → 100 bps)

        Returns DataFrame indexed by (ticker, strategy, commission) with
        columns sharpe, cagr, n_trades.
        """
        if commission_levels is None:
            commission_levels = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.010]

        from src.research.strategies.carry_proxy import CarryProxyStrategy
        from src.research.strategies.pairs_arb import PairsArbStrategy

        rows:  list[dict] = []
        total  = len(ohlcv_dict) * len(strategies) * len(commission_levels)
        done   = 0

        for ticker, ohlcv in ohlcv_dict.items():
            for strategy in strategies:
                pair_df = spy_df if isinstance(strategy, PairsArbStrategy) else None
                mac     = macro  if isinstance(strategy, CarryProxyStrategy) else None

                for comm in commission_levels:
                    try:
                        out = strategy.run(
                            ohlcv,
                            macro           = mac,
                            pair_df         = pair_df,
                            commission      = comm,
                            slippage        = 0.0,    # isolate commission effect
                        )
                        m = out["metrics"]
                    except Exception as exc:
                        log.warning("cost_sens failed %s × %s @ %.4f: %s",
                                    ticker, strategy.name, comm, exc)
                        m = {}

                    rows.append({
                        "ticker":     ticker,
                        "strategy":   strategy.name,
                        "commission": comm,
                        "bps":        round(comm * 10_000),
                        "sharpe":     round(m.get("sharpe_ratio", np.nan), 3),
                        "cagr":       round(m.get("cagr", np.nan), 4),
                        "n_trades":   m.get("n_trades", np.nan),
                    })
                    done += 1

        df = pd.DataFrame(rows).set_index(["ticker", "strategy", "commission"])

        out = self.env_dir / "cost_sensitivity.parquet"
        df.to_parquet(out)
        log.info("[EnvAnalyzer] cost_sensitivity saved → %s", out)
        return df

    # ======================================================================
    # 3. Signal Decay (Rolling Sharpe)
    # ======================================================================

    def signal_decay(
        self,
        strategy_names: Optional[list[str]] = None,
        window: int = 90,
    ) -> pd.DataFrame:
        """
        Compute a rolling `window`-day Sharpe for every (ticker, strategy) pair.

        Saves the full rolling time series and a summary of:
          - trend slope (positive = improving, negative = decaying)
          - end_vs_start: final rolling Sharpe minus first rolling Sharpe

        Loaded from saved equity_curve.parquet artifacts — no re-simulation.
        """
        if strategy_names is None:
            strategy_names = _STRATEGY_NAMES

        _SQRT = np.sqrt(_TRADING_DAYS)
        series_rows: list[dict] = []
        summary_rows: list[dict] = []

        research_path = self.research_dir
        for ticker_dir in sorted(research_path.iterdir()):
            if not ticker_dir.is_dir() or ticker_dir.name == "env":
                continue
            ticker = ticker_dir.name

            for strategy_name in strategy_names:
                ec_path = ticker_dir / strategy_name / "equity_curve.parquet"
                if not ec_path.exists():
                    continue

                returns = (
                    pd.read_parquet(ec_path)["equity"]
                    .pct_change()
                    .fillna(0)
                )
                if len(returns) < window + 10:
                    continue

                # Rolling Sharpe
                roll_mean = returns.rolling(window).mean()
                roll_std  = returns.rolling(window).std(ddof=1)
                roll_sh   = (roll_mean / roll_std.replace(0, np.nan)) * _SQRT
                roll_sh   = roll_sh.dropna()

                # Store time series
                for date, val in roll_sh.items():
                    series_rows.append({
                        "ticker":   ticker,
                        "strategy": strategy_name,
                        "date":     date,
                        "rolling_sharpe": round(float(val), 4),
                    })

                # Summary statistics
                if len(roll_sh) < 10:
                    continue
                x     = np.arange(len(roll_sh))
                slope, _, _, _, _ = stats.linregress(x, roll_sh.values)
                summary_rows.append({
                    "ticker":       ticker,
                    "strategy":     strategy_name,
                    "slope":        round(float(slope) * _TRADING_DAYS, 4),  # annualised slope
                    "mean_sharpe":  round(float(roll_sh.mean()), 3),
                    "end_sharpe":   round(float(roll_sh.iloc[-1]), 3),
                    "start_sharpe": round(float(roll_sh.iloc[0]), 3),
                    "end_vs_start": round(float(roll_sh.iloc[-1] - roll_sh.iloc[0]), 3),
                    "pct_positive": round(float((roll_sh > 0).mean()), 3),
                    "is_decaying":  bool(slope < 0 and roll_sh.iloc[-1] < roll_sh.mean()),
                })

        ts_df  = pd.DataFrame(series_rows)
        sum_df = pd.DataFrame(summary_rows).set_index(["ticker", "strategy"])

        if not ts_df.empty:
            ts_df.to_parquet(self.env_dir / "signal_decay_series.parquet")
        if not sum_df.empty:
            sum_df.to_parquet(self.env_dir / "signal_decay_summary.parquet")

        log.info("[EnvAnalyzer] signal_decay saved → %s", self.env_dir)
        return sum_df

    # ======================================================================
    # 4. Factor Attribution (Fama-French 3-Factor)
    # ======================================================================

    def factor_attribution(
        self,
        strategy_names: Optional[list[str]] = None,
        start: str = "2016-01-01",
    ) -> pd.DataFrame:
        """
        OLS regression of each strategy's returns on Fama-French 3 factors.

          r_strategy - RF = α + β_mkt·(Mkt-RF) + β_smb·SMB + β_hml·HML + ε

        Metrics per (ticker, strategy):
          alpha_ann   : annualised intercept (true alpha after factor exposure)
          alpha_tstat : t-statistic on the intercept (H₀: α = 0)
          beta_mkt    : market beta
          beta_smb    : size factor loading
          beta_hml    : value factor loading
          r2          : OLS R-squared (fraction of return variance explained by factors)

        Falls back to proxy factors (SPY, IWM, XLF, XLK from cache) if the
        Fama-French data server is unavailable.
        """
        if strategy_names is None:
            strategy_names = _STRATEGY_NAMES

        ff3 = self._load_ff3(start)
        if ff3 is None:
            log.warning("[EnvAnalyzer] FF3 download failed — using proxy factors")
            ff3 = self._proxy_ff3(start)
        if ff3 is None:
            log.error("[EnvAnalyzer] No factor data available — skipping attribution")
            return pd.DataFrame()

        rows: list[dict] = []

        for ticker_dir in sorted(self.research_dir.iterdir()):
            if not ticker_dir.is_dir() or ticker_dir.name == "env":
                continue
            ticker = ticker_dir.name

            for strategy_name in strategy_names:
                returns = _load_equity_returns(
                    self.research_dir, ticker, strategy_name
                )
                if returns is None or len(returns) < 60:
                    continue

                # Align with FF3 factors
                returns.index = pd.to_datetime(returns.index).tz_localize(None)
                ff3.index     = pd.to_datetime(ff3.index).tz_localize(None)

                merged = pd.concat(
                    [returns.rename("r"), ff3],
                    axis=1,
                ).dropna()

                if len(merged) < 60:
                    continue

                # Excess return
                y = merged["r"] - merged["RF"]
                X = sm.add_constant(merged[["Mkt-RF", "SMB", "HML"]])

                try:
                    model  = sm.OLS(y, X).fit()
                    alpha_d = float(model.params.get("const", 0))
                    rows.append({
                        "ticker":       ticker,
                        "strategy":     strategy_name,
                        "alpha_ann":    round(alpha_d * _TRADING_DAYS, 4),
                        "alpha_tstat":  round(float(model.tvalues.get("const", 0)), 3),
                        "beta_mkt":     round(float(model.params.get("Mkt-RF", 0)), 4),
                        "beta_smb":     round(float(model.params.get("SMB", 0)), 4),
                        "beta_hml":     round(float(model.params.get("HML", 0)), 4),
                        "r2":           round(float(model.rsquared), 4),
                        "n_obs":        int(model.nobs),
                    })
                except Exception as exc:
                    log.warning("OLS failed %s × %s: %s", ticker, strategy_name, exc)

        df = pd.DataFrame(rows).set_index(["ticker", "strategy"])

        out = self.env_dir / "factor_attribution.parquet"
        df.to_parquet(out)
        log.info("[EnvAnalyzer] factor_attribution saved → %s", out)
        return df

    # ======================================================================
    # Run all
    # ======================================================================

    def run_all(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        strategies: list,
        spy_df: Optional[pd.DataFrame] = None,
        macro: Optional[pd.DataFrame] = None,
        cost_tickers: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Run all four analyses.

        cost_tickers : subset of tickers for cost sensitivity (default: all).
        """
        results: dict[str, pd.DataFrame] = {}

        log.info("── Phase 06C: 1/4 Regime Breakdown ─────────────────────────")
        results["regime_breakdown"] = self.regime_breakdown(ohlcv_dict)

        log.info("── Phase 06C: 2/4 Cost Sensitivity ─────────────────────────")
        cost_dict = (
            {t: ohlcv_dict[t] for t in cost_tickers if t in ohlcv_dict}
            if cost_tickers else ohlcv_dict
        )
        results["cost_sensitivity"] = self.cost_sensitivity(
            cost_dict, strategies, spy_df=spy_df, macro=macro
        )

        log.info("── Phase 06C: 3/4 Signal Decay ──────────────────────────────")
        results["signal_decay"] = self.signal_decay()

        log.info("── Phase 06C: 4/4 Factor Attribution ────────────────────────")
        results["factor_attribution"] = self.factor_attribution()

        return results

    # ======================================================================
    # Reporting helpers
    # ======================================================================

    def print_regime_summary(self, df: Optional[pd.DataFrame] = None) -> None:
        if df is None:
            df = pd.read_parquet(self.env_dir / "regime_breakdown.parquet")

        print("\n── Regime Breakdown: Average Sharpe by (Strategy, Regime) ─────")
        if df.empty:
            print("  (no data — HMM may not have converged on the loaded tickers)")
            return

        pivot = (
            df.reset_index()
            .groupby(["strategy", "regime"])["sharpe"]
            .mean()
            .unstack("regime")
            .reindex(columns=["bear", "ranging", "bull"])
        )
        for col in ["bear", "ranging", "bull"]:
            if col not in pivot.columns:
                pivot[col] = np.nan
        pivot["bull_minus_bear"] = (pivot["bull"] - pivot["bear"]).round(3)
        print(pivot.round(3).sort_values("bull", ascending=False).to_string())

    def print_cost_summary(self, df: Optional[pd.DataFrame] = None) -> None:
        if df is None:
            df = pd.read_parquet(self.env_dir / "cost_sensitivity.parquet")

        print("\n── Cost Sensitivity: Mean Sharpe by (Strategy, Commission) ────")
        pivot = (
            df.groupby(["strategy", "bps"])["sharpe"]
            .mean()
            .unstack("bps")
        )
        print(pivot.round(3).to_string())

    def print_decay_summary(self, df: Optional[pd.DataFrame] = None, n: int = 15) -> None:
        if df is None:
            df = pd.read_parquet(self.env_dir / "signal_decay_summary.parquet")

        print(f"\n── Signal Decay: Top-{n} Strategies by Stability ──────────────")
        display = df.sort_values("mean_sharpe", ascending=False).head(n)
        print(display[["mean_sharpe", "end_sharpe", "slope", "pct_positive", "is_decaying"]].to_string())

    def print_factor_summary(self, df: Optional[pd.DataFrame] = None, n: int = 15) -> None:
        if df is None:
            df = pd.read_parquet(self.env_dir / "factor_attribution.parquet")

        print(f"\n── Factor Attribution: Top-{n} by Annualised Alpha ────────────")
        display = df.sort_values("alpha_ann", ascending=False).head(n)
        print(display[["alpha_ann", "alpha_tstat", "beta_mkt", "beta_smb", "beta_hml", "r2"]].to_string())

    # ======================================================================
    # Private: FF3 loader
    # ======================================================================

    def _load_ff3(self, start: str) -> Optional[pd.DataFrame]:
        """Download Fama-French 3 factors from Kenneth French library."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = web.DataReader(
                    "F-F_Research_Data_Factors_daily", "famafrench", start=start
                )
            df = raw[0] / 100.0   # convert from percent to decimal
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:
            log.warning("FF3 download error: %s", exc)
            return None

    def _proxy_ff3(self, start: str) -> Optional[pd.DataFrame]:
        """
        Construct approximate FF3 factors from cached parquet files.

          Mkt-RF ≈ SPY daily return
          SMB    ≈ IWM return − SPY return  (small-cap minus large-cap)
          HML    ≈ XLF return − XLK return  (value[financials] minus growth[tech])
          RF     ≈ 0 (negligible at daily scale)
        """
        needed = {"SPY": "SPY_daily", "IWM": "IWM_daily",
                  "XLF": "XLF_daily", "XLK": "XLK_daily"}
        frames: dict[str, pd.Series] = {}
        for name, slug in needed.items():
            p = self.parquet_dir / f"{slug}.parquet"
            if not p.exists():
                return None
            df   = pd.read_parquet(p)
            col  = "Adj_Close" if "Adj_Close" in df.columns else "Close"
            frames[name] = df[col].ffill().pct_change().fillna(0)

        spy = frames["SPY"]
        proxy = pd.DataFrame({
            "Mkt-RF": spy,
            "SMB":    frames["IWM"] - spy,
            "HML":    frames["XLF"] - frames["XLK"],
            "RF":     0.0,
        }).loc[start:]
        return proxy
