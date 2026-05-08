"""
Phase 06B — Strategy Ranker

Loads the ZooRunner scorecard and enriches it with:
  - Alpha (Jensen's alpha vs benchmark)
  - Beta  (market exposure)
  - t-statistic on mean daily return (H₀: mean return = 0)
  - Information Ratio (active return / tracking error vs benchmark)
  - Composite Rank Score (weighted multi-metric)
  - Regime breakdown (Sharpe/CAGR per HMM state)

Inputs
------
  scorecard.parquet     : ZooRunner output (indexed by ticker × strategy)
  equity_curve.parquet  : per (ticker, strategy) in data/research/

The benchmark for alpha/beta/IR is buy-and-hold SPY (loaded from parquet cache).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

log = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_RESEARCH_ROOT = _PROJECT_ROOT / "data" / "research"
_PARQUET_ROOT  = _PROJECT_ROOT / "data" / "parquet"

_TRADING_DAYS  = 252


# ---------------------------------------------------------------------------
# Individual extended metrics
# ---------------------------------------------------------------------------

def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    """
    Compute annualised Jensen's alpha and beta via OLS regression.

      r_strategy = α + β·r_benchmark + ε

    Returns
    -------
    (alpha_annual, beta)
    """
    aligned = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")],
        axis=1,
    ).dropna()
    if len(aligned) < 30:
        return 0.0, 1.0

    X = sm.add_constant(aligned["b"])
    model = sm.OLS(aligned["s"], X).fit()
    daily_alpha = float(model.params.get("const", 0.0))
    beta        = float(model.params.get("b", 1.0))
    alpha_ann   = daily_alpha * _TRADING_DAYS
    return round(alpha_ann, 4), round(beta, 4)


def t_stat_returns(daily_returns: pd.Series) -> float:
    """t-statistic testing H₀: mean daily return = 0."""
    clean = daily_returns.dropna().replace([np.inf, -np.inf], 0)
    if len(clean) < 10 or clean.std() == 0:
        return 0.0
    t, _ = stats.ttest_1samp(clean, 0)
    return round(float(t), 4)


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Annualised Information Ratio (active return / tracking error)."""
    aligned = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")],
        axis=1,
    ).dropna()
    if len(aligned) < 10:
        return 0.0
    active  = aligned["s"] - aligned["b"]
    te      = active.std(ddof=1)
    if te == 0:
        return 0.0
    ir = active.mean() / te * np.sqrt(_TRADING_DAYS)
    return round(float(ir), 4)


def composite_score(row: pd.Series) -> float:
    """
    Weighted composite rank score (higher is better).

    Weights:
      Sharpe ratio     0.35
      Calmar ratio     0.25
      CAGR             0.20
      Alpha (annual)   0.15
      t-stat (|t|)     0.05
    """
    sharpe = row.get("sharpe_ratio", 0) or 0
    calmar = row.get("calmar_ratio", 0) or 0
    cagr_v = row.get("cagr", 0) or 0
    alpha  = row.get("alpha", 0) or 0
    tstat  = abs(row.get("t_stat", 0) or 0)

    # Normalise Calmar to same scale as Sharpe (cap at 3)
    calmar_norm = min(calmar, 3.0) / 3.0

    score = (
        0.35 * sharpe
        + 0.25 * calmar_norm
        + 0.20 * cagr_v
        + 0.15 * alpha
        + 0.05 * (tstat / 3.0)
    )
    return round(float(score), 4)


# ---------------------------------------------------------------------------
# Ranker class
# ---------------------------------------------------------------------------

class Ranker:
    """
    Phase 06B — compute extended metrics and rank all strategies.

    Parameters
    ----------
    research_dir : path to data/research/ (default auto-detected)
    parquet_dir  : path to data/parquet/  (for SPY benchmark data)
    """

    def __init__(
        self,
        research_dir: Optional[Path] = None,
        parquet_dir: Optional[Path] = None,
    ) -> None:
        self.research_dir = Path(research_dir) if research_dir else _RESEARCH_ROOT
        self.parquet_dir  = Path(parquet_dir)  if parquet_dir  else _PARQUET_ROOT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, scorecard: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Enrich the scorecard with extended metrics and return ranked DataFrame.

        Parameters
        ----------
        scorecard : pre-loaded scorecard DataFrame (if None, loads from disk)

        Returns
        -------
        ranked_df : DataFrame sorted by composite_score descending
        """
        if scorecard is None:
            sc_path = self.research_dir / "scorecard.parquet"
            if not sc_path.exists():
                raise FileNotFoundError(
                    f"Scorecard not found at {sc_path}. Run ZooRunner first."
                )
            scorecard = pd.read_parquet(sc_path)

        spy_returns = self._load_spy_returns()
        extended_rows: list[dict] = []

        for (ticker, strategy_name), row in scorecard.iterrows():
            strategy_returns = self._load_equity_returns(ticker, strategy_name)

            if strategy_returns is not None and spy_returns is not None:
                alp, bet = alpha_beta(strategy_returns, spy_returns)
                tst      = t_stat_returns(strategy_returns)
                ir       = information_ratio(strategy_returns, spy_returns)
            else:
                alp, bet, tst, ir = 0.0, 1.0, 0.0, 0.0

            extended = row.to_dict()
            extended.update({
                "ticker":   ticker,
                "strategy": strategy_name,
                "alpha":    alp,
                "beta":     bet,
                "t_stat":   tst,
                "info_ratio": ir,
            })
            extended["score"] = composite_score(pd.Series(extended))
            extended_rows.append(extended)

        ranked_df = (
            pd.DataFrame(extended_rows)
            .set_index(["ticker", "strategy"])
            .sort_values("score", ascending=False)
        )

        # Save enriched scorecard
        ranked_path = self.research_dir / "ranked_scorecard.parquet"
        ranked_df.to_parquet(ranked_path)
        log.info("[Ranker] Ranked scorecard saved → %s", ranked_path)

        return ranked_df

    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Load ranked scorecard and return top-N rows."""
        ranked_path = self.research_dir / "ranked_scorecard.parquet"
        if not ranked_path.exists():
            return self.rank().head(n)
        return pd.read_parquet(ranked_path).head(n)

    def print_table(self, df: Optional[pd.DataFrame] = None, n: int = 20) -> None:
        """Pretty-print the top-N ranked strategies."""
        df = df if df is not None else self.top_n(n)
        cols = [
            "sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio",
            "alpha", "beta", "t_stat", "info_ratio", "score",
        ]
        present = [c for c in cols if c in df.columns]
        print(df[present].to_string())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_spy_returns(self) -> Optional[pd.Series]:
        spy_path = self.parquet_dir / "SPY_daily.parquet"
        if not spy_path.exists():
            return None
        df = pd.read_parquet(spy_path)
        close_col = "Adj_Close" if "Adj_Close" in df.columns else "Close"
        return df[close_col].ffill().pct_change().fillna(0).rename("spy")

    def _load_equity_returns(
        self, ticker: str, strategy_name: str
    ) -> Optional[pd.Series]:
        ec_path = self.research_dir / ticker / strategy_name / "equity_curve.parquet"
        if not ec_path.exists():
            return None
        df = pd.read_parquet(ec_path)
        return df["equity"].pct_change().fillna(0).rename("returns")
