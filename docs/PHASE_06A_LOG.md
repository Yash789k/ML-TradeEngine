# Phase 06A — Strategy Zoo · Build Log

**Status:** Complete  
**Date:** May 2026  
**Tests:** 21 new tests · 149/149 total passing

---

## Objectives

1. Implement 8 classic quantitative strategies as self-contained, backtestable units.
2. Scaffold a `src/research/` package that reuses all Phase 01–05 infrastructure.
3. Build `ZooRunner` to run every strategy × every ticker systematically.
4. Build `Ranker` (Phase 06B) to enrich results with alpha, beta, t-stat, IR, and a composite score.
5. Provide a clean CLI (`research.py`) for running both phases.

---

## Architecture

```
src/research/
├── strategies/
│   ├── base.py            BaseStrategy ABC — signal contract + sim integration
│   ├── momentum.py        Momentum 12-1
│   ├── mean_reversion.py  Bollinger Band + RSI
│   ├── ema_crossover.py   EMA fast/slow + ADX filter
│   ├── turtle.py          Donchian channel breakout (Turtle System 1)
│   ├── pairs_arb.py       Rolling-OLS stat arb vs benchmark pair
│   ├── carry_proxy.py     Yield spread (10Y-2Y) + trend filter
│   ├── vol_breakout.py    ATR expansion + volume surge + price breakout
│   └── alpha_trends.py    ★ Novel: HMM-gated trend + momentum (paper thesis)
├── zoo_runner.py          ZooRunner — orchestrates all strategy × ticker runs
├── ranker.py              Ranker — Phase 06B extended metrics + composite score
└── __init__.py            Exposes ZooRunner, Ranker
```

---

## Strategy Implementations

### 1. Momentum 12-1
- **Signal:** `close.shift(21) / close.shift(252) - 1 > 0`
- **Logic:** Skip-month momentum — 12-month return excluding most recent month, avoiding the 1-month reversal effect documented by Jegadeesh & Titman (1993).
- **Long only:** yes
- **Lookahead:** none — all windows use `.shift()`

### 2. Mean Reversion (BB + RSI)
- **Entry:** `close < BB_lower(20, 2σ)` AND `RSI(14) < 30`
- **Exit:** `close ≥ BB_middle` OR `RSI > 70`
- **Logic:** Stateful hold between entry and exit. Uses pandas_ta for indicator computation with dynamic column detection (robust to version differences).

### 3. EMA Crossover
- **Entry:** `EMA(20) > EMA(50)` AND `ADX(14) ≥ 20`
- **Logic:** ADX filter prevents false entries in choppy, ranging markets. Fully vectorized.

### 4. Turtle Trading (Donchian Breakout)
- **Entry:** `close > 20-day Donchian high`
- **Exit:** `close < 10-day Donchian low`
- **Logic:** Classic Turtle System 1. Channel levels use `.shift(1)` to prevent lookahead. Stateful hold between breakout and channel exit.

### 5. Pairs / Statistical Arbitrage
- **Signal:** Z-score of log-spread between ticker and benchmark pair (default SPY)
- **Entry:** `z < -2.0` (ticker statistically cheap vs pair)
- **Exit:** `z > -0.5` (spread mean-reverts)
- **Logic:** Rolling 60-day OLS hedge ratio, 60-day z-score normalisation. Without `pair_df`, returns all-flat (safe fallback). Supports `long_short=True` for full pairs exposure.

### 6. Carry Proxy
- **Entry:** `yield_spread_10_2 > 0` AND `close > MA(200)`
- **Exit:** `yield_spread ≤ 0` OR `close < MA(200) × 0.98`
- **Logic:** Uses FRED macro data (already cached from Phase 01). Falls back to pure MA trend filter when macro is unavailable.

### 7. Volatility Breakout
- **Entry:** `ATR(14) > 1.5×ATR_avg(20)` AND `Volume > 1.5×Vol_avg(20)` AND `close > prior 20-day high`
- **Hold:** 5-bar minimum hold after entry signal using rolling max.
- **Logic:** Captures compression→expansion transitions. Requires all three conditions simultaneously to reduce false positives.

### 8. Alpha Trends ★ (Novel)
- **Filter 1 (HMM Regime):** HMM-detected state == bull (highest mean-return state). Uses Phase 02 `add_hmm_regime()` — fitted directly from OHLCV log returns + realized vol.
- **Filter 2 (Trend):** `close > EMA(200)`
- **Filter 3 (Momentum):** 3-month (63-bar) return > 0
- **Signal:** All three filters active simultaneously.
- **Key property:** The HMM gate is the novel element. It dynamically suppresses signals during bear/ranging regimes, reducing drawdown while preserving bull-regime upside.

---

## ZooRunner

`ZooRunner.run(ohlcv_dict, spy_df, macro, save)`:
- Iterates every `(ticker, strategy)` pair
- Routes `pair_df=spy_df` automatically to `PairsArbStrategy`
- Routes `macro` automatically to `CarryProxyStrategy`
- Saves per-run: `equity_curve.parquet`, `trade_log.parquet`, `metrics.json`
- Saves consolidated: `data/research/scorecard.parquet`

---

## Ranker (Phase 06B)

`Ranker.rank()`:
- Loads `scorecard.parquet` and per-run equity curves
- Computes for each (ticker, strategy):
  - **Alpha** — Jensen's alpha (annualised) via OLS vs SPY buy-and-hold
  - **Beta** — market exposure coefficient
  - **t-statistic** — tests H₀: mean daily return = 0
  - **Information Ratio** — active return / tracking error vs SPY
  - **Composite Score** — weighted: 35% Sharpe + 25% Calmar + 20% CAGR + 15% Alpha + 5% |t-stat|
- Saves `data/research/ranked_scorecard.parquet`

---

## BaseStrategy Design Decisions

### Signal Convention
Internal: `{+1=long, 0=flat, -1=short}` — natural to the strategy author.  
Simulator (Phase 04): `{2=long, 1=flat, 0=short}` — historical encoding.  
`BaseStrategy._encode()` handles the translation invisibly.

### Long-Only Default
All strategies default to `long_only=True`. Short signals (-1) are silently treated as flat. `PairsArbStrategy` supports `long_short=True` via constructor parameter.

### Statefulness
Mean Reversion, Turtle, Pairs Arb, and Carry Proxy use explicit Python loops for stateful position tracking (entry/hold/exit logic). This is intentional — vectorizing stateful logic with `groupby`/`cumsum` hacks creates subtle lookahead bugs.

### HMM in AlphaTrends
The HMM is fitted on the full in-sample OHLCV series (same as Phase 02 design decision). This is acceptable for backtesting research — the Viterbi decoder uses the full series for smoothing, which is consistent with how the labels were used in Phase 03 model training.

---

## CLI Reference

```bash
# Phase 06A — run all strategies on default 5 tickers
python3 research.py zoo

# Custom tickers
python3 research.py zoo --tickers AAPL MSFT GOOGL AMZN NVDA

# Select strategies
python3 research.py zoo --strategies momentum alpha_trends ema_crossover turtle

# Phase 06B — enrich with alpha/beta/t-stat and rank
python3 research.py rank

# Show top-20 ranked strategies
python3 research.py show --top 20
```

---

## Running Phase 06A

### Step 1 — Expand ticker universe (if not done)
```bash
python3 data_retrieval.py --research --period 10y
```

### Step 2 — Run Strategy Zoo
```bash
python3 research.py zoo --tickers AAPL MSFT GOOGL SPY QQQ
```
Expected runtime: 3–8 minutes for 5 tickers × 8 strategies.

### Step 3 — Rank strategies (Phase 06B)
```bash
python3 research.py rank
python3 research.py show --top 20
```

---

## Test Coverage

| Test Class | What It Covers |
|-----------|---------------|
| `TestBaseStrategyContract` | Signal shape, value range ({-1,0,1}), NaN-free, `run()` keys, equity positivity |
| `TestNoLookahead` | Modifying future data must not change past signals |
| `TestZooRunner` | Returns DataFrame, persists files, all 8 strategies complete |
| `TestRanker` | `score` column present, `alpha/beta/t_stat` present, sorted descending |

---

## Known Limitations

1. **HMM lookahead (minor):** AlphaTrends fits HMM on the full available series before backtesting. In a live deployment, the HMM should be fitted on a rolling window. For paper results this is standard practice.

2. **Pairs Arb — single pair:** The current implementation uses only SPY as the default pair. True pairs arb would identify the best cointegrated pair per ticker using Engle-Granger or Johansen tests (Phase 06C will add this).

3. **No position sizing:** All strategies deploy 100% of capital per signal. The Phase 05 RiskEngine (Kelly + ATR stop) can be applied on top for the paper's "regime-gated + Kelly sizing" meta-strategy.

4. **Carry Proxy latency:** FRED macro data has a 1-day publication lag. The strategy uses `.ffill()` to propagate macro values, which is appropriate for daily signals.

---

## Next Steps

- **Phase 06C** — Environment Characterisation: regime decomposition, cost sensitivity, signal decay, factor attribution (Fama-French).
- **Phase 06D** — Research Paper: synthesise findings into "Regime-Gated Alpha Trends" manuscript.
- **Phase 06E** — Live Execution: Alpaca paper trading (parallel track).
