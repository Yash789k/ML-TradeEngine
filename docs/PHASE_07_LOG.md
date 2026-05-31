# Phase 07 — Dashboard

**Status:** Complete  
**New files:**
- `dashboard.py`

---

## Objective

Interactive Streamlit app showing model health, equity curve, live signals,
and strategy ranking — the final deliverable before paper deployment.

---

## Pages

| Page | Data source | Key charts |
|------|-------------|------------|
| 🏠 Overview | `data/backtest/{ticker}/backtest_summary.json`, `data/risk/{ticker}/risk_summary.json`, `data/live/signal_log.db` | Key metric cards, equity curve, MC summary, performance targets |
| 📊 Equity Curves | `data/backtest/{ticker}/equity_curves.parquet`, `mc_percentiles.parquet` | Strategy vs buy-and-hold, multi-ticker overlay, return distribution, MC fan |
| 🔄 Rolling Performance | `equity_curves.parquet` | Adjustable rolling Sharpe + drawdown dual-axis, rolling vol |
| ⚡ Live Signals | `data/live/signal_log.db` (SQLite) | Latest signal per ticker cards, signal log table, order log, account equity history |
| 🏆 Strategy Ranking | `data/research/scorecard.parquet` | Filterable ranked table, Sharpe box plot per strategy, CAGR vs Sharpe scatter |
| 🧬 Feature Importance | `data/models/{ticker}/xgb_final.json`, `feature_cols.json` | XGBoost gain bar chart (top 20), OOS accuracy/F1 |
| 📋 Trade Log | `data/backtest/{ticker}/trade_log.parquet` or `data/risk/{ticker}/trade_log.parquet` | P&L bars + cumulative line, win/loss streak, full trade table |

---

## Run

```bash
# Local
streamlit run dashboard.py

# Streamlit Community Cloud
# 1. Push repo to GitHub
# 2. share.streamlit.io → connect repo → set main file to dashboard.py
```

---

## Dependencies

```
streamlit
plotly
pandas
pyarrow
xgboost    # for feature importance
```

---

## Gate to production

> Public URL, all charts functional.

Verify with:
```bash
streamlit run dashboard.py
# Open http://localhost:8501
# Check all 7 pages render without errors
```
