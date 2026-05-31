# Phase 06E — Live Signal Engine

**Status:** Complete  
**Runs in parallel with:** 06A–06D  
**New files:**
- `src/live/__init__.py`
- `src/live/signal_generator.py`
- `src/live/broker.py`
- `src/live/logger.py`
- `src/live/alerts.py`
- `src/live/engine.py`
- `live.py` (CLI)
- `.github/workflows/daily_signals.yml`
- `tests/test_live.py`

---

## Objective

Run the trained ensemble model daily at market close, push signals to Alpaca
paper trading as bracket orders, log everything to SQLite, and fire alerts
via Slack webhook and/or email.

This phase generates live validation data for the Phase 06D research paper —
out-of-sample performance after the training cutoff date.

---

## Architecture

```
[market close]
      │
      ▼
SignalGenerator.generate_batch(tickers)
  ├── DataLoader.load_equity()        ← fresh OHLCV (force_refresh=True)
  ├── FeatureEngineer.compute_features() ← Phase 02 pipeline
  ├── XGB + LGBM + LSTM final models  ← last row inference only
  ├── EnsembleClassifier              ← soft-vote probabilities
  ├── confidence filter (≥ 0.38)
  ├── Kelly fraction (half-Kelly, Phase 05 calibration)
  └── ATR stop level (entry - 2×ATR)
      │
      ▼
LiveEngine.run()
  ├── Risk gates (in order):
  │   1. direction == FLAT → skip
  │   2. kelly_frac == 0   → skip (no statistical edge)
  │   3. circuit_breaker   → skip entries if equity down >15% from peak
  │   4. already_ordered_today → idempotency guard
  │   5. portfolio_heat > 80%  → skip (max deployed capital guard)
  │
  ├── AlpacaBroker.submit_bracket_buy()
  │   entry  : market order at open
  │   stop   : entry − 2×ATR
  │   target : entry + 3×ATR   (risk:reward ≈ 1.5:1)
  │
  ├── SignalLogger (SQLite at data/live/signal_log.db)
  │   tables: signals, orders, equity, circuit_breaker
  │
  └── Alerts (Slack webhook + SMTP email)
      signal_fired | circuit_breaker_tripped | order_error | daily_summary
```

---

## Risk Controls

| Gate | Condition | Action |
|------|-----------|--------|
| FLAT signal | direction == 1 | Skip — no order |
| No edge | kelly_frac ≤ 0 | Skip — model structurally wrong |
| Circuit breaker | equity < 85% of peak | Skip new entries |
| Idempotency | order already placed today | Skip |
| Portfolio heat | total deployed ≥ 80% equity | Skip |

---

## CLI

```bash
# Full run (submits Alpaca paper orders)
python3 live.py run

# Dry run — signals only, no orders
python3 live.py run --dry-run

# Specific tickers
python3 live.py run --tickers AAPL MSFT NVDA --dry-run

# Inspect the log
python3 live.py status --n 20
python3 live.py orders --n 10
python3 live.py equity --n 30
python3 live.py positions
```

---

## GitHub Actions (`.github/workflows/daily_signals.yml`)

- **Schedule:** `30 21 * * 1-5` — 21:30 UTC Mon–Fri (≈ 30 min post-close)
- **Manual trigger:** via `workflow_dispatch` with optional tickers + dry-run flag
- **Cache:** Parquet data store + trained models persisted between runs via
  `actions/cache` — avoids re-fetching 5 years of OHLCV daily
- **Artifact:** `signal_log.db` uploaded per run, retained 90 days

**Required GitHub Secrets:**

| Secret | Description |
|--------|-------------|
| `ALPACA_API_KEY` | Paper account API key |
| `ALPACA_SECRET_KEY` | Paper account secret key |
| `ALPACA_BASE_URL` | Optional; defaults to paper URL |
| `SLACK_WEBHOOK_URL` | Optional; Slack incoming webhook |
| `ALERT_EMAIL` | Optional; alert recipient address |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` | Optional; SMTP config |

---

## SQLite Schema (`data/live/signal_log.db`)

| Table | Columns |
|-------|---------|
| `signals` | id, ticker, date, direction, label, confidence, p_up, p_flat, p_down, kelly_frac, atr, stop_loss, close, run_ts |
| `orders` | id, order_id, ticker, side, qty, order_type, status, stop_price, take_profit, error, run_ts |
| `equity` | id, equity, buying_power, run_ts |
| `circuit_breaker` | ticker (PK), peak_equity, current_equity, last_updated |

---

## Test Coverage

| Test class | Tests |
|------------|-------|
| `TestLiveSignal` | 7 — label, is_actionable edge cases |
| `TestKellyFraction` | 7 — positive edge, zero edge, negative edge, invalid inputs, custom multiplier |
| `TestSignalLogger` | 11 — CRUD for all 4 tables, circuit breaker logic, ordering |
| `TestOrderResult` | 2 — ok property |
| `TestAlpacaBrokerCredentials` | 2 — credential guard, ImportError bubble |
| `TestAlerts` | 3 — no-op, Slack mock, email mock |
| `TestLiveEngine` | 5 — dry_run flow, FLAT skip, zero-Kelly skip |
| **Total** | **37** |

---

## Gate to Phase 07

> Orders execute cleanly on Alpaca paper; signal_log.db accumulates daily entries;
> no duplicate orders on the same day for the same ticker.

Verify with:
```bash
python3 live.py run --dry-run          # smoke test without credentials
python3 live.py status                 # inspect the log
```
