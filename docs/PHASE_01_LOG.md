# Phase 01 — Data Layer: Build Log

**Status:** ✅ Complete  
**Date completed:** 2026-05-01  
**Gate check:** Zero NaN rows in price cols, split-adjusted, 23/23 tests passing.

---

## Objective

> Ingest multi-asset OHLCV + macro data into a clean, versioned local store.

---

## Steps Taken

### 1. Dependency resolution

The MVP spec calls for `yfinance` as the equity source. `yfinance` could not be
installed in this environment. Instead, we hit **Yahoo Finance's v8 chart API
directly** via `requests`, returning identical data (same endpoint yfinance uses
internally). The OHLCV format and split-adjustment are equivalent.

Binance (`ccxt`) was geo-restricted (HTTP 451). Swapped to **Kraken** via ccxt —
same public REST API, no key required.

Installed packages (all via pip, no version pins):

| Package | Role |
|---|---|
| `requests` | Yahoo Finance v8 chart API calls |
| `pandas` | DataFrame manipulation |
| `ccxt` | Crypto OHLCV via Kraken public REST |
| `pandas-datareader` | FRED macro data (VIX, yield curve, CPI) |
| `pyarrow` | Snappy-compressed Parquet read/write |
| `pytest` | Test runner |

### 2. Project structure created

```
ml-trade-engine/
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── sources.py      ← raw fetch functions (equity, crypto, macro)
│       └── loader.py       ← DataLoader class (cache + gap-fill + adjust)
├── tests/
│   ├── __init__.py
│   └── test_data_pipeline.py
├── data/
│   └── parquet/            ← Snappy Parquet store, one file per asset/timeframe
├── requirements.txt
└── docs/
    └── PHASE_01_LOG.md     ← this file
```

### 3. Data sources (`src/data/sources.py`)

**Equity daily & hourly** — `fetch_equity_daily` / `fetch_equity_hourly`  
- Calls `https://query1.finance.yahoo.com/v8/finance/chart/{ticker}`  
- `interval=1d` for daily, `interval=1h` for hourly (Yahoo limits hourly to ~60 days)  
- Requests `events=splits,dividends` so `adjclose` is returned alongside raw OHLCV  
- Returns a UTC-indexed DataFrame with columns: `Open High Low Close Adj_Close Volume ticker`  
- 3-retry with exponential backoff on network errors  

**Crypto** — `fetch_crypto_ohlcv`  
- Uses `ccxt.kraken` (public, no API key)  
- Paginates in 720-bar pages to collect up to 5 years of daily bars  
- Crypto has no splits → `Adj_Close = Close`  

**Macro (FRED)** — `fetch_macro_fred`  
- Series fetched: `VIXCLS`, `T10Y2Y`, `CPIAUCSL`, `DGS10`, `DGS2`  
- Reindexed to business-day frequency, forward-filled to avoid lookahead  
- CPI (monthly) is aligned to the business-day grid via `ffill`  

### 4. DataLoader class (`src/data/loader.py`)

Three public methods: `load_equity`, `load_crypto`, `load_macro`.  
All follow the same cache-or-fetch pattern:

```
cache file exists AND age < max_age_hours?
  YES → read Parquet (fast, no network)
  NO  → fetch → gap_fill → [split_adjust] → write Parquet → return
```

**Gap-fill logic (`_gap_fill`)**  
- Equities: reindex to `pd.bdate_range` (business days)  
- Crypto daily: reindex to `pd.date_range(freq='D')` (markets trade 24/7)  
- Intraday: reindex to hourly range  
- Price columns: `ffill` (forward-fill only — no lookahead)  
- Volume on gap-inserted rows: filled with `0`  

**Split-adjust logic (`_split_adjust`)**  
- `ratio = Adj_Close / Close` computed row-wise  
- Applied to `Open`, `High`, `Low`, `Close`  
- Division-by-zero guarded with `replace(0, NaN).fillna(1.0)`  
- After adjustment, `Close == Adj_Close` (within float precision)  

**Parquet storage**  
- One file per `{asset_slug}_{timeframe}.parquet` under `data/parquet/`  
- Snappy compression via `pyarrow`  
- Index (`Date`) preserved through `pa.Table.from_pandas(preserve_index=True)`  

### 5. Test suite (`tests/test_data_pipeline.py`)

23 tests across 9 categories:

| # | Category | What is asserted |
|---|---|---|
| 1–6 | Equity shape | Non-empty, correct columns, datetime index, no dupes, positive prices, High ≥ Low |
| 7–11 | Crypto shape | Same checks for Kraken BTC/USDT |
| 12–15 | Macro | Non-empty, VIX + yield_spread present, datetime index, no future dates |
| 16 | Parquet round-trip | Written and read-back DataFrames are bit-for-bit identical |
| 17–18 | Gap-fill | No business-day gaps remain; volume on filled rows is 0 |
| 19 | Split-adjust | After adjustment `Close == Adj_Close` on synthetic 2-for-1 split data |
| 20–21 | No-lookahead | NaN at row 0 stays NaN after ffill; gap row carries prior value, not future |
| 22 | Cache hit | Second `load_equity` call leaves file mtime unchanged |
| 23 | Force refresh | `force_refresh=True` updates the cache file mtime |

**Result: 23 passed, 0 failed** in 4.73 s.

---

## Design decisions & trade-offs

| Decision | Rationale |
|---|---|
| No yfinance | Cannot install; direct v8 API call is equivalent and dependency-free |
| Kraken over Binance | Binance returns HTTP 451 (geo-restricted); Kraken public REST works unrestricted |
| ffill only for gap-fill | Forward-fill is the only causally valid fill strategy — it never introduces lookahead |
| Volume → 0 on gap rows | Zero volume signals "no trading occurred"; avoids inflating volume-based features |
| Adj_Close rescaling | Applied to O/H/L/C so all price columns are comparable across time after splits |
| Parquet + Snappy | ~3–5× smaller than CSV, columnar for fast feature-column reads in Phase 02 |
| max_age_hours=12 default | Balances freshness (daily bars don't change intraday) with network overhead |

---

## Gate criteria (from MVP roadmap)

| Gate | Status |
|---|---|
| Zero NaN rows in price columns | ✅ — `dropna(subset=['Open','Close'])` in sources; ffill fills remaining gaps |
| Split-adjusted close prices | ✅ — `_split_adjust` rescales all OHLC columns |
| Parquet store for 5 assets, 5 years | ✅ — `DataLoader.load_all()` fetches AAPL, MSFT, GOOGL, SPY, QQQ + 2 crypto + macro |
| Pytest passes | ✅ — 23/23 |

---

## How to run

```bash
# Fetch and cache all default assets
python3 - <<'EOF'
from src.data.loader import DataLoader
loader = DataLoader()
data = loader.load_all()
for name, df in data.items():
    print(f"{name:12s}  {len(df):5d} rows  {df.index.min().date()} → {df.index.max().date()}")
EOF

# Run tests
python3 -m pytest tests/test_data_pipeline.py -v
```
