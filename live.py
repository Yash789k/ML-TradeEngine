"""
Phase 06E — Live Signal Engine CLI

Subcommands
-----------
  run       Generate signals and submit paper orders for today
  status    Print the most recent signals from the SQLite log
  orders    Print the most recent orders from the SQLite log
  equity    Print the account equity history
  positions Print open positions from Alpaca (requires credentials)

Usage Examples
--------------
  # Run full pipeline on default 5 tickers (submits Alpaca paper orders)
  python3 live.py run

  # Dry run — generate signals without submitting orders
  python3 live.py run --dry-run

  # Run on specific tickers
  python3 live.py run --tickers AAPL MSFT NVDA

  # Run on the full research universe (22 tickers)
  python3 live.py run --tickers AAPL MSFT GOOGL AMZN NVDA META TSLA JPM BAC GS \
                                XLK XLF XLE XLV XLI XLP GLD TLT IWM DIA SPY QQQ

  # Show last 20 signals from log
  python3 live.py status --n 20

  # Show last 10 orders
  python3 live.py orders --n 10

  # Show open Alpaca positions
  python3 live.py positions

  # Show equity history (last 30 snapshots)
  python3 live.py equity --n 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 06E — Live Signal Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── run ───────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Generate signals and submit paper orders.")
    run_p.add_argument(
        "--tickers", nargs="+", default=None, metavar="TICK",
        help="Ticker list (default: AAPL MSFT GOOGL SPY QQQ).",
    )
    run_p.add_argument(
        "--dry-run", action="store_true",
        help="Generate signals but do NOT submit any orders.",
    )
    run_p.add_argument(
        "--no-refresh", action="store_true",
        help="Use cached data (skip live data fetch).",
    )
    run_p.add_argument(
        "--confidence", type=float, default=0.38,
        help="Minimum confidence threshold to act on a signal (default 0.38).",
    )
    run_p.add_argument(
        "--max-heat", type=float, default=0.80,
        help="Max portfolio heat fraction (default 0.80).",
    )

    # ── status ────────────────────────────────────────────────────────────
    status_p = sub.add_parser("status", help="Show recent signals from log.")
    status_p.add_argument("--n", type=int, default=20, help="Number of rows to show.")

    # ── orders ────────────────────────────────────────────────────────────
    orders_p = sub.add_parser("orders", help="Show recent orders from log.")
    orders_p.add_argument("--n", type=int, default=20, help="Number of rows to show.")

    # ── equity ────────────────────────────────────────────────────────────
    eq_p = sub.add_parser("equity", help="Show paper account equity history.")
    eq_p.add_argument("--n", type=int, default=30, help="Number of snapshots to show.")

    # ── positions ─────────────────────────────────────────────────────────
    sub.add_parser("positions", help="Show open Alpaca paper positions.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    from src.live.engine import LiveEngine

    tickers = [t.upper() for t in args.tickers] if args.tickers else _DEFAULT_TICKERS
    log.info("Tickers: %s", tickers)

    engine = LiveEngine(
        tickers              = tickers,
        confidence_threshold = args.confidence,
        max_portfolio_heat   = args.max_heat,
        dry_run              = args.dry_run,
    )

    result = engine.run(force_refresh=not args.no_refresh)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"  Live Engine Run Summary   dry_run={args.dry_run}")
    print("─" * 70)

    print(f"\n  Signals ({len(result.signals)}):")
    for ticker, sig in result.signals.items():
        print(
            f"    {ticker:<8}  {sig.label:<5}  conf={sig.confidence:.3f}"
            f"  kelly={sig.kelly_frac:.3f}  close=${sig.close:.2f}"
        )

    if result.orders:
        print(f"\n  Orders ({len(result.orders)}):")
        for ticker, order in result.orders.items():
            if order is None:
                print(f"    {ticker:<8}  [dry_run — no order]")
            else:
                status_str = order.status if order.ok else f"ERROR: {order.error}"
                print(
                    f"    {ticker:<8}  {order.side:<5}  qty={order.qty:.4f}"
                    f"  {status_str}"
                )

    if result.skipped:
        print(f"\n  Skipped ({len(result.skipped)}):")
        for ticker, reason in result.skipped:
            print(f"    {ticker:<8}  {reason}")

    if result.account_equity:
        print(f"\n  Account equity: ${result.account_equity:,.2f}")

    print("─" * 70)


def cmd_status(args: argparse.Namespace) -> None:
    from src.live.logger import SignalLogger
    logger = SignalLogger()
    rows   = logger.recent_signals(args.n)

    print("\n" + "─" * 90)
    print(f"  Recent Signals (last {len(rows)})")
    print("─" * 90)
    print(f"  {'id':>4}  {'ticker':<8}  {'date':<12}  {'label':<5}  "
          f"{'conf':>6}  {'kelly':>7}  {'close':>8}  {'stop':>8}")
    print("  " + "─" * 86)
    for r in rows:
        date_str = str(r.get("date", ""))[:10]
        print(
            f"  {r['id']:>4}  {r['ticker']:<8}  {date_str:<12}  {r['label']:<5}"
            f"  {r['confidence']:>6.3f}  {r['kelly_frac']:>7.4f}"
            f"  {r['close']:>8.2f}  {r['stop_loss']:>8.2f}"
        )
    print("─" * 90)


def cmd_orders(args: argparse.Namespace) -> None:
    from src.live.logger import SignalLogger
    logger = SignalLogger()
    rows   = logger.recent_orders(args.n)

    print("\n" + "─" * 90)
    print(f"  Recent Orders (last {len(rows)})")
    print("─" * 90)
    print(f"  {'id':>4}  {'ticker':<8}  {'side':<5}  {'qty':>8}  "
          f"{'status':<15}  {'stop':>8}  {'tp':>8}  run_ts")
    print("  " + "─" * 86)
    for r in rows:
        stop_str = f"{r['stop_price']:.2f}" if r.get("stop_price") else "—"
        tp_str   = f"{r['take_profit']:.2f}" if r.get("take_profit") else "—"
        err_str  = f" [ERR: {r['error']}]" if r.get("error") else ""
        print(
            f"  {r['id']:>4}  {r['ticker']:<8}  {r['side']:<5}  {r['qty']:>8.4f}"
            f"  {r['status']:<15}  {stop_str:>8}  {tp_str:>8}"
            f"  {str(r['run_ts'])[:19]}{err_str}"
        )
    print("─" * 90)


def cmd_equity(args: argparse.Namespace) -> None:
    from src.live.logger import SignalLogger
    logger = SignalLogger()
    rows   = logger.equity_history(args.n)

    print("\n" + "─" * 60)
    print(f"  Account Equity History (last {len(rows)} snapshots)")
    print("─" * 60)
    for r in rows:
        ts_str = str(r.get("run_ts", ""))[:19]
        print(f"  {ts_str}  equity=${r['equity']:>12,.2f}  "
              f"bp=${r.get('buying_power', 0):>12,.2f}")
    print("─" * 60)


def cmd_positions(_args: argparse.Namespace) -> None:
    from src.live.broker import AlpacaBroker
    try:
        broker = AlpacaBroker()
    except EnvironmentError as exc:
        log.error("%s", exc)
        sys.exit(1)

    positions = broker.get_all_positions()
    equity    = broker.account_equity()

    print("\n" + "─" * 80)
    print(f"  Open Positions   (account equity: ${equity:,.2f})")
    print("─" * 80)
    if not positions:
        print("  No open positions.")
    else:
        print(f"  {'ticker':<8}  {'qty':>8}  {'side':<6}  {'avg_entry':>10}"
              f"  {'mkt_val':>10}  {'unreal_pl':>10}")
        for p in positions:
            pl_sign = "+" if p.unrealized_pl >= 0 else ""
            print(
                f"  {p.ticker:<8}  {p.qty:>8.4f}  {p.side:<6}  "
                f"${p.avg_entry:>9.2f}  ${p.market_val:>9,.2f}  "
                f"{pl_sign}{p.unrealized_pl:>9.2f}"
            )
    print("─" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dispatch = {
        "run":       cmd_run,
        "status":    cmd_status,
        "orders":    cmd_orders,
        "equity":    cmd_equity,
        "positions": cmd_positions,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        log.error("Unknown command: %s", args.command)
        sys.exit(1)

    fn(args)


if __name__ == "__main__":
    main()
