"""
Phase 06A/B — Research CLI

Subcommands
-----------
  zoo    Run all strategies on all cached tickers (Phase 06A)
  rank   Enrich scorecard with alpha/beta/t-stat/IR and rank (Phase 06B)
  show   Print the ranked scorecard table

Usage Examples
--------------
  # Run all strategies on default 5 tickers
  python3 research.py zoo

  # Run on the full 22-ticker research universe (requires --research data fetch first)
  python3 research.py zoo --research

  # Run on specific tickers only
  python3 research.py zoo --tickers AAPL MSFT GOOGL SPY QQQ

  # Run only select strategies
  python3 research.py zoo --strategies momentum alpha_trends ema_crossover

  # Phase 06B: rank after zoo completes
  python3 research.py rank

  # Print top-20 ranked strategies
  python3 research.py show --top 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]

_RESEARCH_TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "BAC", "GS",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
    "GLD", "TLT", "IWM", "DIA", "SPY", "QQQ",
]

_STRATEGY_MAP = {
    "momentum":      "MomentumStrategy",
    "mean_reversion": "MeanReversionStrategy",
    "ema_crossover": "EMACrossoverStrategy",
    "turtle":        "TurtleStrategy",
    "pairs_arb":     "PairsArbStrategy",
    "carry_proxy":   "CarryProxyStrategy",
    "vol_breakout":  "VolBreakoutStrategy",
    "alpha_trends":  "AlphaTrendsStrategy",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 06 — Strategy Zoo & Ranking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── zoo ───────────────────────────────────────────────────────────────
    zoo_p = sub.add_parser("zoo", help="Run Phase 06A Strategy Zoo.")
    zoo_p.add_argument(
        "--research", action="store_true",
        help="Run on the full 22-ticker research universe instead of defaults.",
    )
    zoo_p.add_argument(
        "--tickers", nargs="+", default=None, metavar="TICK",
        help="Explicit ticker list (overrides --research and defaults).",
    )
    zoo_p.add_argument(
        "--strategies", nargs="+", default=None, metavar="STRAT",
        choices=list(_STRATEGY_MAP.keys()) + ["all"],
        help="Strategies to include (default: all). "
             f"Choices: {', '.join(_STRATEGY_MAP.keys())}.",
    )
    zoo_p.add_argument("--capital",    type=float, default=10_000.0)
    zoo_p.add_argument("--commission", type=float, default=0.001)
    zoo_p.add_argument("--slippage",   type=float, default=0.0005)
    zoo_p.add_argument(
        "--no-save", action="store_true",
        help="Skip writing results to disk (dry run).",
    )

    # ── rank ──────────────────────────────────────────────────────────────
    sub.add_parser("rank", help="Run Phase 06B Ranking (requires zoo output).")

    # ── show ──────────────────────────────────────────────────────────────
    show_p = sub.add_parser("show", help="Print ranked scorecard.")
    show_p.add_argument("--top", type=int, default=20, help="Top-N rows to show.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Command: zoo
# ---------------------------------------------------------------------------

def cmd_zoo(args: argparse.Namespace) -> None:
    from src.data.loader import DataLoader
    from src.research.strategies import (
        ALL_STRATEGIES,
        AlphaTrendsStrategy,
        CarryProxyStrategy,
        EMACrossoverStrategy,
        MeanReversionStrategy,
        MomentumStrategy,
        PairsArbStrategy,
        TurtleStrategy,
        VolBreakoutStrategy,
    )
    from src.research.zoo_runner import ZooRunner

    _CLASS_MAP = {
        "momentum":       MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "ema_crossover":  EMACrossoverStrategy,
        "turtle":         TurtleStrategy,
        "pairs_arb":      PairsArbStrategy,
        "carry_proxy":    CarryProxyStrategy,
        "vol_breakout":   VolBreakoutStrategy,
        "alpha_trends":   AlphaTrendsStrategy,
    }

    # Resolve strategies
    if args.strategies is None or args.strategies == ["all"]:
        strategies = ALL_STRATEGIES
    else:
        strategies = [_CLASS_MAP[s]() for s in args.strategies]

    log.info("Strategies: %s", [s.name for s in strategies])

    # Resolve tickers: explicit --tickers > --research universe > default 5
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif getattr(args, "research", False):
        tickers = _RESEARCH_TICKERS
        log.info("Using full research universe (%d tickers)", len(tickers))
    else:
        tickers = _DEFAULT_TICKERS
    log.info("Tickers: %s", tickers)

    # Load data
    loader = DataLoader()
    ohlcv_dict: dict = {}
    spy_df = None
    macro  = None

    log.info("Loading data …")
    for ticker in tickers:
        try:
            ohlcv_dict[ticker] = loader.load_equity(ticker)
            log.info("  ✓ %s", ticker)
        except Exception as exc:
            log.warning("  ✗ %s — skipped (%s)", ticker, exc)

    # SPY for pairs strategies
    if "SPY" not in ohlcv_dict:
        try:
            spy_df = loader.load_equity("SPY")
        except Exception as exc:
            log.warning("Could not load SPY for pairs strategies: %s", exc)
    else:
        spy_df = ohlcv_dict["SPY"]

    # Macro for carry strategy
    try:
        macro = loader.load_macro()
        log.info("  ✓ macro")
    except Exception as exc:
        log.warning("Could not load macro data: %s", exc)

    if not ohlcv_dict:
        log.error("No tickers loaded — aborting.")
        sys.exit(1)

    # Run zoo
    runner = ZooRunner(
        strategies      = strategies,
        initial_capital = args.capital,
        commission      = args.commission,
        slippage        = args.slippage,
    )

    log.info("\nRunning Strategy Zoo — %d ticker(s) × %d strategy(ies) …",
             len(ohlcv_dict), len(strategies))

    scorecard = runner.run(
        ohlcv_dict = ohlcv_dict,
        spy_df     = spy_df,
        macro      = macro,
        save       = not args.no_save,
    )

    # Print summary table
    print("\n" + "─" * 80)
    print(f"  Strategy Zoo Results — {len(scorecard)} runs")
    print("─" * 80)
    display_cols = ["sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio",
                    "n_trades", "win_rate"]
    present = [c for c in display_cols if c in scorecard.columns]
    print(scorecard[present].to_string())
    print("─" * 80)


# ---------------------------------------------------------------------------
# Command: rank
# ---------------------------------------------------------------------------

def cmd_rank(_args: argparse.Namespace) -> None:
    from src.research.ranker import Ranker

    ranker = Ranker()
    log.info("Computing extended metrics and ranking …")
    ranked = ranker.rank()

    print("\n" + "─" * 90)
    print(f"  Phase 06B — Ranked Scorecard  ({len(ranked)} strategies)")
    print("─" * 90)
    ranker.print_table(ranked)
    print("─" * 90)


# ---------------------------------------------------------------------------
# Command: show
# ---------------------------------------------------------------------------

def cmd_show(args: argparse.Namespace) -> None:
    from src.research.ranker import Ranker

    ranker = Ranker()
    try:
        df = ranker.top_n(args.top)
    except FileNotFoundError as exc:
        log.error("%s — run 'python3 research.py zoo' first.", exc)
        sys.exit(1)

    print("\n" + "─" * 90)
    print(f"  Top-{args.top} Strategies by Composite Score")
    print("─" * 90)
    ranker.print_table(df, n=args.top)
    print("─" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.command == "zoo":
        cmd_zoo(args)
    elif args.command == "rank":
        cmd_rank(args)
    elif args.command == "show":
        cmd_show(args)
    else:
        log.error("Unknown command: %s", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
