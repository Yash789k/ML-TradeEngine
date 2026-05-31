"""
Phase 06A/B/C — Research CLI

Subcommands
-----------
  zoo    Run all strategies on all cached tickers (Phase 06A)
  rank   Enrich scorecard with alpha/beta/t-stat/IR and rank (Phase 06B)
  show   Print the ranked scorecard table
  env    Run Phase 06C environment characterisation (4 analyses)

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

  # Phase 06C: run all 4 environment analyses (uses saved artifacts)
  python3 research.py env

  # Run only specific analyses
  python3 research.py env --regime
  python3 research.py env --decay
  python3 research.py env --factors
  python3 research.py env --cost --cost-tickers AAPL NVDA SPY TSLA GLD

  # Show saved env results without re-running
  python3 research.py env --show-only
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

    # ── env ───────────────────────────────────────────────────────────────
    env_p = sub.add_parser(
        "env",
        help="Phase 06C — Environment Characterisation (4 analyses).",
    )
    env_p.add_argument(
        "--regime",   action="store_true", help="Run regime breakdown only."
    )
    env_p.add_argument(
        "--cost",     action="store_true", help="Run cost sensitivity only."
    )
    env_p.add_argument(
        "--decay",    action="store_true", help="Run signal decay only."
    )
    env_p.add_argument(
        "--factors",  action="store_true", help="Run factor attribution only."
    )
    env_p.add_argument(
        "--show-only", action="store_true",
        help="Print saved results without re-running any analysis.",
    )
    env_p.add_argument(
        "--cost-tickers", nargs="+", default=None, metavar="TICK",
        help="Subset of tickers for the cost sensitivity sweep "
             "(default: AAPL NVDA SPY TSLA GLD).",
    )
    env_p.add_argument(
        "--research", action="store_true",
        help="Load the full 22-ticker universe for regime/cost analyses.",
    )
    env_p.add_argument(
        "--tickers", nargs="+", default=None, metavar="TICK",
        help="Explicit ticker subset for regime/cost analyses.",
    )

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
# Command: env  (Phase 06C)
# ---------------------------------------------------------------------------

def cmd_env(args: argparse.Namespace) -> None:
    from src.data.loader import DataLoader
    from src.research.env_analyzer import EnvAnalyzer
    from src.research.strategies import ALL_STRATEGIES

    analyzer = EnvAnalyzer()

    # ── show-only: print saved results without running anything ──────────
    if args.show_only:
        env_dir = Path("data/research/env")
        if (env_dir / "regime_breakdown.parquet").exists():
            import pandas as pd
            analyzer.print_regime_summary(
                pd.read_parquet(env_dir / "regime_breakdown.parquet")
            )
        if (env_dir / "cost_sensitivity.parquet").exists():
            import pandas as pd
            analyzer.print_cost_summary(
                pd.read_parquet(env_dir / "cost_sensitivity.parquet")
            )
        if (env_dir / "signal_decay_summary.parquet").exists():
            import pandas as pd
            analyzer.print_decay_summary(
                pd.read_parquet(env_dir / "signal_decay_summary.parquet")
            )
        if (env_dir / "factor_attribution.parquet").exists():
            import pandas as pd
            analyzer.print_factor_summary(
                pd.read_parquet(env_dir / "factor_attribution.parquet")
            )
        return

    # ── Determine which analyses to run ──────────────────────────────────
    run_all = not any([args.regime, args.cost, args.decay, args.factors])

    # ── Resolve tickers ───────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif getattr(args, "research", False):
        tickers = _RESEARCH_TICKERS
    else:
        tickers = _DEFAULT_TICKERS

    # ── Load OHLCV data (only needed for regime + cost analyses) ─────────
    ohlcv_dict: dict = {}
    spy_df  = None
    macro   = None
    loader  = DataLoader()

    if run_all or args.regime or args.cost:
        log.info("Loading OHLCV data for %d tickers …", len(tickers))
        for ticker in tickers:
            try:
                ohlcv_dict[ticker] = loader.load_equity(ticker)
                log.info("  ✓ %s", ticker)
            except Exception as exc:
                log.warning("  ✗ %s (%s)", ticker, exc)

        if "SPY" not in ohlcv_dict:
            try:
                spy_df = loader.load_equity("SPY")
            except Exception:
                pass
        else:
            spy_df = ohlcv_dict["SPY"]

        try:
            macro = loader.load_macro()
        except Exception:
            pass

    # ── Run selected analyses ─────────────────────────────────────────────
    if run_all or args.regime:
        log.info("\n── 1/4 Regime Breakdown ─────────────────────────────────")
        df = analyzer.regime_breakdown(ohlcv_dict)
        analyzer.print_regime_summary(df)

    if run_all or args.cost:
        cost_ticker_list = (
            [t.upper() for t in args.cost_tickers]
            if args.cost_tickers
            else ["AAPL", "NVDA", "SPY", "TSLA", "GLD"]
        )
        cost_dict = {t: ohlcv_dict[t] for t in cost_ticker_list if t in ohlcv_dict}
        if not cost_dict:
            log.warning("No cost-sensitivity tickers available in loaded data — skipping.")
        else:
            log.info("\n── 2/4 Cost Sensitivity (%s) ────────────────────────",
                     list(cost_dict.keys()))
            df = analyzer.cost_sensitivity(
                cost_dict, ALL_STRATEGIES, spy_df=spy_df, macro=macro
            )
            analyzer.print_cost_summary(df)

    if run_all or args.decay:
        log.info("\n── 3/4 Signal Decay ─────────────────────────────────────")
        df = analyzer.signal_decay()
        analyzer.print_decay_summary(df)

    if run_all or args.factors:
        log.info("\n── 4/4 Factor Attribution (Fama-French 3) ───────────────")
        df = analyzer.factor_attribution()
        analyzer.print_factor_summary(df)

    log.info("\nPhase 06C complete. Artifacts saved to data/research/env/")


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
    elif args.command == "env":
        cmd_env(args)
    else:
        log.error("Unknown command: %s", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
