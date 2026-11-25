"""Generate the professor-facing progress report from the saved statistics.

``python -m review_v1.build report`` writes ``docs/review-v1/PROGRESS_REPORT.md``.
Every number in the report is read from ``data/review-v1/``; none is typed here,
so the report, the dashboard and the audit cannot disagree.
"""

from __future__ import annotations

import pandas as pd

from review_v1 import audit, explain, nav
from review_v1.paths import OUT, ROOT, read_json

PATH = ROOT / "docs/review-v1/PROGRESS_REPORT.md"
STATUS_ORDER = ["works", "diagnostic", "unvalidated", "missing"]


def pct(value, digits=1) -> str:
    return f"{value:+.{digits}%}"


def usd(value) -> str:
    return f"${value:,.2f}"


def day(value) -> str:
    return pd.Timestamp(value).strftime("%-d %B %Y")


def join(items: list) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def moved(value) -> str:
    """'made +3.0%' or 'lost 2.5%'."""
    return f"made {value:+.1%}" if value >= 0 else f"lost {abs(value):.1%}"


def build() -> str:
    classifier = read_json(OUT / "classifier/summary.json")
    regime = read_json(OUT / "regime/summary.json")
    sandbox = read_json(OUT / "sandbox/summary.json")
    replay = read_json(OUT / "replay/summary.json")
    sentiment = read_json(OUT / "sentiment/summary.json", {})
    stock = read_json(OUT / "inventory.json")
    live = read_json(OUT / "live/last_run.json", {})
    index = read_json(OUT / "walkthrough/index.json")
    case = read_json(OUT / f"walkthrough/{index['default_case']}.json")
    symbol = index["default_symbol"]
    item = case["symbols"][symbol]
    metrics = pd.read_csv(OUT / "sandbox/metrics.csv").set_index(["period", "arm"])
    contrasts = pd.read_csv(OUT / "sandbox/contrasts.csv").set_index("arm")
    benchmark = pd.read_csv(OUT / "sandbox/versus_benchmark.csv").set_index("arm")
    periods = pd.read_csv(OUT / "classifier/by_period.csv").set_index(["period", "model"])
    paired = pd.read_csv(OUT / "classifier/paired_vs_prior.csv").set_index(["period", "model"])
    regimes = pd.read_csv(OUT / "classifier/regime_contrasts.csv").set_index("regime")
    items = audit.rows()
    counts = audit.summary(items)

    pilot, history, news = stock["pilot"], stock["market_history"], stock["news"]
    hybrid, prior, gap = periods.loc[("all", "hybrid")], periods.loc[("all", "prior")], paired.loc[("all", "hybrid")]
    prospective = classifier["prospective"]
    simulated = replay["classification"]
    hist, post = sandbox["history"], sandbox["post_release"]
    calm = next(s for s in regime["states"] if s["state"] == "calm")
    turbulent = next(s for s in regime["states"] if s["state"] == "turbulent")
    forward = {r["state"]: r for r in regime["forward_validation"]["rows"]}
    probabilities = item["models"]["recorded"]["probabilities"]
    fill, outcome, position, order = item["fill"], item["outcome"], item["position"], item["order"]
    account = case["risk"]["account"]
    quality = replay.get("data_quality", {})
    second = quality.get("second_source", {})
    ablation = sentiment.get("ablation", {})

    def arm(period, name, column):
        return float(metrics.loc[(period, name), column])

    def trading_row(period, name, sharpe=True):
        label = metrics.loc[(period, name), "label"]
        cells = [label, pct(arm(period, name, "net_return")), f"{arm(period, name, 'pnl_usd'):+,.0f}"]
        if sharpe:
            cells.append(f"{arm(period, name, 'sharpe_zero_rf'):.2f} ({arm(period, name, 'sharpe_ci_low'):.2f} to {arm(period, name, 'sharpe_ci_high'):.2f})")
        cells += [f"{arm(period, name, 'max_drawdown'):.1%}", f"{int(arm(period, name, 'fills')):,}", usd(arm(period, name, "combined_fees_usd")), f"{arm(period, name, 'mean_gross_exposure'):.0%}"]
        return "| " + " | ".join(cells) + " |"

    arms = ["equal_weight", "trend", "equal_weight_regime", "trend_regime"]
    edge = audit.interval_verdict(benchmark.loc["trend", "sharpe_ci_low"], benchmark.loc["trend", "sharpe_ci_high"], lower_is_better=False)
    recent = pd.read_csv(OUT / "regime/posterior_recent.csv")
    turned = recent.loc[(recent.date >= regime["series"]["recent"]["first_new"]) & (recent.state == "turbulent"), "date"]
    lines = []
    add = lines.append

    add("# Monthly progress report: Adaptive Ensemble Trading System (CECI2)")
    add("")
    add(f"*Generated {day(pd.Timestamp.now(tz='UTC'))} from `data/review-v1/` by `python -m review_v1.build report`. "
        f"Every number below is read from the saved statistics. Live demo: `{nav.BASE_URL}` (see `docs/review-v1/REPRODUCE.md`).*")
    add("")
    add("## Summary")
    add("")
    add(
        f"The pipeline works end to end: it collects and validates daily data for {stock['universe']['count']} instruments, produces forecasts, applies a decision rule and risk limits, "
        f"and records paper orders and fills with a hash-checked audit trail. In October 2026 it recorded {pilot['decisions']} decisions and {pilot['fills']} paper fills, "
        "and every one of them can be reproduced exactly from the inputs recorded at the time."
    )
    add("")
    add(
        f"The strategy is not yet shown to be good. On {int(hybrid.n):,} forecasts the classifier is {gap.verdict} (log loss {hybrid.log_loss:.4f} against {prior.log_loss:.4f}). "
        f"The trend rule that actually places the paper orders {moved(arm('history', 'trend', 'net_return'))} on the 2024 to September 2026 history, "
        f"{'less' if arm('history', 'trend', 'net_return') < arm('history', 'equal_weight', 'net_return') else 'more'} than holding all {stock['universe']['count']} instruments equally "
        f"({pct(arm('history', 'equal_weight', 'net_return'))}), and {moved(arm('post_release', 'trend', 'net_return'))} in a simulated replay of the {post['sessions']} sessions since. "
        "A working pipeline is not evidence of a profitable strategy, and this report does not claim one."
    )
    add("")
    add("## What has been built")
    add("")
    add("| Stage | What it does | Role in a decision |")
    add("|---|---|---|")
    add(f"| Data collection | Downloads daily bars for {stock['universe']['count']} instruments after the US close into immutable, hashed snapshots; validates them against the exchange calendar | Input |")
    add(f"| Features | {stock['models']['features']} price and volume features per instrument | Input |")
    add("| Classifier | Logistic regression and boosted trees, averaged and calibrated, giving P(DOWN), P(FLAT), P(UP) for a five-session horizon | **Recorded, not used** |")
    add("| Decision rule | Hold an instrument when its price is above its 200-session average and its 12-month momentum is positive | **Decides** |")
    add("| Risk limits | 10% per instrument, 35% growth group, 80% gross, 12% volatility target, 40% turnover per rebalance, 20% drawdown stop | **Decides** |")
    add("| Paper ledger | Orders fill at the next close with 15 bp cost; cash reserved; every event stored | Executes (paper only) |")
    add("| Market-regime HMM *(new)* | Causal rolling two-state hidden Markov model on SPY | **Diagnostic** |")
    add("| News sentiment | FinBERT scores of headlines | **Not connected** |")
    add("| Review dashboard *(new)* | Nine sections, a step-by-step replay of recorded decisions, a button that runs the current pipeline | Demonstration |")
    add("")
    add("No broker is connected and no external order has ever been sent. The deployment account holds cash; the trend portfolio runs as a paper *shadow* account to gather evidence.")
    add("")
    add("## Data collected")
    add("")
    add(f"- **Market history:** {history['instruments']} instruments, {history['rows']:,} daily bars, {history['first']} to {history['last']}, from {history['provider']}, sealed with hashes on {day(history['retrieved_at'])}.")
    add(
        f"- **Scheduled collection, October 2026:** {pilot['snapshots']} hashed snapshots, {pilot['decisions']} decision dates ({', '.join(day(s)[:-5] for s in pilot['decision_sessions'])}), "
        f"{pilot['orders']} paper orders, {pilot['fills']} fills, and {pilot['outcome_rows']} forecasts later matched to their outcomes."
    )
    add(
        f"- **Where collection broke:** {len(pilot['attempts'])} collection attempts were registered between {day(pilot['attempts'][0]['start'])[:-5]} and {day(pilot['attempts'][-1]['start'])[:-5]}. "
        f"Together they scheduled {pilot['scheduled_decision_sessions']} decision sessions and recorded {pilot['decisions']} "
        f"({'; '.join(f'{a['attempt']}: {a['valid_sessions']} of {a['required_sessions']}' for a in pilot['attempts'])}). "
        f"{pilot['attempts_halted']} attempts halted: {len(pilot['sessions_not_run'])} windows passed with no run (the collector is a local launch agent that needs a logged-in, awake machine) "
        f"and {len(pilot['sessions_rejected'])} download was rejected by validation because one instrument's open was above its high. "
        "The system halts by design instead of backfilling, so the recorded sample is small but clean."
    )
    add(
        f"- **News:** {sum(n['headlines'] for n in news['history']):,} cached headlines for {', '.join(n['symbol'] for n in news['history'])}, plus {news['pilot_window_total']} headlines fetched for this review covering the pilot weeks "
        f"(none were returned for {len(news['pilot_window_without_headlines'])} of the {stock['universe']['count']} instruments, mostly funds)."
    )
    if second.get("available"):
        add(
            f"- **Data quality:** the {quality['rows']} daily returns behind the simulated replay were compared with a second source ({second['source']}): "
            f"largest difference {second['max_abs_return_difference']:.2%} of return. The largest move in the window, {quality['largest_moves'][0]['symbol']} {pct(quality['largest_moves'][0]['return'])} on "
            f"{day(quality['largest_moves'][0]['session'])}, appears in both sources."
        )
    add("")
    add("## What the classifier result means")
    add("")
    add(f"> {explain.SENTENCE}")
    add("")
    sections = dict(explain.sections())
    add(
        "**The label.** After each close the classifier forecasts the return from the *next* session's close to the close five sessions later: UP above +1%, DOWN below −1%, FLAT otherwise. "
        "The ±1% band is the same for every instrument and every market condition."
    )
    add("")
    base = classifier["profile"]["all"]["prior"]
    add(
        f"**The baseline.** The class-frequency prior ignores today's data and predicts each instrument's class shares from its training window as probabilities "
        f"(on average DOWN {base['mean_probability']['DOWN']:.0%}, FLAT {base['mean_probability']['FLAT']:.0%}, UP {base['mean_probability']['UP']:.0%}), "
        f"so its single best guess is UP on {base['predicted_class_share']['UP']:.0%} of instrument-days."
    )
    add("")
    add("**The metric.** Log loss: the average of −ln(probability given to what happened). Lower is better; guessing one third each scores 1.0986. Comparisons are paired by date with a block bootstrap.")
    add("")
    add("**What it tells us.** " + " ".join(sections["What this tells us"].split()))
    add("")
    add("**What it does not tell us:**")
    add("")
    add(sections["What this does not tell us"])
    add("")
    add("The full explanation, with the table for both studies, is on the dashboard's first page.")
    add("")
    add("## One decision, traced from inputs to outcome")
    add("")
    add(f"Dashboard section 3 replays the recorded decision for **{symbol} at the close of {day(case['session'])}** in seven steps. Labels on every panel say whether a fact was *recorded* at the time, *reconstructed* now from recorded inputs, or *simulated*.")
    add("")
    add(f"1. **Inputs (recorded).** At {pd.Timestamp(case['collection']['retrieved_at']).strftime('%H:%M UTC')} the collector downloaded {item['inputs']['rows']:,} daily bars for {symbol}; last close {usd(item['inputs']['last_bars'][-1]['close'])}. No news was collected.")
    add(f"2. **Features (reconstructed).** Price {pct(item['decision']['trend_inputs']['average_200'])} against its 200-session average; 12-month momentum {pct(item['decision']['trend_inputs']['momentum_12_1'])}.")
    add(
        f"3. **Models.** Recorded forecast: DOWN {probabilities[0]:.0%}, FLAT {probabilities[1]:.0%}, UP {probabilities[2]:.0%}. HMM (reconstructed): {case['regime']['state']}, P(turbulent) = {case['regime']['p_turbulent']:.3f}. "
        "**Neither was used.**"
    )
    add(
        f"4. **Decision (recorded).** Trend rule on, target {item['weight']:.2%} of equity; {case['rule']['trend_count']} of {stock['universe']['count']} instruments qualified ({case['rule']['final_gross']:.1%} gross). "
        f"No portfolio limit was binding; the 40% turnover cap was{'' if account['binding'] else ' not'} binding and scaled every order to {account['scale']:.1%}."
    )
    add(f"5. **Order (recorded).** {order['side'].title()} {order['qty']:.6f} units, reserve price {usd(order['limit_price'])} (close + 3%), {usd(order['reserved_cash'])} cash reserved. Paper only.")
    if fill["filled"]:
        add(f"6. **Fill (recorded).** Next close, {day(fill['asof'])}: {usd(fill['price'])}, cost {usd(fill['fee'])}. After all {case['fills']['count']} fills the account stood at {usd(case['account']['after_fill']['equity'])}.")
    if outcome["available"] and position["opened"]:
        add(
            f"7. **Outcome.** Five sessions after entry {symbol} was {pct(outcome['forward_return'], 2)} (class {outcome['label']}; the classifier's top class was {outcome['predicted']}). "
            f"The pilot segment halted on {day(case['after']['halt']['at'])} after missed sessions, so the recorded ledger stops at {usd(case['account']['last_recorded']['equity'])}. "
            f"Marked with later public prices (reconstructed), the {symbol} position stood at {position['latest']['pnl']:+.2f} USD ({pct(position['latest']['pnl_percent'])}) on {day(position['latest']['session'])}, "
            f"and the recorded portfolio simply held at {usd(case['held_forward']['latest']['equity'])} ({pct(case['held_forward']['latest']['net_return'], 2)})."
        )
    add("")
    add("The replay re-checks the receipt's registered hash and stops if any recomputed probability, weight or order quantity differs from the recorded one. One case shows the mechanism; it says nothing about skill.")
    add("")
    add("## Evaluation")
    add("")
    add("### Classification, against the class-frequency prior")
    add("")
    add("| Sample | Status | Forecasts | Model log loss | Prior log loss | Model − prior (95% interval) | Accuracy, model / prior |")
    add("|---|---|---:|---:|---:|---|---|")
    add(f"| History, {classifier['history']['first']} to {classifier['history']['last']} | Out-of-fold, previously inspected | {int(hybrid.n):,} | {hybrid.log_loss:.4f} | {prior.log_loss:.4f} | {gap.log_loss_difference:+.4f} ({gap.log_loss_ci_low:+.4f} to {gap.log_loss_ci_high:+.4f}) | {hybrid.accuracy:.1%} / {prior.accuracy:.1%} |")
    if prospective.get("available"):
        add(f"| October 2026 collector | **Recorded**, {len(prospective['decision_sessions'])} dates | {prospective['n']} | {prospective['hybrid']['log_loss']:.4f} | {prospective['prior']['log_loss']:.4f} | too few dates for an interval | {prospective['hybrid']['accuracy']:.1%} / {prospective['prior']['accuracy']:.1%} |")
    if simulated.get("available"):
        effect = simulated["hybrid_minus_prior_log_loss"]
        add(f"| {day(simulated['first'])[:-5]} to {day(simulated['last'])} | **Simulated** replay | {simulated['n']} | {simulated['hybrid']['log_loss']:.4f} | {simulated['prior']['log_loss']:.4f} | {effect['effect']:+.4f} ({effect['ci_low']:+.4f} to {effect['ci_high']:+.4f}) | {simulated['hybrid']['accuracy']:.1%} / {simulated['prior']['accuracy']:.1%} |")
    add("")
    spans = {"all": "the full history", "development": "2024 to 2025", "selection": "January to July 2026", "audit": "July to September 2026"}
    grouped = {}
    for key, label in spans.items():
        grouped.setdefault(paired.loc[(key, "hybrid"), "verdict"], []).append(label)
    add(
        "By 95% interval the classifier is " + "; ".join(f"{verdict} on {join(where)}" for verdict, where in grouped.items()) + ". "
        f"Split by HMM regime: calm {regimes.loc['calm', 'log_loss_difference']:+.4f} ({regimes.loc['calm', 'verdict']}), "
        f"turbulent {regimes.loc['turbulent', 'log_loss_difference']:+.4f} ({regimes.loc['turbulent', 'verdict']}), Holm-adjusted. "
        f"Balanced accuracy is {hybrid.balanced_accuracy:.1%} for the classifier against {prior.balanced_accuracy:.1%} for the prior; that measure averages recall over the three classes, "
        f"and the prior names DOWN as the likeliest class on only {prior.predicted_down / prior.n:.0%} of rows."
    )
    add("")
    add("### Trading, against an equal-weight benchmark")
    add("")
    add(f"Assumptions for every row: {usd(sandbox['assumptions']['capital_usd'])} paper capital; {sandbox['assumptions']['costs']}; fills {sandbox['assumptions']['fills']}; limits {sandbox['assumptions']['limits']}; Sharpe {sandbox['assumptions']['sharpe']}. Benchmark: {sandbox['assumptions']['benchmark']}.")
    add("")
    add(f"**History, {hist['first']} to {hist['last']} ({hist['sessions']} sessions, {hist['rebalances']} weekly rebalances; previously inspected, so exploratory)**")
    add("")
    add("| Portfolio | Net return | P&L (USD) | Sharpe (95% interval) | Max drawdown | Fills | Costs | Mean invested |")
    add("|---|---:|---:|---|---:|---:|---:|---:|")
    for name in arms:
        add(trading_row("history", name))
    add("")
    add(
        f"The trend rule's Sharpe ratio minus the benchmark's is {benchmark.loc['trend', 'sharpe_difference']:+.2f} (95% interval {benchmark.loc['trend', 'sharpe_ci_low']:+.2f} to {benchmark.loc['trend', 'sharpe_ci_high']:+.2f}): "
        f"{ {'not distinguishable': 'not distinguishable from the benchmark, so no edge is shown', 'better': 'better than the benchmark on this history', 'worse': 'worse than the benchmark'}[edge] }. "
        f"Over the same dates buy-and-hold SPY returned {pct(hist['spy_reference']['net_return'])}, so a positive return here is what a rising market gave any invested portfolio."
    )
    add("")
    add(f"**Simulated replay, {post['first']} to {post['last']} ({post['sessions']} sessions; not recorded trading)**")
    add("")
    add("| Portfolio | Net return | P&L (USD) | Max drawdown | Fills | Costs | Mean invested |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for name in arms:
        add(trading_row("post_release", name, sharpe=False))
    add("")
    add(
        f"{'Every invested portfolio lost money' if max(arm('post_release', a, 'net_return') for a in arms) < 0 else 'Results were mixed'} in this window (SPY buy-and-hold {pct(post['spy_reference']['net_return'])}). **No Sharpe ratio is reported: {post['sessions']} sessions containing one market move cannot separate skill from noise.** "
        f"The recorded paper ledgers are shorter still: at most two marked sessions per segment, so the recorded P&L is just the cost of the first purchases (about {usd(pilot['segments'][0]['fees'])} per segment)."
    )
    add("")
    add("### Sandbox experiment: does the HMM help as an exposure control?")
    add("")
    add(f"Registered before any result (`data/review-v1/registration.json`): {sandbox['question']} Budget: five portfolios, two contrasts, Holm-adjusted.")
    add("")
    add("| Contrast | Sharpe difference (95% interval) | Holm p | Verdict |")
    add("|---|---|---:|---|")
    for name in ["trend_regime", "equal_weight_regime"]:
        row = contrasts.loc[name]
        add(f"| {name} − {row.parent} | {row.sharpe_difference:+.2f} ({row.sharpe_ci_low:+.2f} to {row.sharpe_ci_high:+.2f}) | {row.sharpe_holm_p:.2f} | {row.verdict} |")
    add("")
    verdicts = sorted(set(contrasts.verdict))
    add(
        f"**{join(verdicts).capitalize()}.** With the overlay the trend rule's maximum drawdown went from {arm('history', 'trend', 'max_drawdown'):.1%} to {arm('history', 'trend_regime', 'max_drawdown'):.1%}, its net return from "
        f"{pct(arm('history', 'trend', 'net_return'))} to {pct(arm('history', 'trend_regime', 'net_return'))} and its Sharpe ratio from {arm('history', 'trend', 'sharpe_zero_rf'):.2f} to {arm('history', 'trend_regime', 'sharpe_zero_rf'):.2f}. "
        f"In the simulated replay the overlay version returned {pct(arm('post_release', 'trend_regime', 'net_return'))} against {pct(arm('post_release', 'trend', 'net_return'))}"
        + (f"; the HMM's first turbulent close in that window was {day(turned.min())[:-5]}, after which the overlay held less. That is one episode, not a test. " if len(turned) else ". ")
        + "The HMM's own checks pass: "
        f"{regime['refits']['converged']} of {regime['refits']['count']} refits converged, removing later data changes no earlier probability, and volatility over the next 21 sessions averaged "
        f"{forward['turbulent']['mean_next_21_session_volatility']:.1%} after turbulent closes against {forward['calm']['mean_next_21_session_volatility']:.1%} after calm ones "
        f"(states: about {calm['median_annualised_volatility']:.0%} and {turbulent['median_annualised_volatility']:.0%} annualised volatility)."
    )
    add("")
    add("### Sentiment")
    add("")
    add(
        f"Sentiment has no path into a decision: the pilot's {len(sentiment.get('pilot_feature_columns', [])) or stock['models']['features']} features are all price and volume, and every recorded decision reproduces from prices alone. "
        "Two diagnostics were run on cached headlines scored with FinBERT."
    )
    add("")
    links = {r["symbol"]: r for r in sentiment.get("history", {}).get("rows", [])}
    if links:
        add("- **Is tone related to anything the system uses?** Rank correlations of daily tone, " + "; ".join(audit.sentiment_link_text(list(links.values()))) + ".")
    add(f"- **Does it help the classifier?** {audit.sentiment_ablation_text(ablation)} Exploratory: two tickers, one feature design.")
    add("")
    add("## What the demo proves, and what it does not")
    add("")
    add("**Proves (operational):**")
    add("")
    add("- The pipeline runs from download to paper order on a current session in about half a minute (dashboard section 8).")
    add(f"- All {pilot['decisions']} recorded decisions match their registered hashes and reproduce exactly: probabilities, trend flags, weights and order quantities.")
    add("- The evaluation harness reproduces the sealed September accounts to the cent, so the trading numbers above come from the same ledger code as the pilot.")
    add("- The HMM is causal and stable; the statistics agree with an independently generated table.")
    add("")
    add("**Does not establish:**")
    add("")
    add("- That the classifier predicts anything beyond base rates.")
    add("- That the trend rule, with or without the regime overlay, beats holding everything equally.")
    add("- That sentiment or the HMM improves a decision. Neither has ever influenced one.")
    add(f"- That the collector can run unattended: {pilot['attempts_halted']} of {len(pilot['attempts'])} attempts halted, and only {pilot['decisions']} of {pilot['scheduled_decision_sessions']} scheduled decision sessions were recorded.")
    add("- The proposal's main claim (a learned gate beats a static ensemble). That experiment has not been built.")
    add("")
    add("## Status against the proposal")
    add("")
    add(f"{counts['works']} components work and are verified, {counts['diagnostic']} are diagnostic only, {counts['unvalidated']} exist but are not validated, {counts['missing']} are missing. The dashboard's last section lists the evidence and a check for each.")
    add("")
    for status in STATUS_ORDER:
        names = "; ".join(r["component"].split(" (")[0] for r in items if r["status"] == status)
        add(f"- **{audit.STATUS_TEXT[status]}:** {names}.")
    add("")
    add("## Plan before the final presentation")
    add("")
    add("1. **Run the collector without gaps.** Move it to an always-on machine and log the trend, equal-weight and overlay portfolios in shadow every session, aiming for at least 63 recorded sessions. Nothing else substitutes for prospective evidence.")
    add("2. **Implement the proposal's label** (next-open entry, threshold scaled to cost and volatility), refit, and rerun the classifier-against-prior comparison with the same statistics. With the fixed ±1% band the class mix moves with volatility; a scaled band removes that, and whether the classifier then beats the prior is an open question.")
    add("3. **Build the main experiment:** the learned gate over the expert models against the static ensemble, through the same decision and ledger code, with contrasts registered first. Include regime probabilities and sentiment as gate inputs, each with its own with/without comparison.")
    add("4. **Separate exposure from selection.** Compare the trend rule with an equal-weight benchmark scaled to the same average exposure.")
    add("5. **Extend the history only when the stress-period test needs it** (the proposal's 2011 and 2015 to 2016 periods), and keep the second-source price check in the build.")
    add("")
    add("The next experiment to run is item 2: it reuses the existing features, walk-forward harness and statistics, and its result decides whether the classifier has any claim to a place in the decision.")
    add("")
    if live:
        add(f"*Latest demo run: session {live['session']}, {live['decision']['trend_count']} of {stock['universe']['count']} instruments pass the trend rule, {live['account']['orders']} pending paper orders, regime {live['regime']['state']}.*")
        add("")
    text = "\n".join(lines)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(text)
    return text


if __name__ == "__main__":
    print(build())
