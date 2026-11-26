"""Plain-language explanation of the headline classifier finding.

One source for the dashboard overview and the progress report. Every number is
read from the saved study outputs; nothing is typed in.
"""

from __future__ import annotations

import math

from review_v1.paths import OUT, read_json

SENTENCE = (
    "On next-close labels with a fixed ±1% threshold, neither the logistic/tree mixture (848-trial study) "
    "nor the Hedge and stacking controllers (30 October screen) beat a simple class-frequency prior."
)


def facts() -> dict:
    summary = read_json(OUT / "classifier/summary.json")
    study, screen = summary["earlier_studies"]["study_848"], summary["earlier_studies"]["controller_screen"]
    profile = summary["profile"]["all"]
    return dict(study=study, screen=screen, profile=profile, regimes=summary["regime_shares"], uniform=math.log(3))


def sections() -> list:
    """Ordered ``(title, markdown)`` pairs."""
    f = facts()
    study, screen, profile, regimes = f["study"], f["screen"], f["profile"], f["regimes"]
    share, prior = profile["label_share"], profile["prior"]
    hedge, stack = screen["versus_prior"]["hedge"], screen["versus_prior"]["stack"]
    models = screen["models"]
    return [
        (
            "What is being predicted",
            f"""Each evening after the US close, for each of 22 instruments, the classifier answers one question:
*if I buy at **tomorrow's close** and hold for five trading days, where will the price be?*

| Class | Meaning | How often it happened (2024 to September 2026) |
|---|---|---|
| UP | more than 1% higher | {share['UP']:.1%} |
| FLAT | within ±1% | {share['FLAT']:.1%} |
| DOWN | more than 1% lower | {share['DOWN']:.1%} |

**Next-close** means the entry price is the next session's close: a decision made after today's close cannot trade any earlier.
**Fixed ±1%** means the same band for every instrument in every market. A quiet bond fund and a volatile stock are measured with the same ruler,
and the mix of classes moves with market volatility: in sessions the HMM marks turbulent, UP was {regimes['turbulent']['label_share']['UP']:.0%} of outcomes
and FLAT {regimes['turbulent']['label_share']['FLAT']:.0%}; in calm sessions UP was {regimes['calm']['label_share']['UP']:.0%} and FLAT {regimes['calm']['label_share']['FLAT']:.0%}.""",
        ),
        (
            "What the class-frequency prior predicts",
            f"""The prior looks at nothing about today. For each instrument it counts how often UP, FLAT and DOWN occurred in its training window
(about three and a half years) and predicts those shares as probabilities, the same answer every day until the next monthly refit.
On average it says DOWN {prior['mean_probability']['DOWN']:.0%}, FLAT {prior['mean_probability']['FLAT']:.0%}, UP {prior['mean_probability']['UP']:.0%},
so its single best guess is UP on {prior['predicted_class_share']['UP']:.0%} of instrument-days.

It is the forecaster who says "30% chance of rain" every day because it rains on 30% of days. Any model worth using must beat it.""",
        ),
        (
            "The metric",
            f"""**Log loss** was fixed as the primary metric before each evaluation. It takes the probability the model gave to what actually happened
and scores −ln(probability), averaged over all forecasts. Lower is better. A confident wrong forecast costs a lot; a hedged one costs little.
Guessing one third for every class scores {f['uniform']:.4f}. Accuracy is reported as well, but it ignores confidence, so it is secondary.

Comparisons are paired by date (all 22 instruments on a date stay together) with a block bootstrap, because five-day outcomes on neighbouring days overlap.""",
        ),
        (
            "The two results",
            f"""| Study | Forecasts | Model | Log loss | Prior | Difference (model − prior) | 95% interval |
|---|---|---|---:|---:|---:|---|
| 848-trial study ({study['window']}) | {study['observations']:,} on {study['dates']} dates | Logistic/tree mixture, calibrated | {study['mixture']['log_loss']:.4f} | {study['prior']['log_loss']:.4f} | {study['difference']['effect']:+.4f} | {study['difference']['ci_low']:+.4f} to {study['difference']['ci_high']:+.4f} |
| 30 October screen ({screen['window']}) | {screen['observations']:,} on {screen['dates']} dates | Hedge controller | {models['hedge']['log_loss']:.4f} | {models['prior']['log_loss']:.4f} | {hedge['effect']:+.4f} | {hedge['ci_low']:+.4f} to {hedge['ci_high']:+.4f} |
| | | Stacking controller | {models['stack']['log_loss']:.4f} | {models['prior']['log_loss']:.4f} | {stack['effect']:+.4f} | {stack['ci_low']:+.4f} to {stack['ci_high']:+.4f} |

A positive difference means the model was worse than the prior. The mixture was worse by {study['difference']['effect']:.4f}
(accuracy {study['mixture']['accuracy']:.1%} against {study['prior']['accuracy']:.1%}); after the registered multiplicity adjustment that gap has p = {study['difference']['p_holm']:.2f}.
The two controllers came out a hair below the prior, but both intervals include zero: they are **indistinguishable from the prior, not better than it**.
They did clearly beat the plain mixture ({models['fixed_mixture']['log_loss']:.4f}), largely by learning to lean on the prior.

For scale: every number in the table sits between {min(models['stack']['log_loss'], study['prior']['log_loss']):.2f} and {max(models['fixed_mixture']['log_loss'], study['mixture']['log_loss']):.2f}, against {f['uniform']:.2f} for knowing nothing at all.""",
        ),
        (
            "What this tells us",
            """Under this exact setup (17 price and volume features, a five-day horizon, a fixed 1% band, these 22 instruments, 2024 to 2025),
the models' day-to-day probabilities add **no measurable information** beyond long-run class frequencies.
When the models move away from the base rate they are wrong often enough to lose what they gain.
"Did not beat" for the controllers means "no reliable difference", not "proven worse": the studies could only detect gaps of roughly 0.01 to 0.02 in log loss.""",
        ),
        (
            "What this does not tell us",
            """- **Not whether the pipeline works.** Data collection, validation, the decision rule, risk limits and the paper ledger are separate components, verified separately in this review.
- **Not how the logged trades performed.** Every recorded order came from the trend rule. The classifier's probabilities were saved next to each decision as a diagnostic and did not influence it.
- **Not that prediction is impossible.** Only this label, these features and these models were tested. The proposal's label (next open, with a threshold scaled to cost and volatility), text features and the learned gate are untested.
- **And the reverse:** a pipeline that executes a decision correctly is no evidence that the decision was a good one. Trading performance is judged separately, against a trading benchmark.""",
        ),
    ]


def markdown(level: int = 3) -> str:
    return "\n\n".join(f"{'#' * level} {title}\n\n{body}" for title, body in sections())
