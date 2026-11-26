"""One permitted gate change per research method and review year."""

import numpy as np
import pandas as pd

from src.models.gating import GatingModel
from .evaluate import replay

SETTINGS = (0.1, 0.3, 1.0, 3.0, 10.0)


def sharpe(r):
    sd = np.std(r, ddof=1)
    return np.sqrt(252) * np.mean(r) / sd if sd > 1e-12 else 0.0


def losses(r):
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    es = -np.mean(np.sort(r)[: max(1, int(np.ceil(0.05 * len(r))))])
    return max(0.0, es), np.max(1 - wealth / np.maximum.accumulate(wealth))


def review(cache, rows, stack, correct, context, gate, calibration, year, horizon):
    table = {
        name: round(float(v), 5)
        for name, v in zip(calibration, correct[rows.index.year == year - 2].mean(0))
    }
    # Calibration observations are all mature before the proposal's review year.
    from .hosted import propose as hosted_proposal

    hosted = hosted_proposal(cache, table, year, horizon)
    llm, prompt = hosted["l2"], hosted["prompt"]
    rng = np.random.default_rng(year + 28)
    proposals = {
        "LLM review": llm,
        "Random review": float(rng.choice(SETTINGS)),
        "Conventional review": 3.0,
    }
    middle = pd.Timestamp(year=year - 1, month=7, day=1)
    training = gate & (rows.endpoint < middle)
    runs = {}
    for setting in {1.0, *proposals.values()}:
        model = GatingModel(n_experts=stack.shape[1], l2=setting).fit(
            context[training], correct[training]
        )
        segments = []
        for month in (7, 9, 11):
            low = pd.Timestamp(year=year - 1, month=month, day=1)
            end = low + pd.DateOffset(months=2)
            eligible = gate & (rows.index >= low) & (rows.endpoint < end)
            dates = rows.index[eligible].unique().sort_values()
            eligible &= rows.index >= dates[5]
            w = model.predict_weights(context[eligible])
            p = (w[:, :, None] * stack[eligible]).sum(1)
            trial = (
                rows[eligible]
                .assign(down=p[:, 0], up=p[:, 2])
                .rename_axis("decision")
                .reset_index()
            )
            segments.append(replay(trial, "trial"))
        runs[setting] = segments
    receipts = []
    selected = {}
    baseline = runs[1.0]
    for name, setting in proposals.items():
        candidate = runs[setting]
        differences = [
            sharpe(a.net_return) - sharpe(b.net_return)
            for a, b in zip(candidate, baseline)
        ]
        r = pd.concat(candidate).net_return.values
        base = pd.concat(baseline).net_return.values
        # Common block draws for every candidate, separately within each window.
        rng = np.random.default_rng(year)
        bounds = []
        for _ in range(500):
            indices = []
            offset = 0
            for window in candidate:
                n = len(window)
                starts = rng.integers(n, size=(n + 4) // 5)
                indices.extend(
                    offset + ((starts[:, None] + np.arange(5)) % n).ravel()[:n]
                )
                offset += n
            bounds.append(sharpe(r[indices]) - sharpe(base[indices]))
        lower = float(np.quantile(bounds, 0.05))
        es, dd = losses(r)
        base_es, base_dd = losses(base)
        trades = int(pd.concat(candidate).trades.sum())
        accepted = bool(
            min(differences) > 0
            and lower > 0
            and es <= 1.05 * base_es
            and dd <= 1.05 * base_dd
            and trades >= 30
        )
        selected[name] = setting if accepted else 1.0
        receipts.append(
            dict(
                year=year,
                method=name,
                proposed_l2=setting,
                accepted=accepted,
                next_l2=selected[name],
                trades=trades,
                window_sharpe_differences=differences,
                lower_90pct=lower,
                expected_shortfall=es,
                drawdown=dd,
                baseline_es=base_es,
                baseline_drawdown=base_dd,
                proposal_prompt=prompt if name == "LLM review" else None,
                agent=hosted if name == "LLM review" else None,
            )
        )
    return selected, receipts
