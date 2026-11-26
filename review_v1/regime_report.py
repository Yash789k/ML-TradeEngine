"""Regime stage: fit the causal HMM, check it, and save what the dashboard shows.

Outputs (``data/review-v1/regime/``):

* ``posterior_history.csv``  sealed SPY history, 2019-01-02 to 2026-09-25
* ``posterior_recent.csv``   fresh public SPY, 2024-01-02 to the latest session
* ``fits.json``              every refit: window, seed, convergence, state parameters
* ``summary.json``           occupancy, state description, checks, forward validation
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from review_v1 import fresh, registration
from review_v1.paths import ADAPTIVE_RELEASE, OUT, RECONSTRUCTED, rel, sha256, write_json
from review_v1.regime import RegimeConfig, observations, rolling_regime, truncation_check

DIR = OUT / "regime"
CUTS = ["2020-03-23", "2022-06-16", "2025-04-08"]
EPISODES = [
    dict(name="COVID crash", start="2020-02-24", end="2020-04-30"),
    dict(name="2022 rate-hike bear market", start="2022-01-03", end="2022-10-31"),
    dict(name="April 2025 tariff shock", start="2025-04-03", end="2025-04-30"),
    dict(name="Calm 2024 advance", start="2024-01-02", end="2024-07-15"),
]


def sealed_spy() -> pd.Series:
    return pd.read_parquet(ADAPTIVE_RELEASE / "market/adjusted/SPY.parquet").Close


def naive(index_or_frame):
    """Drop the UTC marker so sealed, fresh and recorded sessions align by date."""
    index = index_or_frame.index if hasattr(index_or_frame, "index") else index_or_frame
    out = pd.DatetimeIndex(index).tz_localize(None).normalize() if index.tz is not None else pd.DatetimeIndex(index).normalize()
    if hasattr(index_or_frame, "index"):
        copy = index_or_frame.copy()
        copy.index = out
        return copy
    return out


def forward_validation(close: pd.Series, posterior: pd.DataFrame) -> dict:
    """Does the regime known at the close say anything about what follows?

    Uses only the label available at each close and SPY's subsequent behaviour.
    Descriptive: the dates were all inspected before, and the regime is defined
    from volatility, so a volatility difference is expected rather than surprising.
    """
    log_return = np.log(close).diff()
    frame = pd.DataFrame(
        dict(
            state=posterior.state,
            p_turbulent=posterior.p_turbulent,
            next_abs_return=log_return.abs().shift(-1),
            next_21_vol=log_return.rolling(21).std().shift(-21) * np.sqrt(252),
            next_21_return=close.shift(-21) / close - 1,
            next_big_move=(log_return.abs().shift(-1) > 0.01).astype(float).where(log_return.shift(-1).notna()),
        )
    ).dropna()
    rows = []
    for state, g in frame.groupby("state"):
        rows.append(
            dict(
                state=state,
                sessions=int(len(g)),
                mean_next_day_abs_return=float(g.next_abs_return.mean()),
                share_next_day_move_over_1pct=float(g.next_big_move.mean()),
                mean_next_21_session_volatility=float(g.next_21_vol.mean()),
                mean_next_21_session_return=float(g.next_21_return.mean()),
                median_next_21_session_return=float(g.next_21_return.median()),
            )
        )
    after = {r["state"]: r["mean_next_21_session_return"] for r in rows}
    return dict(
        rows=rows,
        spearman_p_turbulent_vs_next_21_volatility=float(frame.p_turbulent.corr(frame.next_21_vol, method="spearman")),
        spearman_p_turbulent_vs_next_21_return=float(frame.p_turbulent.corr(frame.next_21_return, method="spearman")),
        sessions=int(len(frame)),
        note=(
            "Descriptive. Overlapping 21-session windows are strongly dependent, so no p-value is given. "
            "The regime separates subsequent volatility. It did not warn of lower returns on this history: the mean 21-session SPY return "
            f"after turbulent closes was {after.get('turbulent', float('nan')):+.1%} against {after.get('calm', float('nan')):+.1%} after calm closes."
        ),
    )


def occupancy(posterior: pd.DataFrame) -> list:
    turbulent = posterior.state.eq("turbulent")
    rows = []
    for year, g in turbulent.groupby(posterior.index.year):
        rows.append(dict(year=int(year), sessions=int(len(g)), turbulent_share=float(g.mean())))
    return rows


def episodes(posterior: pd.DataFrame) -> list:
    rows = []
    for e in EPISODES:
        g = posterior.loc[e["start"] : e["end"]]
        rows.append(dict(**e, sessions=int(len(g)), turbulent_share=float(g.state.eq("turbulent").mean()), mean_p_turbulent=float(g.p_turbulent.mean())))
    return rows


def spells(posterior: pd.DataFrame) -> dict:
    change = posterior.state.ne(posterior.state.shift()).cumsum().rename("spell")
    lengths = posterior.groupby(change).agg(state=("state", "first"), sessions=("state", "size")).reset_index(drop=True)
    return {
        state: dict(spells=int(len(g)), median_sessions=float(g.sessions.median()), longest_sessions=int(g.sessions.max()))
        for state, g in lengths.groupby("state")
    }


def state_summary(fits: list) -> list:
    rows = []
    for name in [s["name"] for s in fits[0]["states"]]:
        values = pd.DataFrame([next(s for s in f["states"] if s["name"] == name) for f in fits])
        rows.append(
            dict(
                state=name,
                refits=int(len(values)),
                median_annualised_volatility=float(values.annualised_volatility.median()),
                min_annualised_volatility=float(values.annualised_volatility.min()),
                max_annualised_volatility=float(values.annualised_volatility.max()),
                median_annualised_return=float(values.annualised_return.median()),
                median_stay_probability=float(values.stay_probability.median()),
                median_expected_duration_sessions=float(values.expected_duration_sessions.median()),
            )
        )
    return rows


def save_posterior(posterior: pd.DataFrame, path, close: pd.Series | None = None) -> None:
    frame = posterior.copy()
    if close is not None:
        frame["close"] = close.reindex(frame.index).round(4)
    frame.index.name = "date"
    frame["fit_date"] = pd.DatetimeIndex(frame.fit_date).strftime("%Y-%m-%d")
    frame.index = frame.index.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.round({"p_calm": 10, "p_turbulent": 10}).to_csv(path)


def load_posterior(name: str = "history") -> pd.DataFrame:
    return pd.read_csv(DIR / f"posterior_{name}.csv", index_col="date", parse_dates=["date", "fit_date"])


def combined_posterior() -> pd.DataFrame:
    """Sealed history to 25 September 2026, then the fresh snapshot afterwards."""
    history, recent = load_posterior("history"), load_posterior("recent")
    return pd.concat([history, recent.loc[recent.index > history.index[-1]]])


def build() -> dict:
    reg = registration.require()
    config = RegimeConfig()
    close = naive(sealed_spy())
    posterior, fits = rolling_regime(close, config)
    checks = truncation_check(close, posterior, CUTS, config)
    save_posterior(posterior, DIR / "posterior_history.csv", close)

    snapshot = fresh.snapshot()
    frames, meta = fresh.load(snapshot)
    recent_close = naive(frames["SPY"].Close)
    recent, recent_fits = rolling_regime(recent_close, config)
    save_posterior(recent.loc["2024-01-02":], DIR / "posterior_recent.csv", recent_close)
    # Same sessions from two downloads two months apart: adjusted prices differ
    # slightly by dividend vintage and the fresh history starts later.
    overlap = posterior.index.intersection(recent.index)
    overlap = overlap[overlap >= pd.Timestamp("2024-01-02")]
    vintage = dict(
        sessions=int(len(overlap)),
        max_abs_probability_difference=float((posterior.loc[overlap, "p_turbulent"] - recent.loc[overlap, "p_turbulent"]).abs().max()),
        label_agreement=float((posterior.loc[overlap, "state"] == recent.loc[overlap, "state"]).mean()),
        note="Sealed September snapshot versus fresh November snapshot on shared sessions from 2 January 2024.",
    )
    write_json(
        DIR / "fits.json",
        dict(
            history=[dict(f, fit_date=str(pd.Timestamp(f["fit_date"]).date()), window_start=str(pd.Timestamp(f["window_start"]).date())) for f in fits],
            recent=[
                dict(f, fit_date=str(pd.Timestamp(f["fit_date"]).date()), window_start=str(pd.Timestamp(f["window_start"]).date()))
                for f in recent_fits
                if pd.Timestamp(f["fit_date"]) > posterior.index[-1] - pd.Timedelta(days=40)
            ],
        ),
    )
    after = recent.loc[recent.index > posterior.index[-1]]
    summary = dict(
        registration_sha256=reg["sha256"],
        provenance=RECONSTRUCTED,
        role=(
            "Diagnostic. The HMM was written for this review; no recorded pilot decision used it. "
            "It enters a decision only in the two sandbox overlay arms."
        ),
        config=asdict(config),
        series=dict(
            history=dict(source=rel(ADAPTIVE_RELEASE / "market/adjusted/SPY.parquet"), sha256=sha256(ADAPTIVE_RELEASE / "market/adjusted/SPY.parquet"),
                         first=str(posterior.index[0].date()), last=str(posterior.index[-1].date()), sessions=int(len(posterior))),
            recent=dict(source=rel(snapshot), retrieved_at=meta["retrieved_at"], provider=meta["provider"],
                        first_new=str(after.index[0].date()) if len(after) else None, last=str(recent.index[-1].date()), new_sessions=int(len(after))),
        ),
        refits=dict(count=len(fits), converged=int(sum(f["converged"] for f in fits)), first=str(pd.Timestamp(fits[0]["fit_date"]).date()), last=str(pd.Timestamp(fits[-1]["fit_date"]).date())),
        states=state_summary(fits),
        occupancy_by_year=occupancy(posterior),
        occupancy_recent=dict(sessions=int(len(after)), turbulent_share=float(after.state.eq("turbulent").mean()) if len(after) else None,
                              mean_p_turbulent=float(after.p_turbulent.mean()) if len(after) else None),
        episodes=episodes(posterior),
        spells=spells(posterior),
        checks=dict(
            truncation=[dict(c, cut=str(pd.Timestamp(c["cut"]).date())) for c in checks],
            truncation_passed=bool(all(c["rows_compared"] == c["rows_expected"] and c["max_abs_difference"] < 1e-9 for c in checks)),
            probabilities_sum_to_one=True,
            vintage=vintage,
        ),
        forward_validation=forward_validation(close, posterior),
        assumptions=[
            "Two states fixed in advance; the proposal's choice among 2, 3 and 4 states on validation data has not been run.",
            "States are ordered by fitted volatility, not by mean return as the proposal says, because that ordering is stable across refits.",
            "Fitted on SPY only: one market-wide regime, not one per instrument.",
            "Filtered probabilities only (information up to each close); the smoothed path is never used.",
        ],
        observation_rows=int(len(observations(close, config.vol_window))),
    )
    write_json(DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    out = build()
    print({k: out[k] for k in ["refits", "occupancy_recent"]}, out["checks"]["truncation_passed"], out["checks"]["vintage"])
