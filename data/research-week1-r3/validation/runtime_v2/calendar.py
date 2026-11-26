"""Session, freshness and observed-bar contracts for US equity research."""

from functools import lru_cache
import numpy as np
import pandas as pd
import exchange_calendars as xcals


@lru_cache(maxsize=16)
def calendar(start="2000-01-01", end="2035-12-31"):
    return xcals.get_calendar("XNYS", start=start, end=end)


def session_closes(index):
    dates = pd.DatetimeIndex(index).tz_localize(None).normalize()
    cal = calendar()
    valid = dates.isin(cal.sessions)
    closes = pd.DatetimeIndex(
        [cal.session_close(d) if ok else pd.NaT for d, ok in zip(dates, valid)],
        tz="UTC",
    )
    return closes


def observed_equity_bars(df, now=None, require_fresh=False):
    if df.empty or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV requires nonempty, ordered, unique observations")
    required = ["Open", "High", "Low", "Close", "Volume"]
    if set(required) - set(df):
        raise ValueError(f"Missing OHLCV columns: {set(required)-set(df)}")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    closes = session_closes(df.index)
    out = df.loc[(closes <= now) & closes.notna()].copy()
    # Legacy weekday padding was identifiable by zero volume; it is not a fill.
    out = out.loc[out.Volume > 0]
    if out.empty or not np.isfinite(out[required].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite completed OHLCV bars")
    if (out[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Nonpositive equity prices")
    if (out.High < out[["Open", "Close", "Low"]].max(axis=1)).any() or (
        out.Low > out[["Open", "Close", "High"]].min(axis=1)
    ).any():
        raise ValueError("Invalid OHLC price ranges")
    expected = calendar().sessions_in_range(out.index[0].date(), out.index[-1].date())
    if not out.index.tz_localize(None).normalize().equals(expected):
        raise ValueError("Missing exchange sessions in observed OHLCV history")
    if require_fresh:
        cal = calendar()
        sessions = cal.sessions_in_range(
            (now - pd.Timedelta(days=10)).date(), now.date()
        )
        completed = [s for s in sessions if cal.session_close(s) <= now]
        if not completed or out.index[-1].date() != completed[-1].date():
            raise ValueError(f"Stale OHLCV: last completed observation {out.index[-1]}")
    out.attrs["calendar"] = "XNYS"
    return out
