import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

START = 10_000.0
COST = 0.0005


def replay(group, policy, dynamic=False, blotter=None, exposure=None, cost=COST):
    rows = []
    capital = START
    for _, segment in group.groupby(group.decision.dt.year):
        symbols = sorted(segment.symbol.unique())
        dates = sorted(segment.decision.unique())
        panel = segment.set_index(["decision", "symbol"]).reindex(
            pd.MultiIndex.from_product([dates, symbols])
        )
        n = len(symbols)

        def matrix(column):
            return panel[column].to_numpy().reshape(-1, n)

        entry, exit_price, vols = (matrix(name) for name in ("entry", "exit", "vol"))
        if not (
            np.isfinite(entry).all()
            and np.isfinite(exit_price).all()
            and (entry > 0).all()
        ):
            raise ValueError("Incomplete execution panel")
        scores = matrix("up") - matrix("down")
        mark = matrix("mark")[:, 0]
        if dynamic:
            closes, atr, volume, caps, thresholds = (
                matrix(c) for c in ("close", "atr", "volume", "cap", "threshold")
            )
            sectors = {
                "Technology": ("AAPL", "MSFT", "NVDA", "XLK"),
                "Finance": ("BAC", "GS", "JPM", "XLF"),
                "Communication": ("META", "GOOGL"),
                "Consumer": ("AMZN", "TSLA"),
                "Broad equity": ("SPY", "QQQ", "IWM", "DIA"),
            }
        shares = np.zeros(n)
        cash = peak = capital
        stops = np.zeros(n)
        blocked_until = 0
        for i, date in enumerate(dates):
            value = shares * entry[i]
            equity = cash + value.sum()
            weights = np.zeros(n)
            if exposure is not None:
                weights[symbols.index("SPY")] = exposure.loc[date]
            elif policy == "Equal weight":
                weights[:] = 1 / n
            elif policy == "SPY buy and hold":
                weights[symbols.index("SPY")] = 1
            elif policy != "Cash":
                qualified = scores[i] > (thresholds[i] if dynamic else 0.05)
                if dynamic:
                    account_close = cash + (shares * closes[i]).sum()
                    peak = max(peak, account_close)
                    if i >= blocked_until and account_close < 0.9 * peak:
                        blocked_until = i + 5
                        peak = account_close
                    qualified &= (closes[i] > stops) & (i >= blocked_until)
                chosen = np.flatnonzero(qualified)
                chosen = chosen[np.argsort(-scores[i, chosen], kind="stable")[:5]]
                inverse = 1 / vols[i, chosen]
                if chosen.size:
                    weights[chosen] = np.minimum(
                        inverse / inverse.sum(), caps[i, chosen] if dynamic else 0.15
                    )
                if dynamic:
                    for members in sectors.values():
                        idx = [j for j, s in enumerate(symbols) if s in members]
                        gross = weights[idx].sum()
                        if gross > 0.30:
                            weights[idx] *= 0.30 / gross
            target_nav = equity
            for _ in range(6):
                target_nav = equity - cost * np.abs(weights * target_nav - value).sum()
            orders = weights * target_nav - value
            if policy == "SPY buy and hold" and i:
                orders[:] = 0
            if dynamic:
                capacity = 0.01 * volume[i] * closes[i]
                orders = np.clip(orders, -capacity, capacity)
                # Spend only settled cash plus actual sale proceeds.
                buys = orders > 0
                available = cash - orders[~buys].sum() * (1 - cost)
                buy_cost = orders[buys].sum() * (1 + cost)
                if buy_cost > available and buy_cost:
                    orders[buys] *= max(0.0, available) / buy_cost
            fee = cost * np.abs(orders).sum()
            turnover = np.abs(orders).sum()
            cash -= orders.sum() + fee
            shares += orders / entry[i]
            if dynamic:
                stops = np.where(
                    shares > 1e-9, np.maximum(stops, closes[i] - 2.5 * atr[i]), 0
                )
            deployed = (shares * entry[i]).sum() / (equity - fee)
            end_value = shares * exit_price[i]
            trades = (np.abs(orders) > 1e-6).sum()
            if blotter is not None:
                for j in np.flatnonzero(np.abs(orders) > 1e-6):
                    blotter.append(
                        [
                            date,
                            symbols[j],
                            "next open",
                            orders[j] / entry[i, j],
                            entry[i, j],
                            abs(orders[j]) * cost,
                        ]
                    )
            if i == len(dates) - 1:
                liquidation = end_value.sum()
                if blotter is not None:
                    for j in np.flatnonzero(shares > 1e-9):
                        blotter.append(
                            [
                                mark[i],
                                symbols[j],
                                "fold exit",
                                -shares[j],
                                exit_price[i, j],
                                end_value[j] * cost,
                            ]
                        )
                cash += liquidation * (1 - cost)
                trades += (shares > 1e-9).sum()
                fee += liquidation * cost
                turnover += liquidation
                end_value[:] = 0
            capital = cash + end_value.sum()
            if cash < -1e-6 or capital <= 0 or (shares < -1e-9).any():
                raise RuntimeError("Account reconciliation failed")
            rows.append(
                dict(
                    model=policy,
                    decision=date,
                    mark=mark[i],
                    equity=capital,
                    exposure=deployed,
                    net_return=capital / equity - 1,
                    fees=fee,
                    turnover=turnover / equity,
                    trades=trades,
                )
            )
    return pd.DataFrame(rows)


def summarize(pred):
    metrics, ledgers = [], []
    for name, group in pred.groupby("model"):
        probabilities = group[["down", "flat", "up"]].to_numpy()
        if (
            not np.allclose(probabilities.sum(1), 1)
            or not np.isfinite(probabilities).all()
        ):
            raise ValueError(f"Invalid forecast from {name}")
        y = group.label.to_numpy()
        metrics.append(
            dict(
                model=name,
                observations=len(y),
                log_loss=log_loss(y, probabilities),
                accuracy=accuracy_score(y, probabilities.argmax(1)),
                macro_f1=f1_score(y, probabilities.argmax(1), average="macro"),
                brier=np.square(probabilities - np.eye(3)[y]).sum(1).mean(),
            )
        )
        ledgers.append(replay(group, name))
    common = pred.loc[pred.model == "Uniform vote"]
    for policy in ("Cash", "SPY buy and hold", "Equal weight"):
        ledgers.append(replay(common, policy))
    blotter = []
    ledgers.append(
        replay(
            pred.loc[pred.model == "HMM gate"],
            "HMM gate dynamic",
            dynamic=True,
            blotter=blotter,
        )
    )
    primary = next(frame for frame in ledgers if frame.model.iloc[0] == "HMM gate")
    ledgers.append(
        replay(
            common,
            "Exposure-matched SPY",
            exposure=primary.set_index("decision").exposure,
        )
    )
    momentum = common.assign(up=common.r21, down=0.0)
    ledgers.append(replay(momentum, "21-session momentum"))
    ledger = pd.concat(ledgers)
    perf = []
    for name, sub in ledger.groupby("model"):
        equity = np.r_[START, sub.equity]
        r = sub.net_return
        perf.append(
            dict(
                model=name,
                net_return=equity[-1] / START - 1,
                max_drawdown=(1 - equity / np.maximum.accumulate(equity)).max(),
                sharpe=(np.sqrt(252) * r.mean() / r.std() if r.std() else 0),
                mean_exposure=sub.exposure.mean(),
                trades=int(sub.trades.sum()),
                fees=sub.fees.sum(),
                turnover=sub.turnover.sum(),
            )
        )
    return (
        pd.DataFrame(metrics).merge(pd.DataFrame(perf), on="model", how="outer"),
        ledger,
        order_dates(
            pd.DataFrame(
                blotter, columns=["date", "symbol", "event", "units", "price", "fee"]
            )
        ),
    )


def order_dates(frame):
    import exchange_calendars as xcals

    dates = pd.to_datetime(frame.pop("date"))
    calendar = xcals.get_calendar(
        "XNYS",
        start=dates.min() - pd.Timedelta(days=12),
        end=dates.max() + pd.Timedelta(days=12),
    )
    mapping = {day: calendar.next_session(day) for day in dates.unique()}
    next_open = frame.event.eq("next open")
    frame.insert(0, "execution_date", dates.where(~next_open, dates.map(mapping)))
    frame.insert(0, "signal_date", dates.where(next_open))
    return frame


def interval(pred, ledger):
    names = ["HMM gate", "Uniform vote"]
    subset = pred[pred.model.isin(names)].copy()
    probs = subset[["down", "flat", "up"]].values
    subset["loss"] = -np.log(probs[np.arange(len(subset)), subset.label].clip(1e-12))
    losses = subset.groupby(["decision", "model"]).loss.mean().unstack()[names]
    returns = ledger.pivot(index="decision", columns="model", values="net_return")[
        names
    ]
    diff = (losses.iloc[:, 0] - losses.iloc[:, 1]).to_numpy()
    rng = np.random.default_rng(28)
    draws = []
    for _ in range(1000):
        # Preserve pairs, cross-asset dependence and each fold's boundaries.
        ids = []
        for year in losses.index.year.unique():
            idx = np.flatnonzero(losses.index.year == year)
            start = rng.integers(len(idx), size=(len(idx) + 19) // 20)
            ids.extend(
                idx[((start[:, None] + np.arange(20)) % len(idx)).ravel()[: len(idx)]]
            )
        net = (1 + returns.iloc[ids]).prod()
        draws.append([diff[ids].mean(), net.iloc[0] - net.iloc[1]])
    return dict(
        zip(
            ["mean_log_loss_difference", "compounded_return_difference"],
            np.quantile(draws, [0.025, 0.975], axis=0).T.tolist(),
        )
    )
