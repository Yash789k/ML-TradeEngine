import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss

from src.features.regime import fit_hmm
from src.models.gating import GatingModel
from .experts import predict_experts
from .panel import NUMERIC, TEXT


def observations(frames):
    blocks, rows = [], []
    for frame in frames.values():
        frame = frame.dropna().copy()
        windows = np.lib.stride_tricks.sliding_window_view(
            frame[NUMERIC + TEXT + list(frame.filter(like="jkp_"))].values, 20, axis=0
        )
        blocks.append(windows.transpose(0, 2, 1))
        rows.append(frame.iloc[19:])
    return np.concatenate(blocks).astype("float32"), pd.concat(rows)


def market_states(market, gate_year):
    inputs = pd.DataFrame(
        {"log_return": np.log(market.close).diff(), "realized_vol_21": market.vol}
    ).dropna()
    history = inputs.loc[inputs.index.year < gate_year].tail(504)
    model = fit_hmm(history, covariance_type="diag", random_state=28)
    data = inputs.loc[history.index[0] :]
    emissions = model._compute_log_likelihood(data.values / model.feature_scale_)
    state = model.startprob_
    result = []
    for emission in emissions:
        state = softmax(np.log(state.clip(1e-30)) + emission)
        result.append(state)
        state = state @ model.transmat_
    return pd.DataFrame(
        np.asarray(result)[:, np.argsort(model.means_[:, 0])], index=data.index
    )


def fold(windows, panel, market, year, cache, horizon, artifact_dir, device="cpu"):
    years = panel.index.year
    train = (years < year - 2) & (panel.endpoint.dt.year < year - 2)
    eligible = np.zeros(len(years), dtype=bool)
    dates = np.sort(panel.index.unique())
    for part in range(year - 2, year + 1):
        sessions = dates[pd.DatetimeIndex(dates).year == part]
        eligible |= (
            (years == part)
            & (panel.index >= sessions[5])
            & (panel.endpoint.dt.year == part)
        )
    y = panel.label.to_numpy()
    m = windows[train, -1].mean(0)
    s = windows[train, -1].std(0).clip(1e-6)
    predictions = predict_experts(
        ((windows - m) / s).clip(-10, 10), y, train, eligible, artifact_dir, device
    )
    rows = panel.loc[eligible].copy()
    predictions["Qwen statement"] = rows[["qdown", "qflat", "qup"]].values
    predictions["OpenAI asset digest"] = rows[["mdown", "mflat", "mup"]].values
    predictions["OpenAI Decisions"] = rows[["odown", "oflat", "oup"]].values
    predictions["Granite MoE statement"] = rows[["gdown", "gflat", "gup"]].values
    cal, gate, test = [rows.index.year == part for part in range(year - 2, year + 1)]
    calibrated, temperatures = {}, {}
    for name, probs in predictions.items():
        logits = np.log(np.clip(probs, 1e-12, 1))

        def loss(temp):
            return log_loss(
                rows.label[cal], softmax(logits[cal] / temp, axis=1), labels=[0, 1, 2]
            )

        temp = minimize_scalar(loss, bounds=(0.5, 5), method="bounded").x
        temperatures[name] = float(temp)
        calibrated[name] = softmax(logits / temp, axis=1)
    stack = np.stack(list(calibrated.values()), axis=1)
    forecast = {name: p[test] for name, p in calibrated.items()}
    forecast["Market experts vote"] = np.mean(
        [
            p[test]
            for name, p in calibrated.items()
            if name
            not in (
                "OpenAI asset digest",
                "OpenAI Decisions",
                "Qwen statement",
                "Granite MoE statement",
                "FinBERT logistic",
                "JKP factors",
            )
        ],
        axis=0,
    )
    forecast["Uniform vote"] = stack[test].mean(1)
    prior = np.bincount(y[train], minlength=3) / train.sum()
    forecast["Training prior"] = np.tile(prior, (test.sum(), 1))
    correct = stack.argmax(-1) == rows.label.values[:, None]
    context = rows[["vol", "range", "drawdown", "market"]].copy()
    # Feedback is usable only after its opening-price outcome has matured.
    memory = pd.DataFrame(correct.astype(float), index=rows.endpoint)
    memory = memory.groupby(level=0).mean().rolling(63, min_periods=1).mean()
    aligned = memory.reindex(rows.index - pd.Timedelta(days=1), method="ffill").fillna(
        1 / 3
    )
    context[[f"accuracy_{i}" for i in range(stack.shape[1])]] = aligned.to_numpy()
    context[["fed_positive", "headline_tone", "annual_report_tone"]] = rows[
        ["positive", "news_tone", "filing_tone"]
    ].values
    regimes = market_states(market, year - 1)
    hmm = regimes.reindex(rows.index).to_numpy()
    for name in ("Gate without HMM", "HMM gate"):
        if name == "HMM gate":
            context[["bear", "middle", "bull"]] = hmm
        model = GatingModel(n_experts=stack.shape[1]).fit(context[gate], correct[gate])
        weights = model.predict_weights(context[test])
        forecast[name] = (weights[:, :, None] * stack[test]).sum(1)
    rows["cap"] = np.where((rows.drawdown < -0.10) | (rows.vol > 0.035), 0.075, 0.15)
    model.save(str(artifact_dir / "gate.joblib"))
    regime = regimes.idxmax(axis=1)
    stable = regime.eq(regime.shift()) & regime.eq(regime.shift(2))
    rows["threshold"] = np.where(stable.reindex(rows.index).fillna(False), 0.05, 0.15)
    rows["regime"] = regime.reindex(rows.index).to_numpy()
    output = rows.loc[
        test,
        [
            "symbol",
            "label",
            "entry",
            "exit",
            "vol",
            "mark",
            "endpoint",
            "close",
            "atr",
            "volume",
            "cap",
            "threshold",
            "r21",
            "regime",
        ],
    ]
    example_index = np.flatnonzero(test)[0]
    np.savez_compressed(
        artifact_dir / "example.npz",
        window=windows[eligible][example_index],
        center=m,
        scale=s,
        context=context.iloc[example_index].to_numpy(),
        language=np.stack(
            [
                predictions[k][example_index]
                for k in (
                    "Qwen statement",
                    "OpenAI asset digest",
                    "OpenAI Decisions",
                    "Granite MoE statement",
                )
            ]
        ),
    )
    manifest = dict(
        experts=list(calibrated),
        temperature=temperatures,
        class_order=["DOWN", "FLAT", "UP"],
        symbol=output.symbol.iloc[0],
        decision=str(output.index[0].date()),
        expected=forecast["HMM gate"][0].tolist(),
        fit_endpoint=str(panel.loc[train, "endpoint"].max().date()),
        calibration_endpoint=str(rows.loc[cal, "endpoint"].max().date()),
        gate_endpoint=str(rows.loc[gate, "endpoint"].max().date()),
    )
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    from .adaptation import review

    selected, receipts = review(
        cache, rows, stack, correct, context, gate, calibrated, year, horizon
    )
    for name, strength in selected.items():
        if strength == 1.0:
            forecast[name] = forecast["HMM gate"]
        else:
            fitted = GatingModel(n_experts=stack.shape[1], l2=strength).fit(
                context[gate], correct[gate]
            )
            forecast[name] = (
                fitted.predict_weights(context[test])[:, :, None] * stack[test]
            ).sum(1)
    return pd.concat(
        [
            output.assign(model=name, down=p[:, 0], flat=p[:, 1], up=p[:, 2])
            for name, p in forecast.items()
        ]
    ).rename_axis("decision").reset_index(), receipts
