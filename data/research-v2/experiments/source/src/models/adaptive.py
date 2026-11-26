"""Fixed, rolling, purged hybrid; every fitted input precedes the decision.

This module deliberately has no hyperparameter search. Calibration and the
expert gate use only a trailing, purged, already-observed calibration window.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AdaptiveConfig:
    version: int = 1
    horizon: int = 5
    threshold: float = 0.01
    window: int = 1008
    calibration: int = 126
    min_fit: int = 504
    retrain_every: int = 21
    seed: int = 42
    gate_improvement: float = 0.01


def features(raw, spy):
    c = raw.Close
    r = c.pct_change(fill_method=None)
    x = pd.DataFrame(index=raw.index)
    for n in [1, 5, 21, 63, 126, 252]:
        x[f"return_{n}"] = c.pct_change(n, fill_method=None)
    for n in [21, 63]:
        x[f"vol_{n}"] = r.rolling(n).std()
        x[f"relative_{n}"] = x[f"return_{n}"] - spy.Close.pct_change(
            n, fill_method=None
        ).reindex(x.index)
    for n in [20, 50, 200]:
        x[f"average_{n}"] = c / c.rolling(n).mean() - 1
    x["gap"] = raw.Open / c.shift(1) - 1
    x["range"] = (raw.High - raw.Low) / c
    x["volume_ratio"] = raw.Volume / raw.Volume.rolling(20).mean() - 1
    x["momentum_12_1"] = c.shift(21) / c.shift(252) - 1
    return x.replace([np.inf, -np.inf], np.nan)


def execution_labels(close, config=AdaptiveConfig()):
    # The next close is the first permissible execution; a label ends h+1 bars
    # after the information cutoff. Do not train on unavailable tail labels.
    r = close.shift(-(config.horizon + 1)) / close.shift(-1) - 1
    y = pd.Series(
        np.where(r > config.threshold, 2, np.where(r < -config.threshold, 0, 1)),
        index=close.index,
    )
    return y.where(r.notna(), -1).astype(int), r


def probabilities(model, x):
    p = np.full((len(x), 3), 1e-12)
    p[:, model.classes_.astype(int)] = model.predict_proba(x)
    return p / p.sum(axis=1, keepdims=True)


def temperature(p, value):
    return softmax(np.log(np.clip(p, 1e-12, 1)) / value, axis=1)


class AdaptiveModel:
    def __init__(self, config=AdaptiveConfig()):
        self.config = config

    def fit_at(self, x, close, decision):
        cfg = self.config
        if (
            not x.index.equals(close.index)
            or not x.index.is_unique
            or not x.index.is_monotonic_increasing
        ):
            raise ValueError("Ordered aligned features and prices required")
        decision = pd.Timestamp(decision)
        if decision not in x.index:
            raise ValueError("Decision must be an observed session")
        y, returns = execution_labels(close, cfg)
        endpoints = pd.Series(x.index, index=x.index).shift(-(cfg.horizon + 1))
        eligible = x.notna().all(axis=1) & (y >= 0) & (endpoints < decision)
        ids = np.flatnonzero(eligible.to_numpy())[-cfg.window :]
        if len(ids) < cfg.min_fit + cfg.calibration + cfg.horizon + 1:
            raise ValueError("Insufficient purged rolling training history")
        cal_ids = ids[-cfg.calibration :]
        fit_ids = ids[ids < cal_ids[0] - cfg.horizon - 1]
        if len(fit_ids) < cfg.min_fit or y.iloc[fit_ids].nunique() < 2:
            raise ValueError("Insufficient fitting rows/classes")
        self.columns = list(x.columns)
        self.linear = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=500, random_state=cfg.seed),
        )
        self.tree = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=7,
            l2_regularization=10,
            learning_rate=0.05,
            min_samples_leaf=40,
            early_stopping=False,
            random_state=cfg.seed,
        )
        xf, yf = x.iloc[fit_ids], y.iloc[fit_ids]
        self.linear.fit(xf, yf)
        self.tree.fit(xf, yf)
        self.prior = (np.bincount(yf, minlength=3) + 1) / (len(yf) + 3)
        self.class_returns = np.array(
            [
                float(returns.iloc[fit_ids][yf == c].mean()) if (yf == c).any() else 0
                for c in range(3)
            ]
        )
        xc, yc = x.iloc[cal_ids], y.iloc[cal_ids]
        raw = (probabilities(self.linear, xc) + probabilities(self.tree, xc)) / 2
        self.temperature = min(
            [0.75, 1, 1.5, 2, 3],
            key=lambda v: log_loss(yc, temperature(raw, v), labels=[0, 1, 2]),
        )
        self.calibration_loss = float(
            log_loss(yc, temperature(raw, self.temperature), labels=[0, 1, 2])
        )
        self.prior_loss = float(
            log_loss(yc, np.tile(self.prior, (len(yc), 1)), labels=[0, 1, 2])
        )
        self.gate = bool(self.calibration_loss + cfg.gate_improvement < self.prior_loss)
        self.metadata = dict(
            config=asdict(cfg),
            columns=self.columns,
            decision=str(decision),
            fit_start=str(x.index[fit_ids[0]]),
            fit_end=str(x.index[fit_ids[-1]]),
            fit_label_end=str(endpoints.iloc[fit_ids[-1]]),
            calibration_start=str(x.index[cal_ids[0]]),
            calibration_end=str(x.index[cal_ids[-1]]),
            latest_label_end=str(endpoints.iloc[cal_ids[-1]]),
            n_fit=len(fit_ids),
            n_calibration=len(cal_ids),
            temperature=self.temperature,
            calibration_log_loss=self.calibration_loss,
            prior_log_loss=self.prior_loss,
            gate=self.gate,
            training_hash=hashlib.sha256(
                pd.util.hash_pandas_object(
                    pd.concat(
                        [
                            x.iloc[ids],
                            y.iloc[ids].rename("label"),
                            returns.iloc[ids].rename("return"),
                        ],
                        axis=1,
                    ),
                    index=True,
                ).values.tobytes()
            ).hexdigest(),
        )
        return self

    def predict(self, x):
        if list(x.columns) != self.columns or not np.isfinite(x.to_numpy()).all():
            raise ValueError("Feature schema or finite-value validation failed")
        if (x.index < pd.Timestamp(self.metadata["decision"])).any():
            raise ValueError("Model cannot predict before its training decision")
        linear = probabilities(self.linear, x)
        tree = probabilities(self.tree, x)
        mixed = temperature((linear + tree) / 2, self.temperature)
        return dict(
            linear=linear,
            tree=tree,
            hybrid=mixed,
            prior=np.tile(self.prior, (len(x), 1)),
        )

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        joblib.dump(self, path / "model.joblib")
        manifest = dict(
            self.metadata,
            model_sha256=hashlib.sha256(
                (path / "model.joblib").read_bytes()
            ).hexdigest(),
        )
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path):
        # Trusted local artifacts only: joblib is executable serialization.
        path = Path(path)
        meta = json.loads((path / "manifest.json").read_text())
        if (
            hashlib.sha256((path / "model.joblib").read_bytes()).hexdigest()
            != meta["model_sha256"]
        ):
            raise ValueError("Adaptive model checksum mismatch")
        model = joblib.load(path / "model.joblib")
        if model.metadata != {k: v for k, v in meta.items() if k != "model_sha256"}:
            raise ValueError("Adaptive model metadata mismatch")
        return model


def walk_forward(raw, spy, start="2024-01-02", config=AdaptiveConfig()):
    x = features(raw, spy)
    y, returns = execution_labels(raw.Close, config)
    ids = np.flatnonzero(x.index >= pd.Timestamp(start, tz="UTC"))
    output, folds = [], []
    last_fit = -config.retrain_every
    for i in ids:
        if i - last_fit >= config.retrain_every:
            model = AdaptiveModel(config).fit_at(x, raw.Close, x.index[i])
            last_fit = i
            folds.append(model.metadata)
        ps = model.predict(x.iloc[[i]])
        row = dict(
            date=x.index[i],
            label=int(y.iloc[i]),
            forward_return=returns.iloc[i],
            fit_decision=model.metadata["decision"],
            latest_label_end=model.metadata["latest_label_end"],
            gate=model.gate,
            expected_return=float(ps["hybrid"][0] @ model.class_returns),
        )
        for name, p in ps.items():
            row.update({f"{name}_{c}": float(p[0, c]) for c in range(3)})
        output.append(row)
    return pd.DataFrame(output).set_index("date"), folds, model
