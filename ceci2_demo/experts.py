"""Shared, fixed-budget classifier families for the chronological comparison."""

import joblib
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from xgboost import XGBClassifier

from .panel import NUMERIC, TEXT


class SequenceExpert(nn.Module):
    def __init__(self, family, width):
        super().__init__()
        self.family = family
        if family in ("LSTM", "GRU"):
            self.encoder = getattr(nn, family)(width, 24, batch_first=True)
        elif family == "TCN":
            self.encoder = nn.Sequential(
                nn.Conv1d(width, 24, 3),
                nn.ReLU(),
                nn.Conv1d(24, 24, 3, dilation=2),
                nn.ReLU(),
            )
        else:
            self.patch = nn.Linear(5 * width, 24)
            self.position = nn.Parameter(torch.zeros(1, 4, 24))
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(24, 3, 48, dropout=0, batch_first=True),
                1,
                enable_nested_tensor=False,
            )
        self.head = nn.Linear(24, 3)

    def forward(self, x):
        if self.family in ("LSTM", "GRU"):
            h = self.encoder(x)[0][:, -1]
        elif self.family == "TCN":
            h = self.encoder(x.transpose(1, 2)).mean(-1)
        else:
            h = self.encoder(self.patch(x.reshape(len(x), 4, -1)) + self.position).mean(
                1
            )
        return self.head(h)


def predict_experts(x, y, train, later, artifacts, device="cpu"):
    artifacts.mkdir(parents=True, exist_ok=True)
    results = {}
    columns = len(NUMERIC)
    factories = {
        "Logistic": LogisticRegression(max_iter=400),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.06, n_jobs=1, random_state=28
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            num_leaves=12,
            learning_rate=0.06,
            verbosity=-1,
            n_jobs=1,
            random_state=28,
        ),
        "JKP factors": LogisticRegression(max_iter=400),
        "FinBERT logistic": LogisticRegression(max_iter=400),
    }
    for name, model in factories.items():
        values = x[:, -1, :columns]
        if name == "FinBERT logistic":
            values = x[:, -1, columns : columns + len(TEXT)]
        elif name == "JKP factors":
            values = x[:, -1, columns + len(TEXT) :]
        model.fit(values[train], y[train])
        joblib.dump(model, artifacts / f"{name}.joblib")
        results[name] = model.predict_proba(values[later])
    a = torch.from_numpy(x[train, :, :columns].copy()).to(device)
    b = torch.from_numpy(x[later, :, :columns].copy()).to(device)
    labels = torch.tensor(y[train], dtype=torch.long, device=device)
    for name in ("LSTM", "GRU", "TCN", "Patch Transformer"):
        torch.manual_seed(28)
        model = SequenceExpert(name, columns).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        for _ in range(4):
            for batch in torch.randperm(len(a)).split(256):
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(a[batch]), labels[batch])
                loss.backward()
                optimizer.step()
        torch.save({key: value.cpu() for key, value in model.state_dict().items()}, artifacts / f"{name}.pt")
        model.eval()
        with torch.no_grad():
            results[name] = torch.cat(
                [model(batch).softmax(-1) for batch in b.split(512)]
            ).cpu().numpy()
    return results
