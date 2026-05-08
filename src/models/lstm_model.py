"""
Phase 03 — PyTorch LSTM Sequence Classifier

Architecture
------------
  Input  : (batch, seq_len=60, n_features)
  LSTM   : 2-layer unidirectional (causal — no future leakage)
  Head   : Dropout → Linear(hidden→64) → ReLU → Linear(64→3)
  Output : raw logits; probabilities via softmax

Training features
-----------------
  - Class-weighted CrossEntropyLoss for imbalanced labels
  - Early stopping on validation loss (patience configurable)
  - RobustScaler applied per-feature before sequence construction
  - Gradient clipping (max_norm=1.0) to stabilise training

Predict contract
----------------
  predict_proba(X) → ndarray (n_rows, 3)
  The first (seq_len - 1) rows get a dummy uniform probability [1/3, 1/3, 1/3]
  because there is not enough history to form a complete sequence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Sliding-window dataset: each sample is a (seq_len, n_features) tensor
    paired with the label at the LAST bar of the window.

    Parameters
    ----------
    X       : (n_rows, n_features) float array — already scaled
    y       : (n_rows,) integer label array
    seq_len : number of bars in each sequence window
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 60,
    ) -> None:
        self.seq_len = seq_len
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq   = self.X[idx : idx + self.seq_len]
        label = self.y[idx + self.seq_len - 1]
        return seq, label


# ---------------------------------------------------------------------------
# LSTM model definition
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 128,
        n_layers:   int = 2,
        n_classes:  int = 3,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last timestep → logits


# ---------------------------------------------------------------------------
# LSTMModel  (sklearn-like wrapper)
# ---------------------------------------------------------------------------

class LSTMModel:
    """
    PyTorch LSTM 3-class directional classifier.

    Parameters
    ----------
    seq_len      : sequence window length in bars (default 60)
    hidden_dim   : LSTM hidden state size (default 128)
    n_layers     : number of LSTM layers (default 2)
    dropout      : dropout rate applied between LSTM layers and in head
    n_epochs     : maximum training epochs (default 50)
    batch_size   : mini-batch size (default 64)
    lr           : Adam learning rate (default 1e-3)
    patience     : early stopping patience (epochs without val improvement)
    random_state : seed for reproducibility
    """

    def __init__(
        self,
        seq_len:      int   = 60,
        hidden_dim:   int   = 128,
        n_layers:     int   = 2,
        dropout:      float = 0.3,
        n_epochs:     int   = 50,
        batch_size:   int   = 64,
        lr:           float = 1e-3,
        patience:     int   = 10,
        random_state: int   = 42,
    ) -> None:
        self.seq_len      = seq_len
        self.hidden_dim   = hidden_dim
        self.n_layers     = n_layers
        self.dropout      = dropout
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.patience     = patience
        self.random_state = random_state

        self.net_: Optional[_LSTMNet] = None
        self.scaler_ = RobustScaler()
        self.n_features_: int = 0
        self.train_history_: list[dict] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
        class_weight_map: dict[int, float] | None = None,
    ) -> "LSTMModel":
        """
        Train the LSTM with optional validation set for early stopping.

        If X_val / y_val are not provided the last 20 % of training data
        is used for early stopping.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_train = self.scaler_.fit_transform(X_train)
        self.n_features_ = X_train.shape[1]

        # Validation split
        if X_val is None:
            n_val  = max(self.seq_len + 1, int(len(X_train) * 0.2))
            X_val, X_train = X_train[-n_val:], X_train[:-n_val]
            y_val, y_train = y_val[-n_val:] if y_val is not None else y_train[-n_val:], y_train[:-n_val]
        else:
            X_val = self.scaler_.transform(X_val)

        train_ds = SequenceDataset(X_train, y_train, self.seq_len)
        val_ds   = SequenceDataset(X_val,   y_val,   self.seq_len)

        if len(train_ds) < 1:
            log.warning("LSTM: too few training samples for seq_len=%d; skipping.", self.seq_len)
            self.net_ = _LSTMNet(self.n_features_, self.hidden_dim, self.n_layers, dropout=self.dropout)
            return self

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        # Class weights → loss weight tensor
        cw       = class_weight_map or {0: 1.0, 1: 1.0, 2: 1.0}
        w_tensor = torch.tensor(
            [cw.get(i, 1.0) for i in range(3)], dtype=torch.float32
        ).to(_DEVICE)

        self.net_ = _LSTMNet(
            self.n_features_, self.hidden_dim, self.n_layers, dropout=self.dropout
        ).to(_DEVICE)

        criterion = nn.CrossEntropyLoss(weight=w_tensor)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        best_val_loss   = float("inf")
        patience_count  = 0
        best_state_dict = None

        for epoch in range(self.n_epochs):
            # ── train ──────────────────────────────────────────────────
            self.net_.train()
            train_loss = 0.0
            for X_b, y_b in train_dl:
                X_b, y_b = X_b.to(_DEVICE), y_b.to(_DEVICE)
                optimizer.zero_grad()
                logits = self.net_(X_b)
                loss   = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_b)
            train_loss /= max(1, len(train_ds))

            # ── validate ───────────────────────────────────────────────
            self.net_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_dl:
                    X_b, y_b = X_b.to(_DEVICE), y_b.to(_DEVICE)
                    logits    = self.net_(X_b)
                    val_loss += criterion(logits, y_b).item() * len(X_b)
            val_loss = val_loss / max(1, len(val_ds))

            scheduler.step(val_loss)
            self.train_history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if epoch % 10 == 0:
                log.info("LSTM epoch %3d  train=%.4f  val=%.4f", epoch, train_loss, val_loss)

            # ── early stopping ─────────────────────────────────────────
            if val_loss < best_val_loss - 1e-5:
                best_val_loss   = val_loss
                patience_count  = 0
                best_state_dict = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    log.info("LSTM early stop at epoch %d (best val=%.4f)", epoch, best_val_loss)
                    break

        if best_state_dict is not None:
            self.net_.load_state_dict(best_state_dict)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each row.
        Rows without a full seq_len history (the first seq_len-1 rows) receive
        uniform probability [1/3, 1/3, 1/3].
        """
        if self.net_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X_scaled = self.scaler_.transform(X)
        n        = len(X_scaled)
        result   = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)

        if n < self.seq_len:
            return result

        # Build sequences for all valid positions
        seqs  = np.stack([X_scaled[i : i + self.seq_len]
                          for i in range(n - self.seq_len + 1)])
        t_in  = torch.tensor(seqs, dtype=torch.float32).to(_DEVICE)

        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(t_in)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        result[self.seq_len - 1 :] = probs
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict":  self.net_.state_dict(),
                "config": {
                    "n_features": self.n_features_,
                    "hidden_dim": self.hidden_dim,
                    "n_layers":   self.n_layers,
                    "dropout":    self.dropout,
                    "seq_len":    self.seq_len,
                },
            },
            str(p),
        )
        joblib.dump(self.scaler_, str(p).replace(".pt", "_scaler.pkl"))

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        import joblib
        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt["config"]
        obj  = cls(
            seq_len    = cfg["seq_len"],
            hidden_dim = cfg["hidden_dim"],
            n_layers   = cfg["n_layers"],
            dropout    = cfg["dropout"],
        )
        obj.n_features_ = cfg["n_features"]
        obj.net_ = _LSTMNet(cfg["n_features"], cfg["hidden_dim"], cfg["n_layers"], dropout=cfg["dropout"])
        obj.net_.load_state_dict(ckpt["state_dict"])
        obj.net_.to(_DEVICE)
        obj.net_.eval()
        obj.scaler_ = joblib.load(path.replace(".pt", "_scaler.pkl"))
        return obj
