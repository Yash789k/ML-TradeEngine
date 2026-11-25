"""Batched FinBERT scoring for large headline sets.

``src/features/sentiment.py`` scores through a Hugging Face pipeline, which runs
one headline per forward pass (about 10 headlines a second on this machine). The
review scores about 40,000 cached headlines, so this module runs the same model
(``ProsusAI/finbert``, same tokenizer, same 128-token truncation) in real
batches. ``tests/test_review_v1_sentiment.py`` checks that both give the same
probabilities; the repository pipeline itself is left unchanged.
"""

from __future__ import annotations

import numpy as np

MODEL = "ProsusAI/finbert"


class BatchedFinBERT:
    """Returns ``(n, 3)`` probabilities ordered [positive, negative, neutral]."""

    def __init__(self, batch_size: int = 64, max_length: int = 128):
        self.batch_size, self.max_length = batch_size, max_length
        self._model = self._tokenizer = self._order = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(MODEL).to("cpu").eval()
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        self._order = [labels["positive"], labels["negative"], labels["neutral"]]
        self._torch = torch

    def score(self, headlines: list) -> np.ndarray:
        if not headlines:
            return np.zeros((0, 3))
        self._load()
        torch = self._torch
        out = np.zeros((len(headlines), 3))
        # Sort by length so each batch pads to a similar width; restore order afterwards.
        order = np.argsort([len(h) for h in headlines], kind="stable")
        with torch.no_grad():
            for lo in range(0, len(order), self.batch_size):
                ids = order[lo : lo + self.batch_size]
                batch = self._tokenizer([headlines[i] for i in ids], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                probabilities = torch.softmax(self._model(**batch).logits, dim=-1).numpy()
                out[ids] = probabilities[:, self._order]
        return out
