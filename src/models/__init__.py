"""
Phase 03 — ML Modeling
Exposes the top-level ModelTrainer and supporting utilities.
"""


__all__ = ["ModelTrainer"]


def __getattr__(name):
    if name == "ModelTrainer":
        from .trainer import ModelTrainer
        return ModelTrainer
    raise AttributeError(name)
