"""
Phase 02 — Feature Engineering
Exposes the top-level FeatureEngineer class and individual feature-group adders.
"""


__all__ = ["FeatureEngineer"]


def __getattr__(name):
    if name == "FeatureEngineer":
        from .pipeline import FeatureEngineer
        return FeatureEngineer
    raise AttributeError(name)
