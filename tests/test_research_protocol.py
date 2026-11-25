import numpy as np
import pandas as pd
from src.research.protocol import evidence, Protocol
from src.research.ranker import composite_score


def test_negative_evidence_cannot_outrank_positive_evidence():
    row = dict(sharpe_ratio=0, calmar_ratio=0, cagr=0, alpha=0)
    assert composite_score(pd.Series(dict(row, t_stat=-5))) < composite_score(
        pd.Series(dict(row, t_stat=5))
    )


def test_dependent_uncertainty_reproducible_and_multiplicity_explicit():
    rng = np.random.default_rng(10)
    returns = pd.Series(rng.normal(-0.001, 0.01, 350))
    benchmark = pd.Series(np.zeros(350))
    protocol = Protocol(bootstrap_samples=100)
    a = evidence(returns, benchmark, 330, protocol)
    b = evidence(returns, benchmark, 330, protocol)
    assert a == b and a["family_size"] == 330
    assert a["bonferroni_p"] >= a["one_sided_p"] and not a["positive_evidence"]
