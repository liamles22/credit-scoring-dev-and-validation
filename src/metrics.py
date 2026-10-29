"""Discrimination and stability metrics used in development and validation."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y, score) -> float:
    """Gini = 2 * AUC - 1. Positive when a higher score means a higher default risk."""
    return 2 * roc_auc_score(y, score) - 1


def ks(y, score) -> float:
    """Kolmogorov-Smirnov statistic: max distance between the score distributions of bads and goods."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def psi(expected, actual, eps: float = 1e-4) -> float:
    """Population stability index between two distributions over the same categories.

    `expected` and `actual` are pd.Series of labels (bins); shares are aligned on the union
    of labels, and empty shares are floored at `eps`.
    """
    e = pd.Series(expected).value_counts(normalize=True)
    a = pd.Series(actual).value_counts(normalize=True)
    e, a = e.align(a, fill_value=0)
    e, a = e.clip(lower=eps), a.clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())


def iv(bins, y, eps: float = 1e-4) -> float:
    """Information value of a binned variable: sum over bins of (%bad - %good) * WoE."""
    t = pd.crosstab(pd.Series(bins).to_numpy(), pd.Series(y).to_numpy())
    good = (t[0] / t[0].sum()).clip(lower=eps)
    bad = (t[1] / t[1].sum()).clip(lower=eps)
    return float(((bad - good) * np.log(bad / good)).sum())
