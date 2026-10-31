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


def calibration_table(y, pd_, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by bucket of predicted PD, with a binomial test.

    `binomial_p` is the two-sided p-value of observing the bucket's defaults if the true
    default rate were the mean predicted PD of the bucket.
    """
    from scipy.stats import binomtest

    d = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_)})
    d["bucket"] = pd.qcut(d["pd"].rank(method="first"), n_bins, labels=False) + 1
    t = d.groupby("bucket").agg(loans=("y", "size"), defaults=("y", "sum"),
                                mean_pd=("pd", "mean"), default_rate=("y", "mean"))
    t["binomial_p"] = [binomtest(int(r.defaults), int(r.loans), r.mean_pd).pvalue for r in t.itertuples()]
    return t


def hosmer_lemeshow(y, pd_, *, out_of_sample: bool, n_bins: int = 10):
    """Hosmer-Lemeshow statistic and p-value over buckets of predicted PD.

    `out_of_sample` must be stated: degrees of freedom are n_bins - 2 on the fitting sample
    and n_bins on a sample the model never saw, where no parameters were estimated.
    """
    from scipy.stats import chi2

    t = calibration_table(y, pd_, n_bins)
    expected = t["loans"] * t["mean_pd"]
    stat = (((t["defaults"] - expected) ** 2) / (expected * (1 - t["mean_pd"]))).sum()
    return float(stat), float(chi2.sf(stat, n_bins if out_of_sample else n_bins - 2))


def iv(bins, y, eps: float = 1e-4) -> float:
    """Information value of a binned variable: sum over bins of (%bad - %good) * WoE."""
    t = pd.crosstab(pd.Series(bins).to_numpy(), pd.Series(y).to_numpy())
    good = (t[0] / t[0].sum()).clip(lower=eps)
    bad = (t[1] / t[1].sum()).clip(lower=eps)
    return float(((bad - good) * np.log(bad / good)).sum())
