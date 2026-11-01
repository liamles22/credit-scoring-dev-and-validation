"""Business impact of an approval strategy over the first 12 months.

Simplified, illustrative economics, per loan:
    good: interest = int_rate * loan_amnt * AVG_BALANCE
    bad:  loss = loan_amnt * EAD_SHARE * LGD, with no interest counted

Funding and operating costs and defaults after month 12 are ignored.
"""
import numpy as np
import pandas as pd

AVG_BALANCE = 0.85  # average balance in the first year of a 36-month amortising loan
EAD_SHARE = 0.80  # share of the loan still owed at default
LGD = 0.85


def approved_mask(risk, approval_rate: float, seed: int = 42) -> np.ndarray:
    """Approve the lowest-risk share of loans; ties (such as one sub-grade) are broken at random."""
    risk = np.asarray(risk, dtype=float)
    k = int(round(len(risk) * approval_rate))
    order = np.lexsort((np.random.default_rng(seed).random(len(risk)), risk))
    mask = np.zeros(len(risk), dtype=bool)
    mask[order[:k]] = True
    return mask


def loan_profit(y, loan_amnt, int_rate, lgd: float = LGD) -> np.ndarray:
    """12-month profit of each loan; `int_rate` is in percent, as in the raw data."""
    y, amount, rate = np.asarray(y), np.asarray(loan_amnt, dtype=float), np.asarray(int_rate, dtype=float)
    return np.where(y == 1, -amount * EAD_SHARE * lgd, rate / 100 * amount * AVG_BALANCE)


def strategy_row(y, loan_amnt, int_rate, risk, approval_rate: float, lgd: float = LGD) -> dict:
    """Result of one strategy, with profit as a share of all money lent in the sample."""
    m = approved_mask(risk, approval_rate)
    profit = loan_profit(y, loan_amnt, int_rate, lgd)
    y = np.asarray(y)
    return {"approval_rate": m.mean(), "default_rate": y[m].mean(),
            "defaults_avoided_share": 1 - y[m].sum() / y.sum(),
            "good_loans_rejected_share": ((y == 0) & ~m).sum() / (y == 0).sum(),
            "profit": profit[m].sum(), "profit_pct_of_lending": profit[m].sum() / np.sum(loan_amnt) * 100}


def paired_difference(y, profit, mask_a, mask_b, n_boot: int = 500, seed: int = 42) -> pd.Series:
    """Bootstrap of B minus A: default rate of approved loans and profit."""
    y, profit = np.asarray(y), np.asarray(profit)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        a, b = mask_a[i], mask_b[i]
        rows.append((y[i][b].mean() - y[i][a].mean(), profit[i][b].sum() - profit[i][a].sum()))
    d = np.array(rows)
    return pd.Series({"dr_diff": y[mask_b].mean() - y[mask_a].mean(),
                      "dr_low": np.quantile(d[:, 0], 0.025), "dr_high": np.quantile(d[:, 0], 0.975),
                      "profit_diff": profit[mask_b].sum() - profit[mask_a].sum(),
                      "profit_low": np.quantile(d[:, 1], 0.025), "profit_high": np.quantile(d[:, 1], 0.975)})
