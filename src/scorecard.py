"""Points scaling of the logistic WoE model.

Convention: a score of BASE_SCORE means odds of BASE_ODDS goods per bad, and every PDO
points double the odds. Higher score = lower risk, as in bank scorecards.
"""
import numpy as np
import pandas as pd

PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def score_from_logit(logit):
    """Score from the model's log-odds of default: ln(good/bad odds) = -logit."""
    return OFFSET - FACTOR * np.asarray(logit)


def points_table(binning, coefs: pd.Series, intercept: float) -> pd.DataFrame:
    """Points of every bin of every feature; a loan's score is the sum of its bins' points.

    The intercept and offset are spread evenly over the features, so each feature's
    points are directly comparable.
    """
    n = len(coefs)
    rows = []
    for name, b in coefs.items():
        bt = binning.table(name)
        bt = bt[(bt["Count"].fillna(1) > 0) | bt["Bin"].isin(["Missing", "Not reported"])]
        for _, r in bt.iterrows():
            rows.append({
                "feature": name, "bin": r["Bin"], "share": r.get("Count (%)"),
                "default_rate": r.get("Event rate"), "woe": r["WoE"],
                "points": (OFFSET - FACTOR * intercept) / n - FACTOR * b * r["WoE"],
            })
    return pd.DataFrame(rows)
