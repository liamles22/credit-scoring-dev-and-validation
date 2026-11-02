"""Neutral WoE for fields that did not exist yet (src/binning.py)."""
import numpy as np
import pandas as pd
import pytest

from src.binning import ScorecardBinning
from src.data import TARGET


@pytest.fixture
def loans():
    rng = np.random.default_rng(0)
    n = 4000
    issue = pd.to_datetime(rng.choice(["2015-06-01", "2016-06-01"], n))
    score = rng.normal(size=n)
    late = pd.Series(score + rng.normal(size=n)).where(issue >= "2016-01-01")
    return pd.DataFrame({"issue_d": issue, "early": score, "late": late,
                         TARGET: (rng.random(n) < 1 / (1 + np.exp(2.5 - score))).astype(int)})


def test_older_loans_get_neutral_woe_and_label(loans):
    b = ScorecardBinning(["early", "late"], [], late_features=["late"]).fit(loans)
    old = loans["issue_d"] < "2016-01-01"
    assert (b.transform(loans).loc[old, "late"] == 0).all()
    assert (b.transform(loans, metric="bins").loc[old, "late"] == "Not reported").all()
    assert b.transform(loans).loc[~old, "late"].nunique() > 1


def test_higher_woe_means_higher_risk(loans):
    w = ScorecardBinning(["early"], []).fit(loans).transform(loans)["early"]
    assert np.corrcoef(w, loans[TARGET])[0, 1] > 0
