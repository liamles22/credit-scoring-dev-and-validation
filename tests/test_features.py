"""Affordability features (src/features.py)."""
import numpy as np
import pandas as pd
import pytest

from src.features import AFFORDABILITY, add_affordability
from src.features_config import assert_no_leakage


def loans(income):
    return pd.DataFrame({"annual_inc": [income], "loan_amnt": [12_000.0], "dti": [10.0],
                         "revol_bal": [6_000.0], "tot_cur_bal": [30_000.0],
                         "total_rev_hi_lim": [24_000.0]})


def test_values():
    r = add_affordability(loans(60_000.0)).iloc[0]
    assert r["loan_to_income"] == pytest.approx(0.2)
    # 12,000 / 36 = 333.33 per month on 5,000 monthly income -> 6.67 pp of DTI.
    assert r["dti_with_loan"] == pytest.approx(10 + 100 / 15)
    assert r["revol_bal_to_income"] == pytest.approx(0.1)
    assert r["total_balance_to_income"] == pytest.approx(0.5)
    assert r["credit_limit_to_income"] == pytest.approx(0.4)


@pytest.mark.parametrize("income", [0.0, -1.0])
def test_non_positive_income_gives_missing(income):
    r = add_affordability(loans(income)).iloc[0]
    assert all(np.isnan(r[f]) for f in AFFORDABILITY)


def test_input_is_not_modified_and_features_are_allowed():
    df = loans(60_000.0)
    add_affordability(df)
    assert "loan_to_income" not in df
    assert_no_leakage(AFFORDABILITY)
