"""Time-based split (src/split.py)."""
import numpy as np
import pandas as pd

from src.data import TARGET
from src.split import OOT, final_split, issued_between, select_scope, shift_to_end


def monthly_loans(start="2012-01", end="2018-06", per_month=50, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, end, freq="MS")
    df = pd.DataFrame({"issue_d": np.repeat(months, per_month)})
    df[TARGET] = (rng.random(len(df)) < 0.05).astype("int8")
    return df


def test_shift_to_end_keeps_window_length():
    assert shift_to_end(("2013-01", "2016-06"), end="2016-12") == ("2013-07", "2016-12")


def test_issued_between_is_inclusive():
    df = monthly_loans(per_month=1)
    m = issued_between(df, ("2014-03", "2014-05"))
    assert m.sum() == 3


def test_final_split_has_no_overlap_with_oot():
    out = final_split(monthly_loans(), ("2013-07", "2016-12"))
    dev = out[out["sample"].isin(["train", "test"])]
    oot = out[out["sample"] == "oot"]
    assert dev["issue_d"].max() < oot["issue_d"].min()
    assert issued_between(oot, OOT).all()
    assert set(out["sample"]) == {"train", "test", "oot"}


def test_final_split_drops_loans_outside_window_and_oot():
    out = final_split(monthly_loans(), ("2013-07", "2016-12"))
    assert out["issue_d"].min() == pd.Timestamp("2013-07-01")
    assert out["issue_d"].max() == pd.Timestamp("2017-12-01")


def test_test_share_and_stratification():
    out = final_split(monthly_loans(per_month=200), ("2013-07", "2016-12"))
    dev = out[out["sample"] != "oot"]
    share = (dev["sample"] == "test").mean()
    assert abs(share - 0.2) < 0.01
    dr = dev.groupby("sample")[TARGET].mean()
    assert abs(dr["train"] - dr["test"]) < 0.002


def test_select_scope_keeps_individual_applications():
    df = pd.DataFrame({"application_type": ["Individual", "Joint App", "Individual"]})
    assert len(select_scope(df)) == 2
