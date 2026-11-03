"""ZIP3 fairness proxy (src/fairness.py), on small synthetic examples."""
import numpy as np
import pandas as pd
import pytest

from src.fairness import (adverse_impact, approve_lowest, area_quintiles, area_sums,
                          bootstrap_area_weights, group_rates, lc_zip3, zip3_composition)


def acs(rows):
    cols = ["zcta", "B03002_001E", "B03002_003E", "B03002_004E", "B03002_012E", "B19013_001E"]
    return pd.DataFrame(rows, columns=cols)


def test_composition_sums_zctas_by_first_three_digits():
    s = zip3_composition(acs([["19001", 100, 80, 10, 10, 50_000],
                              ["19002", 300, 60, 180, 60, 30_000],
                              ["57701", 0, 0, 0, 0, -666666666]]))
    assert list(s.index) == ["190"]
    assert s.loc["190", "White"] == pytest.approx(140 / 400)
    assert s.loc["190", "minority_share"] == pytest.approx((190 + 70) / 400)
    assert s.loc["190", "median_income"] == pytest.approx((100 * 50_000 + 300 * 30_000) / 400)


def test_missing_income_code_is_ignored():
    s = zip3_composition(acs([["19001", 100, 100, 0, 0, -666666666], ["19002", 100, 100, 0, 0, 40_000]]))
    assert s.loc["190", "median_income"] == 40_000


def test_lc_zip3():
    assert lc_zip3(pd.Series(["190xx", "007xx"])).tolist() == ["190", "007"]


def test_approve_lowest_breaks_ties_at_random_but_reproducibly():
    score = np.array([1, 2, 2, 2, 3])
    a = approve_lowest(score, 2, seed=0)
    assert a.sum() == 2 and a[0] and not a[4]
    assert (a == approve_lowest(score, 2, seed=0)).all()


def shares_two_areas():
    return pd.DataFrame({"White": [1.0, 0.0], "Black": [0.0, 1.0], "Hispanic": [0.0, 0.0]}, index=["a", "b"])


def test_group_rates_and_air_on_pure_areas():
    area = pd.Series(["a"] * 4 + ["b"] * 4)
    approved = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    sums = area_sums(area, a=approved)
    shares = shares_two_areas().assign(Hispanic=[0.5, 0.5], White=[0.5, 0.0], Black=[0.0, 0.5])
    rates = group_rates(sums, shares, "a")
    assert rates["White"] == pytest.approx(1.0)
    assert rates["Black"] == pytest.approx(0.5)
    assert rates["Hispanic"] == pytest.approx(0.75)
    assert adverse_impact(rates).to_dict() == pytest.approx({"Black": 0.5, "Hispanic": 0.75})


def test_area_weights_reproduce_duplicating_an_area():
    area = pd.Series(["a", "a", "b", "b"])
    sums = area_sums(area, a=[1, 0, 1, 1])
    shares = pd.DataFrame({"White": [0.8, 0.2], "Black": [0.2, 0.8], "Hispanic": [0.0, 0.0]}, index=["a", "b"])
    doubled = area_sums(pd.Series(["a", "a", "b", "b", "b2", "b2"]), a=[1, 0, 1, 1, 1, 1])
    shares2 = pd.concat([shares, shares.loc[["b"]].rename(index={"b": "b2"})])
    expected = group_rates(doubled, shares2, "a")
    got = group_rates(sums, shares, "a", np.array([1.0, 2.0]))
    assert got[["White", "Black"]].to_numpy() == pytest.approx(expected[["White", "Black"]].to_numpy())


def test_bootstrap_weights_resample_all_areas():
    w = bootstrap_area_weights(7, n_boot=50, seed=0)
    assert w.shape == (50, 7) and (w.sum(axis=1) == 7).all()


def test_area_quintiles_keep_areas_whole_and_balance_loans():
    value = pd.Series(np.arange(10.0), index=list("abcdefghij"))
    q = area_quintiles(value, pd.Series(10, index=value.index))
    assert q.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
