"""Discrimination, stability and calibration metrics (src/metrics.py)."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import calibration_table, gini, hosmer_lemeshow, iv, ks, psi


def test_gini_extremes():
    y = np.array([0, 0, 1, 1])
    assert gini(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert gini(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(-1.0)


def test_gini_random_score_is_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    assert abs(gini(y, rng.random(20_000))) < 0.03


def test_ks_perfect_separation():
    assert ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_psi_zero_for_identical_distributions():
    bins = pd.Series(list("aabbcc"))
    assert psi(bins, bins) == pytest.approx(0.0)


def test_psi_known_value():
    expected = pd.Series(["a"] * 50 + ["b"] * 50)
    actual = pd.Series(["a"] * 25 + ["b"] * 75)
    # (0.25 - 0.5) ln(0.25 / 0.5) + (0.75 - 0.5) ln(0.75 / 0.5)
    assert psi(expected, actual) == pytest.approx(0.25 * np.log(2) + 0.25 * np.log(1.5))


def test_psi_handles_new_category():
    assert psi(pd.Series(["a", "b"]), pd.Series(["a", "c"])) > 0.25


def test_iv_zero_without_separation_and_positive_with_it():
    bins = ["x", "x", "y", "y"]
    assert iv(bins, [0, 1, 0, 1]) == pytest.approx(0.0)
    assert iv(["x"] * 3 + ["y"] * 3, [0, 0, 1, 0, 1, 1]) > 0


def test_calibration_of_a_well_calibrated_model():
    rng = np.random.default_rng(1)
    pd_ = rng.uniform(0.01, 0.2, 50_000)
    y = rng.random(50_000) < pd_
    t = calibration_table(y, pd_)
    assert len(t) == 10
    assert (t["binomial_p"] > 0.001).all()
    assert hosmer_lemeshow(y, pd_, out_of_sample=True)[1] > 0.001


def test_calibration_detects_under_prediction():
    rng = np.random.default_rng(2)
    true_pd = rng.uniform(0.01, 0.2, 50_000)
    y = rng.random(50_000) < true_pd
    assert hosmer_lemeshow(y, true_pd * 0.7, out_of_sample=True)[1] < 0.001


def test_hosmer_lemeshow_degrees_of_freedom():
    from scipy.stats import chi2

    rng = np.random.default_rng(3)
    pd_ = rng.uniform(0.01, 0.2, 50_000)
    y = rng.random(50_000) < pd_
    stat, p_fit = hosmer_lemeshow(y, pd_, out_of_sample=False)
    _, p_oos = hosmer_lemeshow(y, pd_, out_of_sample=True)
    assert p_fit == pytest.approx(chi2.sf(stat, 8))
    assert p_oos == pytest.approx(chi2.sf(stat, 10))


def test_hosmer_lemeshow_requires_explicit_sample_type():
    with pytest.raises(TypeError):
        hosmer_lemeshow([0, 1], [0.1, 0.2])
