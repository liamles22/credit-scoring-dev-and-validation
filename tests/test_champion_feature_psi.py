"""Binning for the champion PSI (scripts/champion_feature_psi.py)."""
import numpy as np
import pandas as pd

from scripts.champion_feature_psi import bins
from src.metrics import psi


def test_missing_values_keep_their_own_bin():
    x = pd.Series([1.0, 2.0, 3.0, np.nan], name="dti")
    a, _ = bins(x, x)
    assert a.notna().all()
    assert (a == "missing").sum() == 1


def test_shift_in_missing_share_is_visible():
    reference = pd.Series(np.arange(1000, dtype=float), name="dti")
    new = reference.copy()
    new.iloc[:500] = np.nan
    a, b = bins(reference, new)
    assert psi(a, b) > 0.25
