"""Approval strategy economics (src/business.py)."""
import numpy as np
import pytest

from src.business import AVG_BALANCE, EAD_SHARE, LGD, approved_mask, loan_profit, paired_difference, strategy_row


def test_approves_lowest_risk_loans():
    assert approved_mask([0.9, 0.1, 0.5, 0.3], 0.5).tolist() == [False, True, False, True]


def test_ties_keep_exact_share():
    assert approved_mask(np.ones(1000), 0.8).sum() == 800


def test_loan_profit_values():
    p = loan_profit([0, 1], [1000, 1000], [10, 10])
    assert p[0] == pytest.approx(0.10 * 1000 * AVG_BALANCE)
    assert p[1] == pytest.approx(-1000 * EAD_SHARE * LGD)


def test_strategy_row_counts():
    y = np.array([0, 0, 0, 1])
    r = strategy_row(y, np.full(4, 100.0), np.full(4, 10.0), [0.1, 0.2, 0.3, 0.9], 0.75)
    assert r["default_rate"] == 0
    assert r["defaults_avoided_share"] == 1
    assert r["good_loans_rejected_share"] == 0


def test_paired_difference_sign():
    y = np.array([0, 1] * 100)
    profit = np.where(y == 1, -10.0, 1.0)
    d = paired_difference(y, profit, np.ones(200, bool), y == 0, n_boot=50)
    assert d["dr_diff"] == pytest.approx(-0.5)
    assert d["profit_diff"] > 0
