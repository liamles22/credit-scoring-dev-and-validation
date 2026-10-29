"""12-month target reconstruction (src/data.py)."""
import numpy as np
import pandas as pd

from src.data import (TARGET, TARGET_LIFETIME, add_target_12m, add_target_lifetime,
                      default_month, months_paid, observed_long_enough, snapshot_date)


def loans(statuses, paid_instalments, installment=100.0):
    """Loans whose principal + interest received equals `paid_instalments` instalments."""
    paid = np.asarray(paid_instalments, dtype=float) * installment
    return pd.DataFrame({
        "loan_status": statuses,
        "installment": installment,
        "total_rec_prncp": paid * 0.8,
        "total_rec_int": paid * 0.2,
    })


def test_months_paid_counts_full_instalments_only():
    df = loans(["Charged Off"] * 3, [8, 8.99, 0])
    assert months_paid(df).tolist() == [8, 8, 0]


def test_default_month_is_first_missed_instalment_plus_90_days():
    # 8 instalments paid -> first missed is the 9th -> 90+ dpd in month 12.
    assert default_month(loans(["Charged Off"], [8])).iloc[0] == 12


def test_12m_target_boundary():
    df = add_target_12m(loans(["Charged Off", "Charged Off", "Charged Off"], [7, 8, 9]))
    # default in month 11 and 12 -> bad; month 13 -> outside the window.
    assert df[TARGET].tolist() == [1, 1, 0]


def test_12m_target_never_bad_for_good_or_open_statuses():
    statuses = ["Fully Paid", "Current", "In Grace Period", "Late (16-30 days)"]
    df = add_target_12m(loans(statuses, [0, 0, 0, 0]))
    assert df[TARGET].sum() == 0


def test_12m_target_ignores_late_fees_and_recoveries():
    df = loans(["Charged Off"], [8])
    df["total_rec_late_fee"] = 500.0
    df["recoveries"] = 1000.0
    assert add_target_12m(df)[TARGET].iloc[0] == 1


def test_lifetime_target_is_missing_for_open_loans():
    df = add_target_lifetime(loans(["Charged Off", "Fully Paid", "Current"], [0, 36, 10]))
    assert df[TARGET_LIFETIME].iloc[0] == 1
    assert df[TARGET_LIFETIME].iloc[1] == 0
    assert np.isnan(df[TARGET_LIFETIME].iloc[2])


def test_snapshot_is_latest_payment_month():
    df = pd.DataFrame({"last_pymnt_d": ["Jan-2019", "Mar-2019", None]})
    assert snapshot_date(df) == pd.Timestamp("2019-03-01")


def test_observed_long_enough_needs_window_plus_buffer():
    df = pd.DataFrame({"issue_d": pd.to_datetime(["2017-12-01", "2018-01-01"])})
    ok = observed_long_enough(df, pd.Timestamp("2019-03-01"), horizon=12, buffer=3)
    assert ok.tolist() == [True, False]
