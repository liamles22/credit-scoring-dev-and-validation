"""Loading Lending Club data and building the modelling base.

Pipeline:
    raw CSV (2.26M loans, 151 columns)
      -> build_interim(): 36-month loans, all years, parsed dates -> data/interim/loans_36m.parquet
      -> make_modelling_base(): 2010-2017, loans with 12 months observed, 12-month default target

Target: 1 if the loan reached 90+ days past due within 12 months of issue.
The public data has no monthly payment history, so the default month is
reconstructed from the number of instalments paid (see `default_month`).
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "loans_36m.parquet"

TARGET = "target"  # 12-month default: the model target
TARGET_LIFETIME = "target_lifetime"  # default over the full term: kept for comparison only

HORIZON_MONTHS = 12  # performance window
DPD_MONTHS = 3  # default = 90 days past due (CRR Art. 178)
BUFFER_MONTHS = 3  # extra months after the window before a loan is used

# Statuses of loans that have defaulted at some point.
BAD_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
]
GOOD_STATUSES = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]
# Loans with no final outcome yet. They get a 12-month target, but no lifetime target.
OPEN_STATUSES = ["Current", "In Grace Period", "Late (16-30 days)"]

DEV_YEARS = (2010, 2017)  # inclusive; 2017 is the out-of-time year


def load_raw(path: Path = RAW_PATH, usecols=None) -> pd.DataFrame:
    """Read the raw CSV and drop the summary rows that Lending Club appends at the end."""
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    # Footer rows like "Total amount funded in policy code 1: ..." have no loan amount.
    return df[df["loan_amnt"].notna()].reset_index(drop=True) if "loan_amnt" in df else df


def _parse_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def build_interim(raw_path: Path = RAW_PATH, out_path: Path = INTERIM_PATH) -> pd.DataFrame:
    """Keep 36-month loans of all years and save them as parquet.

    All columns are kept on purpose: leakage columns are removed later, in one
    documented place (features_config.py), so a validator can trace the decision.
    """
    df = load_raw(raw_path)
    df["term"] = df["term"].str.strip()
    df = df[df["term"] == "36 months"].copy()

    df["issue_d"] = _parse_month(df["issue_d"])
    df["issue_year"] = df["issue_d"].dt.year.astype("int16")
    df["earliest_cr_line"] = _parse_month(df["earliest_cr_line"])

    # Mixed-type text columns break parquet; store them as strings.
    for col in df.columns[df.dtypes == object]:
        df[col] = df[col].astype("string")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_interim(path: Path = INTERIM_PATH, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Date the data was extracted: the latest payment month in the file."""
    return _parse_month(df["last_pymnt_d"]).max()


def months_paid(df: pd.DataFrame) -> pd.Series:
    """Number of full instalments covered by principal and interest received.

    Late fees and post charge-off recoveries are not counted, so they do not
    push the default month later.
    """
    paid = df["total_rec_prncp"].fillna(0) + df["total_rec_int"].fillna(0)
    return np.floor(paid / df["installment"] + 1e-6)


def default_month(df: pd.DataFrame, dpd_months: int = DPD_MONTHS) -> pd.Series:
    """Month after issue in which the loan reached 90+ days past due.

    The first missed instalment is the one after the last instalment paid; the
    loan reaches 90+ dpd `dpd_months` later. Only meaningful for bad statuses.
    """
    return months_paid(df) + 1 + dpd_months


def add_target_lifetime(df: pd.DataFrame) -> pd.DataFrame:
    """Lifetime target: 1 = bad, 0 = good, NaN = no final outcome."""
    status = df["loan_status"]
    df[TARGET_LIFETIME] = np.select(
        [status.isin(BAD_STATUSES), status.isin(GOOD_STATUSES)], [1.0, 0.0], default=np.nan
    )
    return df


def add_target_12m(df: pd.DataFrame, horizon: int = HORIZON_MONTHS) -> pd.DataFrame:
    """12-month target: 1 if a defaulted loan reached 90+ dpd within `horizon` months."""
    bad = df["loan_status"].isin(BAD_STATUSES)
    df[TARGET] = (bad & (default_month(df) <= horizon)).astype("int8")
    return df


def observed_long_enough(df: pd.DataFrame, snapshot: pd.Timestamp,
                         horizon: int = HORIZON_MONTHS, buffer: int = BUFFER_MONTHS) -> pd.Series:
    """True for loans whose whole performance window, plus a buffer, is before the snapshot."""
    return df["issue_d"] + pd.DateOffset(months=horizon + buffer) <= snapshot


def make_modelling_base(df: pd.DataFrame, years=DEV_YEARS, horizon: int = HORIZON_MONTHS,
                        buffer: int = BUFFER_MONTHS) -> pd.DataFrame:
    """Loans issued in `years` with the full performance window observed, plus both targets."""
    snapshot = snapshot_date(df)  # taken from all loans, before any filtering
    df = df[df["issue_year"].between(*years)].copy()
    df = df[observed_long_enough(df, snapshot, horizon, buffer)].copy()
    df = add_target_12m(df, horizon)
    df = add_target_lifetime(df)
    return df.reset_index(drop=True)


def vintage_table(df: pd.DataFrame) -> pd.DataFrame:
    """By issue year: loans, share still open, lifetime and 12-month default rates."""
    cols = ["issue_d", "issue_year", "loan_status", "last_pymnt_d",
            "total_rec_prncp", "total_rec_int", "installment"]
    d = add_target_lifetime(add_target_12m(df[cols].copy()))
    # 12-month default rate only over loans observed for the full window.
    d[TARGET] = d[TARGET].astype(float).where(observed_long_enough(d, snapshot_date(df)))
    return (
        d.groupby("issue_year")
        .agg(
            loans=("loan_status", "size"),
            open_share=("loan_status", lambda s: s.isin(OPEN_STATUSES).mean()),
            dr_lifetime=(TARGET_LIFETIME, "mean"),
            dr_12m=(TARGET, "mean"),
        )
        .round(3)
    )
