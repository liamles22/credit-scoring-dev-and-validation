"""Sample scope and time-based split.

The out-of-time year is fixed in advance (2017). The training window is chosen in
notebook 02 on a separate validation slice (H2 2016), never on the out-of-time year,
so that the out-of-time Gini stays an honest estimate.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT, TARGET

OOT = ("2017-01", "2017-12")
WINDOW_VALIDATION = ("2016-07", "2016-12")  # used only to choose the training window
CANDIDATE_WINDOWS = {
    "H1 2016": ("2016-01", "2016-06"),
    "2015 - H1 2016": ("2015-01", "2016-06"),
    "2013 - H1 2016": ("2013-01", "2016-06"),
}
TRAIN_END = "2016-12"  # the chosen window is shifted to end here for the final split
TEST_SHARE = 0.2
SEED = 42

# Result of the experiment in notebook 02: highest validation Gini, consistent in every month.
CHOSEN_WINDOW = "2013 - H1 2016"
CHOSEN_FEATURE_SET = "full"  # base features plus the bureau fields that appear in 2016

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "samples.parquet"


def select_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Individual applications only: joint applications appear late 2015 with their own fields."""
    return df[df["application_type"] == "Individual"].reset_index(drop=True)


def issued_between(df: pd.DataFrame, window) -> pd.Series:
    """True for loans issued in the inclusive month range `window` = ("YYYY-MM", "YYYY-MM")."""
    month = df["issue_d"].dt.to_period("M")
    return month.between(pd.Period(window[0], "M"), pd.Period(window[1], "M"))


def shift_to_end(window, end: str = TRAIN_END):
    """Same number of months as `window`, ending at `end`."""
    start, stop = pd.Period(window[0], "M"), pd.Period(window[1], "M")
    new_end = pd.Period(end, "M")
    return (str(new_end - (stop - start).n), str(new_end))


def final_split(df: pd.DataFrame, train_window, test_share: float = TEST_SHARE,
                seed: int = SEED) -> pd.DataFrame:
    """Label loans as train / test (random, stratified, inside the window) or oot.

    Loans outside the training window and the out-of-time year are dropped.
    """
    in_window = df[issued_between(df, train_window)]
    train_idx, test_idx = train_test_split(
        in_window.index, test_size=test_share, stratify=in_window[TARGET], random_state=seed
    )
    sample = pd.Series(pd.NA, index=df.index, dtype="string")
    sample[train_idx] = "train"
    sample[test_idx] = "test"
    sample[issued_between(df, OOT)] = "oot"
    out = df.assign(sample=sample)
    return out[out["sample"].notna()].reset_index(drop=True)
