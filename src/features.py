"""Derived affordability features, defined in docs/preregistration_model_v2.md."""
import pandas as pd

AFFORDABILITY_DIRECTIONS = {
    "loan_to_income": 1,
    "dti_with_loan": 1,
    "revol_bal_to_income": 0,
    "total_balance_to_income": 0,
    "credit_limit_to_income": 0,
}
AFFORDABILITY = list(AFFORDABILITY_DIRECTIONS)


def add_affordability(df: pd.DataFrame) -> pd.DataFrame:
    """Add the features; income at or below zero gives a missing value."""
    out = df.copy()
    income = out["annual_inc"].where(out["annual_inc"] > 0)
    out["loan_to_income"] = out["loan_amnt"] / income
    out["dti_with_loan"] = out["dti"] + 100 * (out["loan_amnt"] / 36) / (income / 12)
    out["revol_bal_to_income"] = out["revol_bal"] / income
    out["total_balance_to_income"] = out["tot_cur_bal"] / income
    out["credit_limit_to_income"] = out["total_rev_hi_lim"] / income
    return out
