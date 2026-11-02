"""Applicant-level reasons from LightGBM contributions.

Features are grouped into plain-language factors and the contributions of a group are summed, so
that related features do not take up several reasons. The text names the factor, not a direction,
and is a prototype for review, not approved adverse-action wording.
"""
import numpy as np
import pandas as pd

REASON_GROUPS = {
    "Credit score": ["fico_range_low", "fico_range_high"],
    "Debt payments relative to income": ["dti"],
    "Income level": ["annual_inc"],
    "Requested loan amount": ["loan_amnt"],
    "Recent credit inquiries": ["inq_last_6mths", "inq_last_12m", "inq_fi", "mths_since_recent_inq"],
    "Recently opened accounts": [
        "acc_open_past_24mths", "num_tl_op_past_12m", "open_acc_6m", "open_il_12m", "open_il_24m", "open_rv_12m",
        "open_rv_24m", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mths_since_rcnt_il", "mths_since_recent_bc"],
    "Use of available credit": [
        "revol_util", "bc_util", "all_util", "il_util", "percent_bc_gt_75", "bc_open_to_buy", "revol_bal", "max_bal_bc"],
    "Length of credit history": ["credit_hist_months", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op"],
    "Delinquency, collection or public record history": [
        "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog", "pub_rec",
        "pub_rec_bankruptcies", "tax_liens", "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt",
        "chargeoff_within_12_mths", "delinq_amnt", "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq",
        "num_accts_ever_120_pd", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "pct_tl_nvr_dlq"],
    "Total balances and credit limits": [
        "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
        "total_il_high_credit_limit", "total_bal_il"],
    "Number and type of accounts": [
        "open_acc", "total_acc", "mort_acc", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl",
        "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "open_act_il", "total_cu_tl"],
    "Length of employment": ["emp_length"],
    "Home ownership": ["home_ownership"],
    "Stated loan purpose": ["purpose"],
    "Income verification status": ["verification_status"],
    "Loan listing or disbursement type": ["initial_list_status", "disbursement_method"],
}
FEATURE_TO_REASON = {f: reason for reason, features in REASON_GROUPS.items() for f in features}
# Set by the lender, so a poor basis for an explanation to the applicant.
LENDER_SIDE_REASONS = {"Income verification status", "Loan listing or disbursement type"}


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    """Sum feature contributions (rows = loans, columns = features) by reason."""
    return contrib.T.groupby(FEATURE_TO_REASON).sum().T


def top_reasons(contrib: pd.DataFrame, k: int = 4) -> list:
    """For each loan, up to `k` reasons that raise its risk, strongest first."""
    grouped = group_contributions(contrib)
    values, names = grouped.to_numpy(), np.asarray(grouped.columns)
    order = np.argsort(-values, axis=1)[:, :k]
    return [[names[j] for j in row if values[i, j] > 0] for i, row in enumerate(order)]
