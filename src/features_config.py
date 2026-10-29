"""Classification of every raw column: may it enter the model, and why.

This file is the single source of truth for leakage control. Each of the 151
raw columns belongs to exactly one group; `check_coverage` fails loudly if a
column is missing or listed twice.

Groups
------
CANDIDATE     known at origination, describes the borrower or the loan request
LENDER_SCORE  known at origination, but is the output of Lending Club's own risk
              model (grade, rate). Excluded from our model, kept as a benchmark.
PROXY         known at origination, but location data: a proxy for protected
              characteristics. Excluded from the model, used for fairness tests.
LEAKAGE       only known after origination (payments, hardship, settlement)
TECHNICAL     identifiers, free text, constants
KEY           used to define the sample or the target, not a feature

TARGET_ONLY_COLS is a separate list on top of the groups: post-origination columns
that build the 12-month target. `assert_no_leakage` is called on every feature
matrix before training.
"""

CANDIDATE = {
    # Loan request
    "loan_amnt": "requested amount",
    "purpose": "stated purpose of the loan",
    "application_type": "individual or joint application",
    "initial_list_status": "whole or fractional listing, set at listing",
    "disbursement_method": "cash or direct pay, set at origination",
    # Borrower profile
    "emp_length": "years in current job",
    "home_ownership": "rent / own / mortgage",
    "annual_inc": "self-reported annual income",
    "verification_status": "whether income was verified",
    "dti": "debt-to-income ratio, excluding mortgage and this loan",
    # Joint applications
    "annual_inc_joint": "combined income, joint applications only",
    "dti_joint": "combined DTI, joint applications only",
    "verification_status_joint": "joint income verification",
    "revol_bal_joint": "combined revolving balance",
    "sec_app_fico_range_low": "co-borrower FICO at application",
    "sec_app_fico_range_high": "co-borrower FICO at application",
    "sec_app_earliest_cr_line": "co-borrower oldest credit line",
    "sec_app_inq_last_6mths": "co-borrower inquiries",
    "sec_app_mort_acc": "co-borrower mortgage accounts",
    "sec_app_open_acc": "co-borrower open accounts",
    "sec_app_revol_util": "co-borrower revolving utilisation",
    "sec_app_open_act_il": "co-borrower active instalment loans",
    "sec_app_num_rev_accts": "co-borrower revolving accounts",
    "sec_app_chargeoff_within_12_mths": "co-borrower charge-offs",
    "sec_app_collections_12_mths_ex_med": "co-borrower collections",
    "sec_app_mths_since_last_major_derog": "co-borrower major derogatory",
    # Credit bureau at application
    "fico_range_low": "FICO at application, lower bound",
    "fico_range_high": "FICO at application, upper bound",
    "earliest_cr_line": "date of oldest credit line -> credit history length",
    "delinq_2yrs": "30+ dpd incidences in past 2 years",
    "inq_last_6mths": "credit inquiries in past 6 months",
    "mths_since_last_delinq": "months since last delinquency",
    "mths_since_last_record": "months since last public record",
    "mths_since_last_major_derog": "months since last 90+ dpd rating",
    "open_acc": "open credit lines",
    "pub_rec": "derogatory public records",
    "pub_rec_bankruptcies": "bankruptcies",
    "tax_liens": "tax liens",
    "revol_bal": "revolving balance",
    "revol_util": "revolving utilisation",
    "total_acc": "total credit lines ever",
    "collections_12_mths_ex_med": "collections in past 12 months",
    "acc_now_delinq": "accounts currently delinquent",
    "tot_coll_amt": "total collection amounts ever owed",
    "tot_cur_bal": "total current balance",
    "total_rev_hi_lim": "total revolving credit limit",
    "open_acc_6m": "accounts opened in past 6 months",
    "open_act_il": "active instalment loans",
    "open_il_12m": "instalment accounts opened in past 12 months",
    "open_il_24m": "instalment accounts opened in past 24 months",
    "mths_since_rcnt_il": "months since most recent instalment account",
    "total_bal_il": "total instalment balance",
    "il_util": "instalment utilisation",
    "open_rv_12m": "revolving accounts opened in past 12 months",
    "open_rv_24m": "revolving accounts opened in past 24 months",
    "max_bal_bc": "max balance on bankcard",
    "all_util": "utilisation across all trades",
    "inq_fi": "personal finance inquiries",
    "total_cu_tl": "finance trades",
    "inq_last_12m": "inquiries in past 12 months",
    "acc_open_past_24mths": "trades opened in past 24 months",
    "avg_cur_bal": "average current balance",
    "bc_open_to_buy": "open-to-buy on bankcards",
    "bc_util": "bankcard utilisation",
    "chargeoff_within_12_mths": "charge-offs in past 12 months",
    "delinq_amnt": "past-due amount on delinquent accounts",
    "mo_sin_old_il_acct": "months since oldest instalment account",
    "mo_sin_old_rev_tl_op": "months since oldest revolving account",
    "mo_sin_rcnt_rev_tl_op": "months since most recent revolving account",
    "mo_sin_rcnt_tl": "months since most recent account",
    "mort_acc": "mortgage accounts",
    "mths_since_recent_bc": "months since most recent bankcard",
    "mths_since_recent_bc_dlq": "months since most recent bankcard delinquency",
    "mths_since_recent_inq": "months since most recent inquiry",
    "mths_since_recent_revol_delinq": "months since most recent revolving delinquency",
    "num_accts_ever_120_pd": "accounts ever 120+ dpd",
    "num_actv_bc_tl": "active bankcard accounts",
    "num_actv_rev_tl": "active revolving trades",
    "num_bc_sats": "satisfactory bankcard accounts",
    "num_bc_tl": "bankcard accounts",
    "num_il_tl": "instalment accounts",
    "num_op_rev_tl": "open revolving accounts",
    "num_rev_accts": "revolving accounts",
    "num_rev_tl_bal_gt_0": "revolving trades with balance",
    "num_sats": "satisfactory accounts",
    "num_tl_120dpd_2m": "accounts 120+ dpd in past 2 months",
    "num_tl_30dpd": "accounts 30 dpd in past 2 months",
    "num_tl_90g_dpd_24m": "accounts 90+ dpd in past 24 months",
    "num_tl_op_past_12m": "accounts opened in past 12 months",
    "pct_tl_nvr_dlq": "percent of trades never delinquent",
    "percent_bc_gt_75": "percent of bankcards above 75% utilisation",
    "tot_hi_cred_lim": "total high credit limit",
    "total_bal_ex_mort": "total balance excluding mortgage",
    "total_bc_limit": "total bankcard limit",
    "total_il_high_credit_limit": "total instalment high credit limit",
}

LENDER_SCORE = {
    "grade": "Lending Club risk grade: output of the lender's own model",
    "sub_grade": "Lending Club risk sub-grade: output of the lender's own model",
    "int_rate": "priced from sub-grade, so it encodes the lender's model",
    "installment": "derived from amount, term and int_rate",
    "funded_amnt": "set by the lender after review; near-duplicate of loan_amnt",
    "funded_amnt_inv": "amount funded by investors; investor-side, not borrower risk",
}

PROXY = {
    "zip_code": "3-digit ZIP: geographic proxy for race and ethnicity",
    "addr_state": "state: geographic proxy, also legal-regime differences",
}

LEAKAGE = {
    "out_prncp": "outstanding principal at data snapshot",
    "out_prncp_inv": "outstanding principal at data snapshot",
    "total_pymnt": "payments received to date",
    "total_pymnt_inv": "payments received to date",
    "total_rec_prncp": "principal received to date",
    "total_rec_int": "interest received to date",
    "total_rec_late_fee": "late fees received: implies delinquency",
    "recoveries": "post charge-off recoveries: implies default",
    "collection_recovery_fee": "post charge-off fee: implies default",
    "last_pymnt_d": "date of last payment",
    "last_pymnt_amnt": "amount of last payment",
    "next_pymnt_d": "next scheduled payment date",
    "last_credit_pull_d": "most recent bureau pull, after origination",
    "last_fico_range_high": "FICO at last pull, after origination",
    "last_fico_range_low": "FICO at last pull, after origination",
    "pymnt_plan": "payment plan put in place after origination",
    "hardship_flag": "hardship programme, after origination",
    "hardship_type": "hardship programme, after origination",
    "hardship_reason": "hardship programme, after origination",
    "hardship_status": "hardship programme, after origination",
    "deferral_term": "hardship programme, after origination",
    "hardship_amount": "hardship programme, after origination",
    "hardship_start_date": "hardship programme, after origination",
    "hardship_end_date": "hardship programme, after origination",
    "payment_plan_start_date": "hardship programme, after origination",
    "hardship_length": "hardship programme, after origination",
    "hardship_dpd": "hardship programme, after origination",
    "hardship_loan_status": "hardship programme, after origination",
    "orig_projected_additional_accrued_interest": "hardship programme, after origination",
    "hardship_payoff_balance_amount": "hardship programme, after origination",
    "hardship_last_payment_amount": "hardship programme, after origination",
    "debt_settlement_flag": "debt settlement: implies default",
    "debt_settlement_flag_date": "debt settlement: implies default",
    "settlement_status": "debt settlement: implies default",
    "settlement_date": "debt settlement: implies default",
    "settlement_amount": "debt settlement: implies default",
    "settlement_percentage": "debt settlement: implies default",
    "settlement_term": "debt settlement: implies default",
}

TECHNICAL = {
    "id": "loan identifier",
    "member_id": "empty in this extract",
    "url": "listing URL",
    "desc": "free-text description, mostly empty after 2014",
    "emp_title": "free-text job title, very high cardinality",
    "title": "free-text loan title, duplicates purpose",
    "policy_code": "constant",
}

KEY = {
    "issue_d": "origination month: defines train / test / OOT",
    "term": "sample filter: 36-month loans only",
    "loan_status": "source of the target",
}

# Columns used only to build the 12-month target (src/data.py). They stay in their
# group above, and must never enter the model.
TARGET_ONLY_COLS = ["last_pymnt_d", "total_rec_prncp", "total_rec_int", "installment", "recoveries"]

# Candidate fields that only exist for joint applications. The scope is individual
# applications, so they are always empty and are not used.
JOINT_ONLY = [
    "annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
    "sec_app_open_act_il", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog",
]

# Bureau fields that Lending Club started reporting in Q1 2016 (notebook 01, section 5).
NEW_IN_2016 = [
    "all_util", "il_util", "inq_fi", "inq_last_12m", "max_bal_bc", "mths_since_rcnt_il",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "open_rv_12m", "open_rv_24m",
    "total_bal_il", "total_cu_tl",
]

GROUPS = {
    "CANDIDATE": CANDIDATE,
    "LENDER_SCORE": LENDER_SCORE,
    "PROXY": PROXY,
    "LEAKAGE": LEAKAGE,
    "TECHNICAL": TECHNICAL,
    "KEY": KEY,
}


def column_table():
    """All raw columns with their group and reason, as a DataFrame."""
    import pandas as pd

    rows = [(c, g, r) for g, cols in GROUPS.items() for c, r in cols.items()]
    return pd.DataFrame(rows, columns=["column", "group", "reason"])


def check_coverage(raw_columns) -> None:
    """Every raw column is classified exactly once, and nothing extra is listed."""
    listed = [c for cols in GROUPS.values() for c in cols]
    dupes = {c for c in listed if listed.count(c) > 1}
    missing = set(raw_columns) - set(listed)
    extra = set(listed) - set(raw_columns)
    assert not dupes, f"columns in more than one group: {sorted(dupes)}"
    assert not missing, f"unclassified columns: {sorted(missing)}"
    assert not extra, f"listed but not in data: {sorted(extra)}"
    assert set(TARGET_ONLY_COLS) <= set(listed), "TARGET_ONLY_COLS must be classified columns"


def assert_no_leakage(feature_columns) -> None:
    """Fail if a post-origination, target-building, key or target column is a model feature."""
    forbidden = set(LEAKAGE) | set(TARGET_ONLY_COLS) | set(KEY) | {"target", "target_lifetime"}
    bad = sorted(set(feature_columns) & forbidden)
    assert not bad, f"forbidden columns in the feature matrix: {bad}"
