# Pre-registration: model v2 experiment

Written before any v2 calculation. If this plan changes, the change and its reason are recorded
in the Changes section below, and affected results are labelled exploratory.

## Question

Do affordability features and time-based tuning improve LightGBM monotone and the WoE scorecard,
without using OOT 2017 for any decision?

## Data

Only `train` rows from `data/processed/samples.parquet` are used for fitting, tuning and selection:

| Slice | Issued | Role |
|---|---|---|
| Fit | 2013-07 to 2016-03 | Model fitting |
| Tuning | 2016-04 to 2016-06 | Early stopping and Optuna objective |
| Selection | 2016-07 to 2016-12 | Decision rule below |

The `test` sample is reported for information only. OOT 2017 is not loaded.

## Features

Income below or equal to zero gives a missing value.

| Feature | Definition | Monotone direction |
|---|---|---|
| `loan_to_income` | `loan_amnt / annual_inc` | +1 |
| `dti_with_loan` | `dti + 100 * (loan_amnt / 36) / (annual_inc / 12)` | +1 |
| `revol_bal_to_income` | `revol_bal / annual_inc` | none |
| `total_balance_to_income` | `tot_cur_bal / annual_inc` | none |
| `credit_limit_to_income` | `total_rev_hi_lim / annual_inc` | none |

`dti_with_loan` ignores interest, because `installment` and `int_rate` are excluded. A separate
payment-to-income feature is not added: without interest it is just `loan_to_income / 3`.

## Variants

LightGBM monotone, with existing monotone directions:

| ID | Features | Parameters |
|---|---|---|
| A0 | Current 78 | Current |
| A1 | Current 78 + 5 new | Current |
| A2 | Current 78 | Retuned |
| A3 | Current 78 + 5 new | Retuned |

Retuning: 25 Optuna trials with the existing search space, separately for A2 and A3, on seed 42.
Every variant is then fitted with seeds 42, 1, 2, 3 and 4.

WoE scorecard, with bins and coefficients refitted on the Fit slice:

| ID | Features |
|---|---|
| S0 | Current 21 |
| S1 | Current 21 + new features with IV of at least 0.02, then the existing correlation, VIF, sign and p-value rules |

## Decision rule

The primary metric is Gini on the Selection slice. A variant replaces its baseline (A0 or S0) only
if all conditions hold:

1. Mean Gini gain over the seeds is at least 0.005 (the scorecard has one fit, so this is its gain).
2. The gain is positive for every seed.
3. On seed 42, the paired bootstrap 95% interval (500 resamples) of the gain is above zero.
4. `|observed / predicted - 1|` on the Selection slice is not worse than baseline by more than 0.02.

If several LightGBM variants pass, the one with the highest mean Gini wins. A gap below 0.002
between passing variants goes to the simpler one, in the order A1, A2, A3. If none passes, the
current model stays and this is reported as the result.

## After the decision

An adopted variant is refitted on the full `train` window, with early stopping on the Tuning
slice, and then requires a separate go-ahead for a one-time confirmation on 2018 loans. That
cohort's definition and the same checks must be recorded here before its outcomes are read. It
would be small and have limited observation time, so it is supporting evidence only.

## Disclosed prior knowledge

These facts weaken the experiment and must be reported with its results:

- Current parameters were tuned with a holdout drawn from the same `train` window, and H2 2016
  rows were part of that fit.
- H2 2016 was already used to choose the training window in notebook 02.
- On 29 October 2026, loan-to-income, revolving-balance-to-income and `loan_amnt` were tried in the
  scorecard on the `test` sample, adding 0.012–0.019 Gini. The idea for this experiment came from
  that exploratory result.
- OOT 2017 results are already known, and `loan_amnt` importance comes from SHAP on OOT loans.
- The observed default rate of January 2018 loans (about 3.6%) was printed during target checks on
  29 October 2026. Earlier project notes saying 2018 outcomes were unread are not fully accurate.
  No model has been scored on 2018 loans.

## Changes

None yet.
