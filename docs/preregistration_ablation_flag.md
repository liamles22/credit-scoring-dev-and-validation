# Pre-registration: 2016-field ablation with an availability flag

This file is pushed to GitHub on its own before the run, so that the push and its CI run can be seen
before the results exist. Commit dates in this repository are not independent proof: earlier history
was rewritten with its original dates. No result from this experiment exists at the time of writing.

## Question

The earlier ablation removed the 14 `NEW_IN_2016` fields and their missing pattern together. Does the
calibration effect come from what the fields contain, or from the pattern that shows when a loan was
issued?

## Design

LightGBM monotone, same split, early-stopping holdout, parameters and five seeds (42, 1, 2, 3, 4) as
`scripts/ablation_2016_fields.py`:

| ID | Features |
|---|---|
| C0 | Current 78 |
| C1 | Without the 14 fields |
| C2 | C1 + `has_2016_fields`, equal to 1 if any of the 14 fields is filled, unconstrained |

Primary metric: mean observed / predicted PD on `test` loans issued in 2016.

## Decision rule

`recovered = (C1 - C2) / (C1 - C0)`, the share of the C1 calibration shift that the flag alone restores.

| Recovered | Conclusion |
|---|---|
| At least 0.7 | Mainly the availability pattern, which reveals the period |
| At most 0.3 | Mainly the content of the fields |
| Between | Mixed; neither explanation is claimed |

OOT 2017 figures are diagnostics only. Almost every 2017 loan has the flag equal to 1, so the flag
cannot show drift inside 2017. Nothing here changes the champion.

## Limits

The earlier ablation results are known. The flag is an imperfect stand-in for the pattern, as
it does not record which fields are missing.

## Changes

None. The script ran once under this plan, after this file was pushed in commit `4d526f1`.

## Results

Recorded on 2 November 2026. **Conclusion under the rule: mainly the availability pattern.**

| Metric | C0 | C1 | C2 |
|---|---|---|---|
| Observed / predicted, test 2016 | 1.016 | 1.169 | 1.021 |
| Observed / predicted, test before 2016 | 0.995 | 0.895 | 0.990 |
| Gini, test | 0.442 | 0.431 | 0.440 |
| Observed / predicted, OOT diagnostic | 1.051 | 1.194 | 1.046 |
| Gini, OOT diagnostic | 0.430 | 0.422 | 0.427 |

`recovered = 0.969`. The flag equals 1 for all 2017 loans and, in development, for loans issued from
December 2015 onward, except 44 in January 2016, so it works as a period indicator. On random test data it also restores most of the Gini
loss, so part of that pooled test Gini reflects the period, not borrower ranking.
