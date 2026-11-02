# Pre-registration: 2016-field ablation and 2018 confirmation

Written before these calculations and before any model is scored on 2018 loans. The model v2
pre-registration planned a 2018 check only for an adopted variant, and none was adopted. This file
defines a separate check of the current models. Changes are recorded at the end of this file.

## Part A. Ablation of the 2016 bureau fields

**Question:** does boosting's calibration lead over the scorecard depend on the 14 `NEW_IN_2016`
fields, including the missing pattern that reveals older loans (finding F4)?

LightGBM monotone, same split, early-stopping holdout and parameters as notebook 04, 5 seeds
(42, 1, 2, 3, 4):

| ID | Features |
|---|---|
| C0 | Current 78 |
| C1 | Without the 14 fields |

Primary metric: observed / predicted PD on `test` loans issued in 2016.

**Rule:** the artefact explanation is supported if mean observed / predicted of C1 exceeds C0 by at
least 0.05. Otherwise it is not supported. OOT 2017 results are reported as diagnostics only. Nothing
here changes the champion.

## Part B. 2018 confirmation

**Cohort:** individual 36-month loans issued in January and February 2018, with 13–14 months between
issue and the March 2019 snapshot. Target as in `src/data.py`. March 2018 is excluded so that every
loan keeps at least one month of buffer. The sample is held in memory only and never added to
`samples.parquet`.

**Models, unchanged and not refitted:**

| Model | Recalibration, fitted in notebook 06 |
|---|---|
| LightGBM monotone | Intercept shift |
| CatBoost monotone | Logistic |
| Scorecard | Logistic |
| Lending Club sub-grade | Benchmark |

**Success criteria for LightGBM monotone:**

1. Gini is above the sub-grade, with the paired bootstrap 95% interval (1,000 resamples) above zero.
2. Recalibrated observed / predicted PD is within 0.9–1.1.

Both are reported as passed or failed, whatever the result. No model, threshold or recalibration
changes after this run. The script runs once; a rerun is allowed only to fix a crash before outcomes
are printed, or a disclosed bug.

**Limits:** the cohort is small, observation time is short, and the January 2018 default rate (about
3.6%) was already seen. The 0.9–1.1 band was itself set after seeing OOT 2017. This is supporting
evidence, not a clean new out-of-time test.

## Changes

None yet.
