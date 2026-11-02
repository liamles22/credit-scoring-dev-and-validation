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

None. Each script ran once under this plan.

## Results

Recorded on 2 November 2026.

**Part A: artefact explanation supported.** Mean observed / predicted on 2016 test loans rose from
1.016 (C0) to 1.169 (C1), a shift of 0.153 against the 0.05 rule. Without the 14 fields, OOT
observed / predicted is 1.19 (diagnostic), the scorecard's level. Test Gini fell from 0.442 to 0.431,
so these fields also carry ranking information. The experiment removes both the fields and their
missing pattern, so it cannot separate those two effects.

**Part B: LightGBM monotone passed both criteria, narrowly on the first.** Cohort: 42,162 loans, 1,495
defaults.

| Model | Gini | vs sub-grade (95% CI) | Observed / predicted, original | Recalibrated |
|---|---|---|---|---|
| LightGBM monotone | 0.450 | +0.027 (+0.001 to +0.052) | 0.96 | 0.94 |
| CatBoost monotone | 0.447 | +0.024 (−0.001 to +0.050) | 0.89 | 0.92 |
| Scorecard | 0.363 | −0.060 (−0.088 to −0.034) | 1.17 | 0.98 |
| Lending Club sub-grade | 0.423 | benchmark | n/a | n/a |

For LightGBM the intercept shift moved the PD further from observed, from 0.96 to 0.94, though inside
the band. Hosmer–Lemeshow still rejects its recalibrated PD. Short observation may also lower the
observed rate in this cohort.
