# Pre-registration: fairness proxy by ZIP3 and census data

Pushed on its own before the run. Commit dates here are not independent proof, because earlier history
was rewritten with its original dates. No result from this analysis exists at the time of writing.

## Question

The data has no protected characteristics, so fairness was not measured directly (F7). Using the racial
and ethnic composition of each borrower's 3-digit ZIP area as a proxy: do the champion's approvals and
calibration differ by group, and by more than the lender's own sub-grade?

## Data

**Census:** American Community Survey 5-year estimates 2013–2017, by ZIP Code Tabulation Area (ZCTA),
from `api.census.gov`, saved to `data/external/` and not committed. Table B03002: total population (001),
non-Hispanic White (003), non-Hispanic Black (004), Hispanic or Latino (012). Table B19013, median household
income, is kept as context and enters no rule.

**ZIP3:** ZCTA counts are summed by their first three digits and matched to Lending Club's `zip_code`
(`190xx` → `190`). Loans whose ZIP3 has no ZCTA population are excluded, and their share is reported.

**Loans:** OOT 2017, 296,484 loans. No model is refitted, tuned or selected, so OOT is only evaluated.

## Scores

| Score | PD for calibration |
|---|---|
| LightGBM monotone (champion) | Model PD, not recalibrated |
| Scorecard | Model PD, not recalibrated |
| Lending Club sub-grade | None, approval only |

## Approval

The champion approves OOT loans with PD at or below the 85th percentile of its train PD, as in the
state analysis of notebook 05. Every score approves the same number of OOT loans, the lowest-risk ones by
that score. Ties in the sub-grade at the boundary are broken at random, seed 0.

## Metrics

Each loan `i` gets weights `w_ig`, the share of group `g` (White, Black, Hispanic) in its ZIP3.

1. **Approval by group:** `r_g = sum_i(w_ig * approved_i) / sum_i(w_ig)`. Adverse impact ratio
   `AIR_g = r_g / r_White` for Black and Hispanic.
2. **Calibration by area:** ZIP3 minority share `s = (Black + Hispanic) / total`. Loans are split into
   quintiles of `s`, keeping each ZIP3 whole, so each quintile holds about 20% of loans. Observed /
   predicted is computed in each quintile for the champion and the scorecard.
3. **Against the grade:** champion AIR minus sub-grade AIR, per group.

Intervals: 95%, from 1,000 bootstrap resamples of ZIP3 areas (seed 0), as weights vary by area, not by
loan. Differences against the grade are paired within each resample.

Reported without a rule: default rate among approved loans by group with the same weights, and AIR by
quintile of median household income.

## Decision rules

Applied to the champion; rules use point estimates, and intervals are reported next to them.

1. **Approval:** AIR below 0.9 for either group is Amber; below 0.8 is Red.
2. **Calibration:** a concern if observed / predicted in the lowest minority-share quintile exceeds that
   in the highest quintile by more than 0.10, meaning the model overstates risk in high-minority areas
   relative to low-minority areas. A gap of more than 0.10 in the other direction is reported as
   understated risk, not as a concern for applicants.
3. **Against the grade:** the champion adds disparity if its AIR is lower than the sub-grade's by more
   than 0.02 for either group.

F7 stays open whatever the result. If rule 1 gives Red, F7 becomes High. If any rule flags, the champion
recommendation gains a condition; it is not replaced, and no model or cut-off changes after this run.

## Disclosed prior knowledge

State-level results on OOT were already seen in notebook 05 (minimum AIR 0.96 for the champion;
state observed / predicted 0.75–1.22). ZIP3 results, linked to census data, have not been computed.

## Limits

- ZIP3 is coarse: about 350,000 people per area on average. A geography-only proxy, without surnames,
  assigns group membership with error and pulls group differences towards zero, so AIR is biased towards 1.
- Borrowers may not resemble the residents of their area.
- Only funded loans are observed, so applicants Lending Club declined are missing.
- ZCTAs approximate ZIP codes, and some ZIP3 areas are PO boxes only.
- A clean result does not show the absence of discrimination. This is a monitoring indicator, not a
  fair-lending test.

## Code

`src/fairness.py` (tested in `tests/test_fairness.py`), `scripts/download_acs.py` and
`scripts/fairness_zip3.py`. The analysis script refuses to run if its result table exists.

## Changes

None. The script ran once under this plan. Two download attempts failed before any data was received,
because the new API key was not yet active.

## Results

Recorded on 3 November 2026. 99.96% of OOT loans matched a census ZIP3; every score approved 83.8%.

| Score | AIR, Black (95% CI) | AIR, Hispanic (95% CI) |
|---|---|---|
| LightGBM monotone | 0.996 (0.993–0.998) | 0.994 (0.991–0.997) |
| Scorecard | 0.995 (0.991–0.999) | 0.998 (0.993–1.002) |
| Sub-grade | 0.992 (0.988–0.996) | 0.994 (0.989–0.998) |

1. **Approval: Green.** Minimum AIR 0.994.
2. **Calibration: no concern for applicants by the rule, but understated risk.** Observed / predicted for
   LightGBM rises from 0.96 in the lowest minority-share quintile to 1.14 in the highest (scorecard: 1.07
   to 1.28). The gap is −0.18 (95% CI −0.25 to −0.09), beyond the 0.10 reported threshold in the other
   direction.
3. **Against the grade: no added disparity.** Champion minus sub-grade AIR: +0.004 (−0.000 to +0.008)
   for Black and +0.001 (−0.003 to +0.005) for Hispanic.

Without a rule: weighted default rate among approved loans 2.95% (Black) and 3.08% (Hispanic) against
2.85% (White); AIR within income quintiles 0.992–1.000, with no pattern. Tables:
`reports/tables/fairness_zip3_*.csv`.
