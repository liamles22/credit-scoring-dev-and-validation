# Pre-registration: explicit training on the 2016 regime

Pushed on its own before the run. Commit dates here are not independent proof, because earlier history
was rewritten with its original dates. No result from this experiment exists at the time of writing.

## Question

Can the implicit period signal in the champion be replaced by training only on the regime in which
all fields exist? Part of the answer is already known: notebook 02 preferred the longest window on
ranking, so this tests the trade-off with calibration.

## Design

LightGBM monotone, 78 features, current parameters and directions, seeds 42, 1, 2, 3, 4. Only `train`
rows are used. The evaluation slice is H2 2016 (2016-07 to 2016-12), and early stopping uses the
last month before it, 2016-06:

| ID | Fit window | Meaning |
|---|---|---|
| W0 | 2013-07 to 2016-05 | Current design, period signal available |
| W1 | 2016-01 to 2016-05 | Regime only |
| W2 | 2015-12 to 2016-05 | Regime, including December 2015, when the fields first appear |

Metrics on the slice: Gini, and `|observed / predicted - 1|`.

## Decision rule

The regime design (W1 or W2, the one with the smaller gap) is recommended for the next redevelopment
only if both hold:

1. Its calibration gap is smaller than W0's by at least 0.02 at every seed.
2. Its mean Gini loss against W0 is at most 0.005.

Otherwise the recommendation is not made and the trade-off is reported. **The champion is not replaced
in either case**, as 2018 has been used and no untouched cohort remains. OOT 2017 and 2018 are not used.

## Limits

The slice was used before, in notebook 02 and in model v2. Parameters were tuned on the longer window,
which favours W0 on ranking.

## Changes

None. The script ran once under this plan.

## Results

Recorded on 2 November 2026. **Not recommended.**

| ID | Fit loans | Gini | Observed / predicted | Gap |
|---|---|---|---|---|
| W0 | 518,879 | 0.402 | 1.059 | 0.059 |
| W1 | 115,175 | 0.379 | 1.046 | 0.046 |
| W2 | 139,345 | 0.383 | 1.054 | 0.054 |

W1 had the smaller gap, but its calibration gain was only 0.005 at the worst seed (0.02 required) and it
lost 0.023 Gini (0.005 allowed). Training on the regime alone does not fix the level on this slice and
gives up ranking.
