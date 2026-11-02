# Pre-registration: early full repayment and the 12-month target

Pushed on its own before the run. Commit dates here are not independent proof, because earlier history
was rewritten with its original dates. No result from this experiment exists at the time of writing.

## Question

A loan repaid in full early is labelled good, though it was not exposed to default for the whole
12 months. Does this distort the target or the ranking?

## Definitions

**Early repaid:** status `Fully Paid` (or its credit-policy equivalent) with `last_pymnt_d` less than
12 months after `issue_d`. Payment fields define this group only and never enter a model.

## Design

Only `train` and `test` rows are used; OOT 2017 and 2018 are not.

**Part 1, size.** Share of early-repaid loans by issue year and by model risk decile on `test`, and
their share among all good loans.

**Part 2, effect.** LightGBM monotone, 78 features, current parameters and directions, seeds 42, 1, 2,
3, 4:

| ID | Fit sample |
|---|---|
| P0 | Current train |
| P1 | Current train without early-repaid loans |

Primary metric: mean Gini on `test` without early-repaid loans. Secondary: Gini on full `test`, and
observed / predicted on `test` without early-repaid loans.

## Decision rule

Treatment of early repayment matters for ranking only if P1 beats P0 by at least 0.005 on average and
at every seed. Otherwise labelling these loans as good is reported as not materially affecting
ranking. Separately, the observed / predicted change is reported without a decision rule. The champion
is not replaced in either case.

## Limits

Removing these loans is a sensitivity test, not a full competing-risk model. Partial prepayments cannot
be identified this way, as the data has no payment history, and `test` is development data.

## Changes

None yet.
