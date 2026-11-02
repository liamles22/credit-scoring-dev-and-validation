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

None. The script ran once under this plan.

## Results

Recorded on 2 November 2026. **Not material for ranking under the rule.**

Early-repaid loans are 13.75% of `test` (12.6–14.7% by issue year), and their share is nearly flat across
risk deciles (12.3–14.7%). P1 gains 0.0032 Gini on `test` without them (0.0024 at the worst seed, 0.005
required), and loses 0.0011 on full `test` (0.442 to 0.441).

Reported without a rule: observed / predicted on `test` without early-repaid loans is 1.167 for P0 and
1.006 for P1. Loans not repaid early default more often than the current PD implies, so the PD means
default among all loans, including those that exit early, not default of a loan that stays open.
