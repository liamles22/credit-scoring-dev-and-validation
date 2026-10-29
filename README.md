# Credit Scoring: Development and Validation

🇬🇧 English · 🇷🇺 [Русский](README.ru.md)

Building and comparing credit scoring (PD) models on **Lending Club 2007–2018** data:

- **WoE logistic regression**: interpretable baseline model (scorecard)
- **LightGBM**: challenger
- **CatBoost**: challenger

Outcome: a comparison of the models on performance, stability and calibration, plus a validation report.

## Problem statement

**Goal:** estimate a borrower's probability of default (PD) at loan origination
and compare the three approaches on performance, stability, calibration and interpretability.

| Parameter | Choice |
|---|---|
| Unit | Lending Club loan, **36-month** term |
| Prediction point | Origination date (`issue_d`) |
| Horizon | **12 months** from origination (one-year PD, as in Basel / CRR) |
| Target | `1`: the loan reached **90+ days past due within its first 12 months** (CRR Art. 178 default definition)<br>`0`: everything else, including loans still open and defaults after month 12 |
| Sample | 36-month loans issued 2010–2017, individual applications, with the full 12 months observed |
| Features | Only information available at origination: application, credit history, loan terms |
| Primary metric | **Gini** on OOT |
| Additional metrics | KS, PSI, calibration (binomial test), Gini drop train → OOT |

### Why a 12-month target
The data was extracted in March 2019. Among 2016 loans 27% are still open, among
2017 loans 58%. A lifetime target ("how did the loan end?") is only known for
closed loans, and dropping open loans would leave mostly early-closed loans in the
later years and bias their default rate. With that target the sample would have to
stop at 2015.

A "default within 12 months" target is known for every loan older than 12 months,
open ones included. It is also the bank standard: regulatory PD is a one-year
probability.

| Issue year (36 m) | Loans | Open | Lifetime DR (closed loans) | 12-month DR |
|---|---|---|---|---|
| 2010 | 9,156 | 0% | 10.9% | 3.5% |
| 2011 | 14,101 | 0% | 10.6% | 3.1% |
| 2012 | 43,470 | 0% | 13.6% | 3.7% |
| 2013 | 100,422 | 0% | 12.3% | 2.9% |
| 2014 | 162,570 | 0% | 13.7% | 2.9% |
| 2015 | 283,173 | 0% | 14.9% | 3.6% |
| 2016 | 323,495 | 27% | 20.8% (biased) | 4.4% |
| 2017 | 320,419 | 58% | 23.1% (biased) | 4.1% |

### How the target is reconstructed
The public data has no monthly payment history, only the current status and
cumulative amounts. The default month is reconstructed from the number of
instalments paid:

```
months_paid   = floor((total_rec_prncp + total_rec_int) / installment)
default_month = months_paid + 1 + 3     # first missed instalment + 90 days
target        = 1 if the status is a default status and default_month <= 12
```

Check on matured 2010–2014 loans: the flag agrees with an independent estimate
from the last payment date for **96.3%** of defaulted loans. A loan enters the
sample if `issue_d + 12 months + 3 months buffer` is not later than the snapshot
(March 2019). The columns that build the target are listed in `TARGET_ONLY_COLS`
and never become features: `assert_no_leakage` checks this before training.

### Target limitations
- The target is **reconstructed**, not observed: delinquencies that were later cured
  are invisible, and partial prepayment before default pushes the default month later.
  Both slightly lower the default rate.
- Debt settlements (up to 2.5% of 2016 loans) make the payment schedule irregular;
  Part 2 includes a sensitivity test without them.
- Joint applications (7.5% of 2017 loans) are excluded: their fields only exist from late 2015.

### Success criteria
- OOT Gini of the logistic regression **above the benchmark**: Lending Club's own sub-grade on the same OOT loans (Gini ≈ 0.40); boosting must add enough uplift to justify the loss of interpretability
- Gini drop train → OOT ≤ 5 pp (≤ 10% relative)
- Score PSI train → OOT < 0.1 (stable); 0.1–0.25 acceptable with justification
- WoE signs and monotonicity in the logistic regression agree with business logic

## Data

Source: Kaggle, *Lending Club* dataset (wordsforthewise), file `accepted_2007_to_2018Q4.csv`
(2.26 million loans, 151 columns). Place it in `data/raw/`. Data is never committed to git.

**Important — leakage:** only features known at origination are used
(no `total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`, `last_fico_range_*`, etc.).

## Plan

| # | Stage | Notebook |
|---|------|---------|
| 1 | Loading, target definition, origination-time feature selection, EDA | `01_data_eda.ipynb` |
| 2 | Time-based split (`issue_d`): train / test / OOT | `02_split.ipynb` |
| 3 | Binning, WoE, IV, feature selection, correlations / VIF | `03_features.ipynb` |
| 4 | Models: LogReg (WoE), LightGBM, CatBoost, hyperparameter tuning | `04_modeling.ipynb` |
| 5 | Validation and comparison | `05_validation.ipynb` |
| 6 | Final report | `reports/` |

### Split
- **OOT**: loans issued in 2017 (~296k individual), the latest year with 12 months observed
- **The training window is chosen by experiment**, not fixed in advance. Some bureau
  fields only appear in Q2–Q4 2012 and 14 more in Q1 2016, so a long window gives more
  history and a short one more features. Candidates: H1 2016; 2015 – H1 2016;
  2013 – H1 2016, each with and without the 2016 fields. Windows are compared on a
  separate slice (H2 2016), not on OOT, so that OOT stays an honest test.
- **Result:** the longest window with all fields wins (validation Gini 0.402 against
  0.377–0.399), consistently in every month; Gini grows with window length. Its 42-month
  length is shifted to end in December 2016:

| Sample | Issued | Loans | Default rate |
|---|---|---|---|
| Train | 2013-07 … 2016-12 | 658,463 | 3.70% |
| Test (random 20%) | 2013-07 … 2016-12 | 164,616 | 3.70% |
| OOT | 2017-01 … 2017-12 | 296,484 | 4.04% |
- Loans from 2007–2009 are not used: there are few of them, and they were issued during the crisis

### Feature selection (scorecard)
Optimal binning on train (≥ 5% of loans per bin, monotonic risk trend, missing as its own
bin), then a documented funnel: IV < 0.02 → PSI / risk-order stability → correlation > 0.7 →
VIF > 5. **25 of 78 features** are kept; the decision and reason for every candidate are in
`reports/tables/03_feature_selection.csv`. The 14 bureau fields that only exist from 2016 are
binned on 2016 loans and get a neutral WoE for older loans, so that "missing" does not encode
the issue period.

### Models (test sample, before OOT)
| Model | Gini test |
|---|---|
| Scorecard: logistic regression on WoE, 21 features | 0.356 |
| LightGBM / LightGBM monotone | 0.443 / 0.443 |
| CatBoost / CatBoost monotone | 0.438 / 0.438 |
| Benchmark: Lending Club sub-grade | 0.417 |

Boosting beats the lender's own grade; the scorecard does not, because the grade uses
information that is not in the public data. Monotone constraints cost no accuracy. Every
model still **adds** information to the grade: combined with it, LightGBM lifts Gini from
0.417 to 0.472 and the scorecard to 0.435. The final verdict is given on OOT (2017) in Part 2.

### Part 2: independent validation on OOT 2017
| Model | Gini OOT | vs benchmark | Observed / predicted PD | Verdict |
|---|---|---|---|---|
| Scorecard | 0.340 | −0.071 | 1.19 | **Not approved** as primary score |
| **LightGBM monotone** | **0.430** | **+0.019** | 1.05 | **Approved with conditions** (champion) |
| CatBoost monotone | 0.426 | +0.015 | 0.99 | Approved with conditions (challenger) |
| Benchmark: sub-grade | 0.411 | — | — | — |

Ranking holds on new borrowers (Gini drop test → OOT ≤ 1.8 pp, score PSI ≤ 0.03), but PD
is under-predicted as default rates rose in 2017: recalibration is required before any PD use.
9 findings (2 High, 5 Medium, 2 Low) with evidence and recommendations are in
`reports/tables/05_findings.csv`, together with a monitoring plan in notebook 05.

### Validation metrics
- **Discrimination:** Gini, KS (train / test / OOT), Gini drop
- **Stability:** PSI of the score and features, Gini by quarter
- **Calibration:** predicted vs observed DR by bucket, binomial test, Hosmer–Lemeshow
- **Interpretability:** coefficients and WoE monotonicity (LogReg), SHAP (boosting)

## Structure

```
data/
  raw/          # source data (not in git)
  interim/      # intermediate (parquet)
  processed/    # final samples
notebooks/      # one notebook per stage
src/            # reusable code: binning, metrics, PSI
models/         # saved models
reports/        # report and figures
```

## Environment

```bash
conda activate ml
pip install -r requirements.txt
```
