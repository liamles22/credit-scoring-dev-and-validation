# Model Validation Report: Lending Club 12-Month PD Models

| | |
|---|---|
| **Models reviewed** | WoE logistic scorecard; LightGBM and CatBoost, plain and monotone |
| **Purpose** | Rank applicants by risk of early default and estimate the 12-month probability of default at origination |
| **Evidence** | Notebooks 01–07, `reports/tables/`, unit tests, `docs/preregistration_*.md` |
| **Out-of-time sample** | 296,484 individual 36-month loans issued in 2017 |
| **Status** | Portfolio exercise on public data. Not a real bank approval. |

**Read this first.** The developer and the validator are the same person.
This report simulates a second-line review; it is not independent in the
regulatory sense (finding F10). "Approved" below means the evidence would support that
recommendation, not that any committee approved a model.

## 1. Executive summary

| Model | Gini OOT (95% CI) | vs benchmark (95% CI) | Observed / predicted PD | Recommendation |
|---|---|---|---|---|
| **LightGBM monotone** | **0.430** (0.421–0.439) | **+0.019** (+0.011 to +0.028) | 1.05 | **Conditional** champion, for ranking only |
| CatBoost monotone | 0.426 (0.417–0.434) | +0.015 (+0.006 to +0.023) | 0.99 | Conditional challenger |
| LightGBM, CatBoost (plain) | 0.428, 0.420 | +0.018, +0.010 | 1.05, 1.05 | Not recommended: no accuracy gain over monotone versions |
| WoE scorecard | 0.340 (0.330–0.349) | −0.071 (−0.081 to −0.061) | 1.19 | **Do not approve** as primary score |
| Lending Club sub-grade | 0.411 (0.402–0.420) | benchmark | n/a | Benchmark |

- **Ranking holds on new borrowers.** Gini falls by 1.2–1.8 pp from test to OOT and is stable
  across the four quarters of 2017. Score PSI is at most 0.03.
- **The scorecard fails its main success criterion.** It ranks significantly worse than the
  lender's own grade, and recalibration cannot change that.
- **PD level needed attention.** The mixed development sample has a 3.70% default rate; 2017 has
  4.04%. This is not a 2017 rise: by issue year the rate was 3.6% in 2015, 4.4% in 2016 and 4.1% in
  2017. That pattern is consistent with models fitted mostly on earlier, lower-risk years starting
  too low, but the cause is not proven (F4, F12). Recalibration on the 2016 cohort moves the scorecard from 1.19 to 1.01 and LightGBM monotone
  from 1.05 to 1.03–1.04, though the method choice was not fully blind (F11).
- **Boosting fails one criterion as written.** Its Gini drop from train to OOT is 9.2–15.8 pp
  against a 5 pp limit. It passes only on the later test-to-OOT reading (section 6), so the champion
  recommendation is conditional and should not be read as a clean pass.
- **The target is shorter than the product.** Models target 12-month default on 36-month loans.
  Matured 2013–2015 test loans default at 3.2% in 12 months and 14.2% over their life. A LightGBM
  fitted on 2013–2014 ranks 2015 loans below the sub-grade, on lifetime defaults too (section 5.7), so
  lifetime superiority is not shown.
- **A small 2018 check gives a borderline result on ranking.** On 42,162 loans from January and
  February 2018, LightGBM monotone scored Gini 0.450 against 0.423 for the sub-grade. It passed the
  pre-set rule, but the lower interval bound of +0.0008 is close enough to zero that bootstrap noise
  alone could change the verdict. Recalibrated PD stayed in the 0.9–1.1 band at 0.94 (section 5.8).
- **Boosting's calibration lead comes from a period signal.** Without the 2016 fields its OOT level
  falls to the scorecard's 1.19, and one flag showing they are available restores 97% of the lead
  (section 5.8). It is not evidence of better borrower-level calibration. Its ranking lead over the
  scorecard stays: Gini 0.422 against 0.340.
- **13 findings:** 2 High, 8 Medium, 3 Low. F1 remains open at High severity; F2 is remediated.

## 2. Scope and intended use

**In scope:** ranking applicants for unsecured 36-month consumer loans by risk of default within
12 months, and a 12-month PD after recalibration. Loans defaulting later are labelled good.

**Out of scope:** 60-month loans, joint applications, regulatory capital, stress testing,
lifetime expected credit loss, and any population unlike funded Lending Club borrowers.

**Transferability.** Lending Club was a US marketplace lender, and the data holds only loans it approved.
A bank would also have full bureau files, its own customer data and rejected applications, and in the
EU it would use local bureau scores and the EBA definition of default. The method of this report
transfers; the coefficients, cut-offs, PD level and benchmark result do not. Regulatory references in
section 11 organise the validation and are not a claim of compliance.

## 3. Data and target

| Item | Decision | Evidence |
|---|---|---|
| Source | Kaggle Lending Club accepted loans, snapshot March 2019 | Notebook 01 |
| Target | 90+ days past due within 12 months | Reconstructed from instalments paid |
| Target check | Agrees with a date-based estimate for 96.3% of defaulted 2010–2014 loans | Notebook 01, section 3.1 |
| Leakage control | All 151 raw columns classified; `assert_no_leakage` on feature lists | `src/features_config.py`, tests |
| Development window | Train and test 2013-07 to 2016-12 (42 months) | Notebook 02 |
| Out-of-time | 2017, individual applications | Notebook 02 |

The 96.3% is agreement between two reconstructions from the same payment fields. It is a
consistency check, not a measure of accuracy against true delinquency history, which the public
data does not contain.

## 4. Conceptual soundness

- **Target:** a one-year horizon and 90 days past due follow the CRR default framework
  (Art. 178). A reconstructed target is reasonable given the data, but it is a proxy (F5).
- **Training window:** chosen on a separate H2 2016 slice with a decision rule set beforehand.
  The winner led in all six validation months. The runner-up was not significantly worse.
- **Scorecard:** 21 features with positive coefficients on risk-oriented WoE, all with
  p < 0.01. Selection funnel: 78 candidates, 25 after IV and correlation filters, 21 after
  sign and significance checks. The filters cost about 0.02 Gini against all 70 binned features.
- **Boosting:** 78 raw features. LightGBM tuned with 25 Optuna trials; CatBoost not tuned (F8).
  "Monotone" means partly monotone: 21 of 78 features are constrained, and `loan_amnt`, the second
  driver by SHAP, is not. Directions come from binning trends via Spearman correlation and were
  reviewed by the author only, so they are data-derived, not independent business rules.
- **2016 bureau fields:** the scorecard neutralises them for older loans with WoE 0. Boosting
  uses them raw, so missingness may partly encode the issue period (F4).

## 5. Outcomes analysis on OOT 2017

### 5.1 Discrimination

| Model | Gini test | Gini OOT | Drop | KS OOT |
|---|---|---|---|---|
| Scorecard | 0.356 | 0.340 | 1.6 pp | 0.247 |
| LightGBM | 0.443 | 0.428 | 1.5 pp | 0.309 |
| LightGBM monotone | 0.443 | 0.430 | 1.3 pp | 0.313 |
| CatBoost | 0.438 | 0.420 | 1.8 pp | 0.306 |
| CatBoost monotone | 0.438 | 0.426 | 1.2 pp | 0.309 |
| Lending Club sub-grade | 0.417 | 0.411 | 0.6 pp | 0.301 |

Combined with the benchmark on OOT, Gini rises from 0.411 to 0.427 with the scorecard, 0.462
with CatBoost monotone and 0.465 with LightGBM monotone. Each model therefore carries
information the grade does not.

### 5.2 Calibration

| Model | Observed / predicted, test | Observed / predicted, OOT | Deciles failing binomial test, OOT |
|---|---|---|---|
| Scorecard | 1.00 | 1.19 | 8 of 10 |
| LightGBM monotone | 1.00 | 1.05 | 4 of 10 |
| CatBoost monotone | 0.96 | 0.99 | 6 of 10 |

CatBoost monotone's OOT level looks best, but it over-predicted on test (7 deciles failed) and
the 2017 level was above its development mix. The good OOT average may partly be offsetting errors, so it should not
be read as better calibration overall. Hosmer–Lemeshow rejects every model on OOT; at this sample
size it detects small deviations, so the observed / predicted ratio is more informative. It uses
n degrees of freedom on samples the models never saw; on test, LightGBM monotone sits at the
5% boundary (p = 0.050) and CatBoost monotone is rejected.

The boosting advantage over the scorecard on calibration (1.05 against 1.19) should not be read as
model quality. The ablation in section 5.8 traces it to a period signal (F4).

### 5.3 Stability and robustness

- **Score PSI train to OOT:** at most 0.027. Non-2016 features: at most 0.054.
- **2016 fields:** PSI near 5.9 against all of train, by construction; at most 0.016 against 2016
  train loans.
- **Champion inputs:** PSI train to OOT for all 78 features, with missing as its own bin, is at most
  0.055
  (`tables/champion_feature_psi.csv`). The 2016 fields use 2016 train loans as reference.
- **Debt settlements:** removing them changes OOT Gini by at most 0.003.

### 5.4 Interpretability

SHAP on 20,000 OOT loans ranks DTI, `loan_amnt`, accounts opened in 24 months and FICO highest.
All except `loan_amnt` are also leading scorecard drivers (F6).

### 5.5 Sensitivity to lender-controlled fields

`verification_status` and `initial_list_status` are set by Lending Club and may partly carry its
own risk view. LightGBM monotone was refitted without them, on train and test only.

| Variant | Gini test | Difference | 95% CI, loan sampling |
|---|---|---|---|
| Baseline | 0.4430 | n/a | n/a |
| Without `verification_status` | 0.4415 | −0.0014 | −0.0043 to +0.0013 |
| Without `initial_list_status` | 0.4406 | −0.0024 | −0.0049 to +0.0004 |
| Without both | 0.4399 | −0.0031 | −0.0062 to −0.0001 |

Removing both lowers Gini in all four seeds tested (−0.0016 to −0.0039, mean −0.0027), and baseline
Gini itself varies from 0.4414 to 0.4430 across seeds. The total effect on Gini is small, but this
does not show the model is free of lender information: other features may carry it, and effects on
individual applicants were not tested. On train, the share of Verified loans rises from 18% in
grade A to 59% in grade G. That fits the hypothesis in F12 but does not prove it, because the grade
also reflects borrower risk. The champion is unchanged, and this variant was not run on OOT.

### 5.6 Fairness

The data contains no protected characteristics, so fairness was **not measured directly**. As a
limited proxy, 19 states with at least 5,000 OOT loans were compared at an 85% approval cut-off.

| Model | Adverse impact ratio, minimum | Gini by state | Observed / predicted by state |
|---|---|---|---|
| LightGBM monotone | 0.96 | 0.36–0.49 | 0.75–1.22 |
| Scorecard | 0.95 | 0.24–0.41 | 0.85–1.41 |

Passing the four-fifths rule by state does not show the absence of discrimination (F7).

**ZIP3 proxy with census data.** Pre-registered in `docs/preregistration_fairness_zip3.md`. Each OOT loan
is weighted by the White, Black and Hispanic population shares of its 3-digit ZIP area (ACS 2013–2017;
99.96% of loans matched), and every score approves the same 83.8% of loans.

| Score | AIR, Black (95% CI) | AIR, Hispanic (95% CI) |
|---|---|---|
| LightGBM monotone | 0.996 (0.993–0.998) | 0.994 (0.991–0.997) |
| Scorecard | 0.995 (0.991–0.999) | 0.998 (0.993–1.002) |
| Sub-grade | 0.992 (0.988–0.996) | 0.994 (0.989–0.998) |

Approval is Green under the rule, and the champion adds no disparity over the grade. This is weak
evidence: group membership comes only from area composition, so differences within an area are
invisible and the ratios are pulled towards 1.

Calibration by area is the main result:

| Minority-share quintile | Mean minority share | Default rate | Observed / predicted, LightGBM | Scorecard |
|---|---|---|---|---|
| 1 | 8% | 3.66% | 0.96 | 1.07 |
| 2 | 18% | 3.86% | 1.02 | 1.16 |
| 3 | 29% | 3.99% | 1.05 | 1.19 |
| 4 | 41% | 4.21% | 1.09 | 1.24 |
| 5 | 62% | 4.48% | 1.14 | 1.28 |

Both models **understate** risk more as an area's minority share rises; for LightGBM the gap between
the first and fifth quintile is −0.18 (95% CI −0.25 to −0.09). Under the pre-set rule this is not a
concern for applicants, because their risk is not overstated. It matters in two other ways. Approved
loans in high-minority areas carry more risk than the PD shows, so one recalibration for everyone leaves
them under-provisioned. And the obvious fix, adjusting PD by location, would use a proxy for race and
risks redlining. The cause, such as local economic conditions or borrower factors missing from the
data, should be found instead. AIR within income quintiles (0.992–1.000) shows no pattern.

### 5.7 Lifetime defaults on matured loans

The models are trained on 12-month default, so this checks the product's whole term. The first table
uses 101,082 test loans issued before 2016. The boosting models were trained on those same years, while
the sub-grade was set in advance, so this comparison favours boosting.

| Score | Gini, 12 months | Gini, lifetime | Lifetime vs sub-grade (95% CI) |
|---|---|---|---|
| LightGBM monotone | 0.444 | 0.368 | +0.028 (+0.020 to +0.036) |
| CatBoost monotone | 0.436 | 0.366 | +0.026 (+0.017 to +0.034) |
| Lending Club sub-grade | 0.418 | 0.340 | benchmark |
| Scorecard | 0.360 | 0.323 | −0.017 (−0.026 to −0.008) |

A check across time reverses this. LightGBM monotone was refitted on 2013-07 to 2014-12, without the
2016 fields, and scored on 282,848 matured loans issued in 2015:

| Target | LightGBM (3 seeds) | Sub-grade | Difference, seed 42 (95% CI) |
|---|---|---|---|
| 12 months | 0.419–0.421 | 0.440 | −0.020 (−0.029 to −0.013) |
| Lifetime | 0.349 | 0.357 | −0.008 (−0.013 to −0.004) |

**Lifetime superiority over the grade is therefore not shown, and the same-period gap above is
inflated.** This refit is a weaker model than the champion: 172k fitting loans against 593k, and no
2016 fields. It does not contradict the 2017 OOT result, but it shows that the boosting lead over the
grade depends on the setup. Lifetime performance of the champion on 2017 cannot be measured yet, as
58% of those loans were still open at the snapshot.

### 5.8 What later checks showed

Nine checks followed the OOT verdict, one row each below. None changed it, but together they narrow it.
The seven pre-registered checks had their conditions pushed before running (files in `docs/`); the other
two are exploratory. Tables and details are in Appendix A.

| # | Check | Answer | Kind |
|---|---|---|---|
| 1 | 2018 confirmation: does the lead hold, and does recalibration? | Ranking barely: Gini 0.450 against 0.423 for the sub-grade, lower bound +0.0008. Recalibration held for the scorecard (1.17 to 0.98), not for LightGBM (0.96 to 0.94) (F3) | Pre-registered, `preregistration_confirmation_2018.md` part B |
| 2 | Ablation of the 14 2016 fields | Without them OOT observed / predicted is 1.19, as for the scorecard; test Gini falls by 0.011 | Pre-registered, same file, part A |
| 3 | Ablation with an availability flag | The flag alone restores 97% of the calibration shift, so boosting's calibration lead is a period signal (F4) | Pre-registered, `preregistration_ablation_flag.md` |
| 4 | Training on 2016 loans only | Gap 0.059 to 0.046, not enough, at a cost of 0.023 Gini | Pre-registered, `preregistration_2016_training.md` |
| 5 | Early full repayment | Not material for ranking (+0.0032 Gini). PD covers loans that exit early: on the other loans observed / predicted is 1.17 (F5) | Pre-registered, `preregistration_prepayment.md` |
| 6 | Model v2: affordability features, time-based tuning | LightGBM gained at most 0.002 Gini; nothing adopted | Pre-registered, `preregistration_model_v2.md` |
| 7 | ZIP3 fairness proxy with census data | Approval ratios 0.99 or above and no added disparity over the grade, but a weak test. PD understates risk more in high-minority areas: observed / predicted 0.96 to 1.14 (F7) | Pre-registered, `preregistration_fairness_zip3.md` |
| 8 | Realised money on 2015 loans | Refit at 80% approval: $224m against $206m for the sub-grade before costs, with no better default rate; approving everyone earned most ($237m) | Exploratory |
| 9 | Reason codes | Prototype; first reason stable for 90% of loans across refits; no legal review | Exploratory |

**What we learned.** LightGBM monotone ranks better than the lender's grade on 2017, by +0.019 Gini
(95% CI +0.011 to +0.028), but the margin is thin: on early 2018 its lower bound was +0.0008, and a
2013–2014 refit ranked 2015 loans below the grade (section 5.7). Its PD level is not reliable: the apparent calibration advantage comes
from the period, and neither recalibration nor regime-only training fixed it. It also understates
risk more in areas with a higher minority share. The verdict stays
"conditional champion, for ranking only, alongside the grade".

## 6. Success criteria

| Criterion | Result | Status |
|---|---|---|
| Scorecard OOT Gini above the sub-grade | 0.340 vs 0.411 | **Fail** |
| Gini drop train to OOT at most 5 pp (scorecard) | 2.4 pp | Pass |
| Gini drop train to OOT at most 5 pp (boosting, as written) | 9.2–15.8 pp | **Fail** |
| Gini drop test to OOT at most 5 pp (boosting, as reinterpreted) | 1.2–1.8 pp | Pass |
| Score PSI train to OOT below 0.1 | At most 0.027 | Pass |
| WoE signs and monotonicity agree with business logic | All 21 coefficients positive | Pass |

Boosting fails the criterion as written because in-sample Gini is inflated. Measuring it from
test is the more meaningful test, but that reinterpretation came after development results were
known and is disclosed in F11. The benchmark criterion also replaced "Gini ≥ 0.35" once the
benchmark's strength was seen.

## 7. Remediation

Recalibration was fitted only on 2016 test loans and re-tested on OOT. Gini is unchanged.

| Model | Original | Intercept shift | Logistic |
|---|---|---|---|
| Scorecard: observed / predicted (deciles failing) | 1.19 (8) | 1.02 (6) | 1.01 (4) |
| LightGBM monotone | 1.05 (4) | 1.04 (3) | 1.03 (5) |
| CatBoost monotone | 0.99 (6) | 1.02 (7) | 1.02 (4) |

A shift fitted on all of 2013–2016 test leaves the scorecard at 1.19, so the most recent cohort
matters. The notebook-preferred method per model was picked after seeing these OOT results (F11),
so both methods are shown. The later 2018 check (section 5.8) supports the scorecard's method but not LightGBM's: its shift moved PD
further from observed.

## 8. Findings

**Severity.** High: do not use for the stated purpose until resolved. Medium: usable with
conditions. Low: improvement or documentation point.

| ID | Severity | Models | Finding | Status | Action |
|---|---|---|---|---|---|
| F1 | High | Scorecard | Ranks below the benchmark (−0.071 Gini) | Open | Do not use as primary score |
| F2 | High | Scorecard | PD under-predicted on OOT (1.19) | Remediated; held on early 2018 (0.98) | Keep recalibration; recheck on larger matured cohorts |
| F3 | Medium | Boosting | PD level is unstable (1.05 on 2017, 0.96 on early 2018) and recalibration moved it further from observed on 2018 | Open | Regime-only training did not fix it (section 5.8); keep monitoring and add a margin of conservatism |
| F4 | Medium | All | Calibration differs before and from 2016. Boosting's lead is a period signal: one availability flag restores 97% of it. The underlying level shift is still unexplained | Open | Re-estimate on 2016+ data; do not credit boosting for the lead |
| F5 | Medium | All | Target is reconstructed; cured delinquencies are invisible. Early repayment (13.75% of loans) does not affect ranking materially, but defines the PD as covering loans that exit early | Open | Validate against real payment history; state the PD definition |
| F6 | Medium | Boosting | Constraints cover 21 of 78 features, with data-derived directions; `loan_amnt` is a top driver, absent from the scorecard and unconstrained | Open | Business review of directions and of the `loan_amnt` shape |
| F7 | Medium | All | Fairness only proxied. By ZIP3 and census data, AIR is 0.99 or above, but PD understates risk as an area's minority share rises (observed / predicted 0.96 to 1.14); state calibration varies | Open | Find the cause of the area gap without putting location in the model; fair-lending analysis with individual-level proxies before use |
| F8 | Low | Boosting | Tuning and early stopping share a holdout; CatBoost not tuned | Open | Separate tuning holdout; tune CatBoost |
| F9 | Low | All | No recession in window; funded loans only; 42 months of development data | Open | Document; add downturn and reject-inference work |
| F10 | Medium | All | Validation is not independent: same author | Open | Real review by an independent party |
| F11 | Medium | All | Post-hoc choices: success criteria changed, recalibration method and the 0.9–1.1 tolerance set after seeing OOT. Monotone preference was set before OOT, on explainability, but not as a written rule (plain LightGBM was 0.0002 higher on test). Rewritten history means commit dates cannot prove pre-registration order | Open | Pre-register; confirm on an untouched cohort |
| F12 | Low | All | Explanations are not proven: why the grade is strong, why verification raises risk, why default rates differ by issue year. Section 5.5 is consistent with lender information, without proving it | Open | Test or label as hypotheses |
| F13 | Medium | All | Target horizon is shorter than the product: most lifetime defaults are labelled good, and a 2015 time check shows no lifetime lead over the grade | Open | Limit use to early-default ranking; do not claim lifetime superiority; repeat when 2017 matures |

Full evidence: [`tables/05_findings.csv`](tables/05_findings.csv).

## 9. Conditions and monitoring

**Conditions for the champion:** apply and document recalibration; resolve F6; complete a
fair-lending analysis (F7); obtain a truly independent review (F10); confirm on an untouched
matured cohort (F11).

| Metric | Frequency | Amber | Red |
|---|---|---|---|
| Score PSI against development | Quarterly | > 0.10 | > 0.25 |
| Feature PSI, all 78 champion inputs | Quarterly | > 0.10 | > 0.25 |
| Gini of cohorts after 12 months | Quarterly | Drop > 3 pp | Drop > 5 pp |
| Observed / predicted PD | Quarterly | Outside 0.9–1.1 | Outside 0.8–1.2 |
| Adverse impact ratio by state and by ZIP3 proxy | Semi-annually | < 0.9 | < 0.8 |
| Observed / predicted, gap between lowest and highest minority-share quintile | Semi-annually | > 0.10 | > 0.20 |

Red triggers a model review. Two consecutive ambers trigger recalibration. These thresholds are
proposals and were set after seeing results (F11).

## 10. Model card: LightGBM monotone

| Item | Description |
|---|---|
| **Task** | Binary classification: 90+ days past due within 12 months of issue, among all loans including early exits; not lifetime default |
| **Intended use** | Ranking individual 36-month consumer loan applicants by risk of default within 12 months, alongside the existing grade; not lifetime risk |
| **Not intended for** | Fully automated decisions, capital, stress testing, other products or populations |
| **Algorithm** | LightGBM, partly monotone: constraints on 21 of 78 features |
| **Inputs** | 78 application and bureau fields known at origination; no state or ZIP code |
| **Training data** | 592,616 loans issued 2013-07 to 2016-12; 65,847 for early stopping and tuning |
| **Performance** | Test Gini 0.443; OOT 2017 Gini 0.430, KS 0.313 |
| **Calibration** | Observed / predicted 1.05 on OOT, 1.03–1.04 recalibrated; 0.96 and 0.94 on early 2018; level relies on a period signal from the 2016 fields |
| **Main drivers** | DTI, loan amount, accounts opened in 24 months, FICO |
| **Known limits** | Proxy target, no recession, funded loans only, fairness measured only by area proxies; PD understates risk more in high-minority areas |
| **Explainability** | SHAP values and prototype reason codes; first reason stable for 90% of loans across refits; no legal review |
| **Human oversight** | Required; override and escalation process still needs to be designed |

## 11. Regulatory mapping

This section is a project gap analysis, not legal advice. Article references should be checked
against the consolidated text before any real use.

**EU AI Act.** Creditworthiness evaluation of natural persons is high-risk under Annex III,
point 5(b). Regulation (EU) 2026/1744 moved the application date of these obligations from
2 August 2026 to **2 December 2027**.

| Requirement | Evidence in this project | Gap |
|---|---|---|
| Risk management (Art. 9) | Findings and monitoring plan | No lifecycle process or owner |
| Data governance (Art. 10) | Column classification, leakage checks, target checks | Bias examination incomplete |
| Technical documentation (Art. 11, Annex IV) | Notebooks, this report, model card | Not a complete Annex IV file |
| Record-keeping (Art. 12) | Git history, pinned versions | No production event logs |
| Transparency to deployers (Art. 13) | Model card and known limits | No instructions for use |
| Human oversight (Art. 14) | Stated as a requirement | Not designed |
| Accuracy, robustness, cybersecurity (Art. 15) | OOT tests, stability, settlement sensitivity | No security testing |
| Quality management (Art. 17) | Tests and version control | No formal system |
| Rights impact assessment (Art. 27) | None | Deployer must carry it out |
| Post-market monitoring (Art. 72) | Proposed monitoring table | Not implemented |
| Explanation of decisions (Art. 86) | SHAP analysis, prototype reason codes | No reviewed process or wording |

**US lending rules, the home jurisdiction of this data.** ECOA and Regulation B require specific,
accurate principal reasons for adverse action (12 CFR 1002.9), and a low score alone is not a reason.
CFPB Circular 2022-03 says complex algorithms are no exception, and FCRA key factors do not replace
those reasons. The project has a reason-code prototype (Appendix A.4) without legal review, so the champion could
not be said to meet this today. The CFPB final rule published on 22 April 2026 removed the effects test from Regulation B, but
disparate treatment and intentional proxy use remain prohibited, the Fair Housing Act and state laws
are outside that rule, and its treatment by courts is unsettled. The fairness test here therefore
stays a risk indicator, not a compliance conclusion. The status of these rules and circulars must be
rechecked before real use.

**US: SR 26-2.** On 17 April 2026 the Federal Reserve, OCC and FDIC replaced SR 11-7 with revised
model risk guidance. It is risk-based, not itself enforceable, and aimed mainly at banks with over
$30 billion in assets. This report keeps the classic validation elements (conceptual soundness,
outcomes analysis, ongoing monitoring) as an organising structure.

**UK: PRA SS1/23** applies to banks with internal model permissions. Its principle on independent
validation is the one this project cannot meet (F10).

**EU: CRR.** The default definition follows Art. 178 in spirit. Art. 180 expects at least five
years of data for retail PD estimation under internal models; the 42-month window would not meet
that, so the model is not suitable for capital use.

## Appendix A. Details of later checks

### A.1 Model v2 and business impact

- **Model v2:** a pre-registered test of affordability features and time-based tuning adopted
  nothing. LightGBM gained at most 0.002 Gini on average; the scorecard gained 0.017 but exceeded the
  calibration limit. See `docs/preregistration_model_v2.md`.
- **Business impact on 2017 loans (notebook 07):** at 80% approval the default rate is 2.76% for
  LightGBM monotone and 2.82% for the sub-grade, against 4.04% with no screening. All these loans
  were already funded by Lending Club. The 12-month profit estimate is rejected for ranking models,
  because it ignores defaults after month 12.

### A.2 Pre-registered ablation and 2018 check

Conditions were committed first (`docs/preregistration_confirmation_2018.md`).

**Ablation.** Removing the 14 2016 fields from LightGBM monotone raises observed / predicted on 2016
test loans from 1.02 to 1.17, far beyond the 0.05 rule, and OOT from 1.05 to 1.19 (diagnostic). Test
Gini falls from 0.442 to 0.431, so the fields also add ranking information, though OOT Gini stays at
0.422 against 0.340 for the scorecard.

A follow-up added one flag, "fields are available", to the model without the fields. Its conditions were
pushed before the run (`docs/preregistration_ablation_flag.md`).

| LightGBM monotone | All fields | No fields | No fields + flag |
|---|---|---|---|
| Observed / predicted, 2016 test | 1.016 | 1.169 | 1.021 |
| Gini, test | 0.442 | 0.431 | 0.440 |
| Observed / predicted, OOT diagnostic | 1.051 | 1.194 | 1.046 |
| Gini, OOT diagnostic | 0.430 | 0.422 | 0.427 |

The flag restores 97% of the calibration shift, so the rule's conclusion is **mainly the availability
pattern**. The flag is 1 for loans issued from December 2015 onward, except 44 in January 2016, and 0 before, so it works as a period indicator. The
content of the 14 fields adds little: about 0.002–0.003 Gini here. The flag's Gini gain on the pooled test
sample also partly reflects separating periods, not ranking borrowers.

**2018 check.** January and February 2018, 42,162 loans, 1,495 defaults; no model was refitted.

| Model | Gini | vs sub-grade (95% CI) | Observed / predicted | Recalibrated |
|---|---|---|---|---|
| LightGBM monotone | 0.450 | +0.027 (+0.001 to +0.052) | 0.96 | 0.94 |
| CatBoost monotone | 0.447 | +0.024 (−0.001 to +0.050) | 0.89 | 0.92 |
| Scorecard | 0.363 | −0.060 (−0.088 to −0.034) | 1.17 | 0.98 |
| Lending Club sub-grade | 0.423 | benchmark | n/a | n/a |

LightGBM met both pre-set criteria, but the first is a borderline result: the lower bound is +0.0008,
within the likely Monte Carlo error of a 1,000-resample bootstrap, so another bootstrap seed could
reverse the verdict. Its recalibration moved PD further from
observed, and Hosmer–Lemeshow still rejects it, so the method is not confirmed as an improvement. The
scorecard's recalibration held up (1.17 to 0.98). The cohort is small and short, and the January
default rate had already been seen, so this is supporting evidence only.

### A.3 Realised money on matured 2015 loans

Actual lifetime payments minus funded amount, for a LightGBM fitted on 2013–2014, not the champion.
Funding and operating costs are excluded, returns are not annualised, and declined applicants are
unobserved. This section and A.4 were not pre-registered.

| Strategy | Approval | Lent, $m | Profit, $m | Return | Lifetime default rate |
|---|---|---|---|---|---|
| Approve all | 100% | 3,622 | 237 | 6.6% | 14.9% |
| Sub-grade | 80% | 2,930 | 206 | 7.0% | 11.9% |
| LightGBM refit | 80% | 3,004 | 224 | 7.5% | 11.9% |
| Sub-grade | 70% | 2,584 | 184 | 7.1% | 10.7% |
| LightGBM refit | 70% | 2,675 | 205 | 7.7% | 10.8% |

The refit earns $18m more than the sub-grade at 80% (95% CI $16–20m) and $22m at 70% ($20–24m), with no
better default rate. So the gain comes from which loans are approved in money terms, including size
and pricing, not from better default ranking: this refit ranks defaults below the grade (section 5.7).
This is not a clean measure of model value. Rates were set from the grade, so the refit may be
keeping loans that policy underpriced, and repricing by the model would change that. About $5m of the
$18m is volume, as the refit lends $74m more. The interval covers loan sampling only: one model, seed
and year. Its parameters and directions came from a window including 2015.

Screening raises return but lowers total profit before costs, so approving everyone still earns the
most in dollars ($237m). The evidence favours higher return, not higher total profit. At 70% approval, the loans LightGBM
declines returned 3.4% and those the grade declines returned 5.2%; whether declining pays off depends on
funding and operating costs that are not in the data.

### A.4 Reason codes

A prototype groups the 78 features into 16 plain-language factors and gives up to four reasons from
LightGBM contributions, for 20,000 sampled 2017 loans above the 80% cut-off.

- Contributions add up to the model output, and every sampled loan received four reasons.
- After refits with two other seeds, the first reason matches for 90% of loans and the four-reason
  sets overlap by 87%, so about one explanation in ten would change its lead reason.
- Most frequent reasons: recently opened accounts (66%), use of available credit (48%), debt payments
  relative to income (42%). Lender-set factors appear for 3.4% of loans and should be replaced.
- A reason means "raised this applicant's score above the model average", not that fixing it would
  change the decision; that has not been tested.
- Text names a factor without a direction. `loan_amnt` has no monotone constraint, so the 24% of
  declines citing the amount have no clear meaning for the applicant.
- Reason frequency is inflated by group size: recently opened accounts sum 11 features.
- Only seed stability was tested. Wording has had no legal or consumer review.

### A.5 Regime-only training and early repayment

Both checks were pre-registered and pushed before running; neither uses OOT 2017 or 2018.

**Regime-only training: not recommended.** On H2 2016, training only on loans from 2016 cut the calibration
gap from 0.059 to 0.046, but gained only 0.005 at the worst seed (0.02 required) and lost 0.023 Gini (0.005
allowed). Replacing the period signal this way does not solve the PD level.

**Early repayment: not material for ranking, material for what PD means.** 13.75% of test loans were fully
repaid within 12 months, and the share is nearly flat across risk deciles (12.3–14.7%). Removing them from
training adds 0.0032 Gini on the remaining test loans (0.005 required). On those loans, however, observed /
predicted is 1.17 for the current design and 1.01 without early-repaid loans. The PD therefore means
default among all loans, including those that exit early. It is not the risk of a loan that stays open.
Partial prepayments remain unobservable.

## Sources

- [Regulation (EU) 2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng)
- [Federal Reserve SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)
- [OCC Bulletin 2026-13](https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html)
- [FDIC: Agencies Revise the Interagency Model Risk Management Guidance](https://www.fdic.gov/news/financial-institution-letters/2026/agencies-revise-interagency-model-risk-management-guidance)
- [Regulation B, 12 CFR Part 1002](https://www.ecfr.gov/current/title-12/chapter-X/part-1002)
- [CFPB Circular 2022-03](https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/)
- [CFPB Regulation B final rule, 22 April 2026](https://www.federalregister.gov/documents/2026/04/22/2026-07804/equal-credit-opportunity-act-regulation-b)
