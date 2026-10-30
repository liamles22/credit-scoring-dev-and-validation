# Model Validation Report: Lending Club 12-Month PD Models

| | |
|---|---|
| **Models reviewed** | WoE logistic scorecard; LightGBM and CatBoost, plain and monotone |
| **Purpose** | Rank applicants and estimate the probability of default at origination |
| **Evidence** | Notebooks 01–06, `reports/tables/`, 42 unit tests |
| **Out-of-time sample** | 296,484 individual 36-month loans issued in 2017 |
| **Status** | Portfolio exercise on public data. Not a real bank approval. |

**Read this first.** The developer and the validator are the same person, and an AI assistant
helped with the code. This report simulates a second-line review; it is not independent in the
regulatory sense (finding F10). "Approved" below means the evidence would support that
recommendation, not that any committee approved a model.

## 1. Executive summary

| Model | Gini OOT (95% CI) | vs benchmark (95% CI) | Observed / predicted PD | Recommendation |
|---|---|---|---|---|
| **LightGBM monotone** | **0.430** (0.421–0.439) | **+0.019** (+0.011 to +0.028) | 1.05 | **Approve with conditions** as champion for ranking |
| CatBoost monotone | 0.426 (0.417–0.434) | +0.015 (+0.006 to +0.023) | 0.99 | Approve with conditions as challenger |
| LightGBM, CatBoost (plain) | 0.428, 0.420 | +0.018, +0.010 | 1.05, 1.05 | Not recommended: no accuracy gain over monotone versions |
| WoE scorecard | 0.340 (0.330–0.349) | −0.071 (−0.081 to −0.061) | 1.19 | **Do not approve** as primary score |
| Lending Club sub-grade | 0.411 (0.402–0.420) | benchmark | n/a | Benchmark |

- **Ranking holds on new borrowers.** Gini falls by 1.2–1.8 pp from test to OOT and is stable
  across the four quarters of 2017. Score PSI is at most 0.03.
- **The scorecard fails its main success criterion.** It ranks significantly worse than the
  lender's own grade, and recalibration cannot change that.
- **PD level needed attention.** Default rates rose from 3.70% in development to 4.04% in 2017.
  Recalibration on the 2016 cohort moves the scorecard from 1.19 to 1.01 and LightGBM monotone
  from 1.05 to 1.03–1.04, though the method choice was not fully blind (F11).
- **12 findings:** 2 High, 7 Medium, 3 Low. F1 remains open at High severity; F2 is remediated.

## 2. Scope and intended use

**In scope:** ranking applicants for unsecured 36-month consumer loans at origination, and a
12-month PD after recalibration.

**Out of scope:** 60-month loans, joint applications, regulatory capital, stress testing,
lifetime expected credit loss, and any population unlike funded Lending Club borrowers.

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
  Monotone directions come from binning trends and were reviewed by the author only.
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
default rates then rose. The good OOT average may partly be offsetting errors, so it should not
be read as better calibration overall. Hosmer–Lemeshow rejects every model on OOT; at this sample
size it detects small deviations, so the observed / predicted ratio is more informative.

### 5.3 Stability and robustness

- **Score PSI train to OOT:** at most 0.027. Non-2016 features: at most 0.054.
- **2016 fields:** PSI near 5.9 against all of train, by construction; at most 0.016 against 2016
  train loans.
- **Debt settlements:** removing them changes OOT Gini by at most 0.003.

### 5.4 Interpretability

SHAP on 20,000 OOT loans ranks DTI, `loan_amnt`, accounts opened in 24 months and FICO highest.
All except `loan_amnt` are also leading scorecard drivers (F6).

### 5.5 Fairness

The data contains no protected characteristics, so fairness was **not measured directly**. As a
limited proxy, 19 states with at least 5,000 OOT loans were compared at an 85% approval cut-off.

| Model | Adverse impact ratio, minimum | Gini by state | Observed / predicted by state |
|---|---|---|---|
| LightGBM monotone | 0.96 | 0.36–0.49 | 0.75–1.22 |
| Scorecard | 0.95 | 0.24–0.41 | 0.85–1.41 |

Passing the four-fifths rule by state does not show the absence of discrimination (F7).

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
so both methods are shown. A clean confirmation needs a later matured cohort, such as 2018 loans.

## 8. Findings

**Severity.** High: do not use for the stated purpose until resolved. Medium: usable with
conditions. Low: improvement or documentation point.

| ID | Severity | Models | Finding | Status | Action |
|---|---|---|---|---|---|
| F1 | High | Scorecard | Ranks below the benchmark (−0.071 Gini) | Open | Do not use as primary score |
| F2 | High | Scorecard | PD under-predicted on OOT (1.19) | Remediated | Keep recalibration; confirm on a later cohort |
| F3 | Medium | Boosting | PD moderately under-predicted (1.05) | Within tolerance | Monitor; add a margin of conservatism |
| F4 | Medium | All | Calibration differs before and from 2016; may reflect the 2016 fields, credit policy or both | Mitigated | Re-estimate on 2016+ data |
| F5 | Medium | All | Target is reconstructed; cured delinquencies are invisible | Open | Validate against real payment history |
| F6 | Medium | Boosting | `loan_amnt` is a top driver, absent from the scorecard and unconstrained | Open | Review its shape and rationale |
| F7 | Medium | All | Fairness not directly measured; state calibration varies | Open | Fair-lending analysis before use |
| F8 | Low | Boosting | Tuning and early stopping share a holdout; CatBoost not tuned | Open | Separate tuning holdout; tune CatBoost |
| F9 | Low | All | No recession in window; funded loans only; 42 months of development data | Open | Document; add downturn and reject-inference work |
| F10 | Medium | All | Validation is not independent: same author, AI-assisted | Open | Real review by an independent party |
| F11 | Medium | All | Post-hoc choices: success criteria changed, recalibration method and the 0.9–1.1 tolerance set after seeing OOT | Open | Pre-register; confirm on an untouched cohort |
| F12 | Low | All | Explanations are untested hypotheses: why the grade is strong, why verification raises risk, why default rates rose | Open | Test or label as hypotheses |

Full evidence: [`tables/05_findings.csv`](tables/05_findings.csv).

## 9. Conditions and monitoring

**Conditions for the champion:** apply and document recalibration; resolve F6; complete a
fair-lending analysis (F7); obtain a truly independent review (F10); confirm on an untouched
matured cohort (F11).

| Metric | Frequency | Amber | Red |
|---|---|---|---|
| Score PSI against development | Quarterly | > 0.10 | > 0.25 |
| Feature PSI | Quarterly | > 0.10 | > 0.25 |
| Gini of cohorts after 12 months | Quarterly | Drop > 3 pp | Drop > 5 pp |
| Observed / predicted PD | Quarterly | Outside 0.9–1.1 | Outside 0.8–1.2 |
| Adverse impact ratio by state | Semi-annually | < 0.9 | < 0.8 |

Red triggers a model review. Two consecutive ambers trigger recalibration. These thresholds are
proposals and were set after seeing results (F11).

## 10. Model card: LightGBM monotone

| Item | Description |
|---|---|
| **Task** | Binary classification: 90+ days past due within 12 months of issue |
| **Intended use** | Ranking individual 36-month consumer loan applicants, alongside the existing grade |
| **Not intended for** | Fully automated decisions, capital, stress testing, other products or populations |
| **Algorithm** | LightGBM with monotone constraints on 21 numeric features |
| **Inputs** | 78 application and bureau fields known at origination; no state or ZIP code |
| **Training data** | 592,616 loans issued 2013-07 to 2016-12; 65,847 for early stopping and tuning |
| **Performance** | Test Gini 0.443; OOT 2017 Gini 0.430, KS 0.313 |
| **Calibration** | Observed / predicted 1.05 on OOT before recalibration, 1.03–1.04 after |
| **Main drivers** | DTI, loan amount, accounts opened in 24 months, FICO |
| **Known limits** | Proxy target, no recession, funded loans only, fairness not directly measured |
| **Explainability** | SHAP values; reason codes still need to be built and tested |
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
| Explanation of decisions (Art. 86) | SHAP analysis | No individual explanations |

**US: SR 26-2.** On 17 April 2026 the Federal Reserve, OCC and FDIC replaced SR 11-7 with revised
model risk guidance. It is risk-based, not itself enforceable, and aimed mainly at banks with over
$30 billion in assets. This report keeps the classic validation elements (conceptual soundness,
outcomes analysis, ongoing monitoring) as an organising structure.

**UK: PRA SS1/23** applies to banks with internal model permissions. Its principle on independent
validation is the one this project cannot meet (F10).

**EU: CRR.** The default definition follows Art. 178 in spirit. Art. 180 expects at least five
years of data for retail PD estimation under internal models; the 42-month window would not meet
that, so the model is not suitable for capital use.

## Sources

- [Regulation (EU) 2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng)
- [Federal Reserve SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)
- [OCC Bulletin 2026-13](https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html)
- [FDIC: Agencies Revise the Interagency Model Risk Management Guidance](https://www.fdic.gov/news/financial-institution-letters/2026/agencies-revise-interagency-model-risk-management-guidance)
