"""Realised cash results on matured 2015 loans, by approval strategy.

Profit of a loan = total_pymnt - funded_amnt, using actual lifetime payments. These post-origination
columns are outcomes here and never model inputs. The model is the 2013-2014 refit of
scripts/lifetime_temporal_check.py, not the champion. It was not fitted on 2015, but its parameters
and monotone directions came from a window including 2015. Funding and operating costs are not
included, and declined applicants are not observed.

Run from the project root: python -m scripts.realised_money
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.lifetime_temporal_check import EVAL, FIT, STOP
from src.business import approved_mask, paired_difference
from src.data import PROJECT_ROOT, TARGET, TARGET_LIFETIME, load_interim
from src.features_config import CATEGORICAL, assert_no_leakage, model_features
from src.split import PROCESSED_PATH, issued_between

TABLES = PROJECT_ROOT / "reports" / "tables"
RATES = [0.7, 0.8, 0.9]


def main():
    d = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "in", ["train", "test"])])
    features = model_features(include_2016=False)
    assert_no_leakage(features)
    with open(PROJECT_ROOT / "models" / "boosting_meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    cats = {c: sorted(d[c].dropna().astype(str).unique()) for c in CATEGORICAL}

    def frame(df):
        X = df[features].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype("string"), categories=cats[c])
        return X

    fit, stop = d[issued_between(d, FIT)], d[issued_between(d, STOP)]
    ev = d[issued_between(d, EVAL) & d[TARGET_LIFETIME].notna()].copy()
    money = load_interim(columns=["id", "total_pymnt", "funded_amnt"])
    ev = ev.drop(columns="funded_amnt").merge(money, on="id", how="left", validate="one_to_one")
    assert ev["total_pymnt"].notna().all()

    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, subsample_freq=1, random_state=42, verbose=-1,
                           n_jobs=12, monotone_constraints=[meta["directions"].get(f, 0) for f in features],
                           **meta["lgb_params"])
    m.fit(frame(fit), fit[TARGET].to_numpy(), eval_X=(frame(stop),), eval_y=(stop[TARGET].to_numpy(),),
          eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    pd_model = m.predict_proba(frame(ev))[:, 1]
    grade = ev["sub_grade"].astype(str).rank(method="dense").to_numpy()
    profit = (ev["total_pymnt"] - ev["funded_amnt"]).to_numpy()
    lent = ev["funded_amnt"].to_numpy()
    y = ev[TARGET_LIFETIME].to_numpy()

    def row(name, rate, mask):
        return {"strategy": name, "approval_rate": rate, "loans": int(mask.sum()), "lifetime_default_rate": y[mask].mean(),
                "lent_musd": lent[mask].sum() / 1e6, "profit_musd": profit[mask].sum() / 1e6,
                "return_on_lent_pct": profit[mask].sum() / lent[mask].sum() * 100}

    rows, diffs = [row("Approve all", 1.0, np.ones(len(ev), bool))], {}
    for rate in RATES:
        mg, mm = approved_mask(grade, rate), approved_mask(pd_model, rate)
        rows += [row("Lending Club sub-grade", rate, mg), row("LightGBM, 2013-2014 refit", rate, mm)]
        diffs[rate] = paired_difference(y, profit, mg, mm)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(TABLES / "realised_money_2015.csv", index=False)
    diff = pd.DataFrame(diffs).T
    for c in ["profit_diff", "profit_low", "profit_high"]:
        diff[c] = diff[c] / 1e6
    diff.index.name = "approval_rate"
    diff.round(5).to_csv(TABLES / "realised_money_vs_grade.csv")
    print(out.round(3).to_string(index=False))
    print(diff.round(4).to_string())


if __name__ == "__main__":
    main()
