"""Lifetime ranking on a later matured year, which the model was not trained on.

Fits LightGBM monotone on loans issued 2013-07 to 2014-09, stops early on 2014-10 to 2014-12,
and scores all loans issued in 2015. The 2016 bureau fields are left out, as they do not exist in
these years. Parameters are those of notebook 04, which saw 2015 loans, so this is a strong check
of the time gap but not a fully clean one. OOT rows are never loaded.

Run from the project root: python -m scripts.lifetime_temporal_check
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, TARGET, TARGET_LIFETIME
from src.features_config import CATEGORICAL, assert_no_leakage, model_features
from src.metrics import gini
from src.split import PROCESSED_PATH, issued_between

FIT, STOP, EVAL = ("2013-07", "2014-09"), ("2014-10", "2014-12"), ("2015-01", "2015-12")
SEEDS = [42, 1, 2]
N_BOOT = 500


def main():
    d = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "in", ["train", "test"])])
    assert set(d["sample"].unique()) == {"train", "test"}
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
    ev = d[issued_between(d, EVAL) & d[TARGET_LIFETIME].notna()]
    assert fit["issue_d"].max() < stop["issue_d"].min() <= stop["issue_d"].max() < ev["issue_d"].min()
    y12, life = ev[TARGET].to_numpy(), ev[TARGET_LIFETIME].to_numpy()
    bench = ev["sub_grade"].astype(str).rank(method="dense").to_numpy()

    preds = {}
    for seed in SEEDS:
        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, subsample_freq=1, random_state=seed,
                               verbose=-1, n_jobs=12,
                               monotone_constraints=[meta["directions"].get(f, 0) for f in features],
                               **meta["lgb_params"])
        m.fit(frame(fit), fit[TARGET].to_numpy(), eval_X=(frame(stop),), eval_y=(stop[TARGET].to_numpy(),),
              eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        preds[seed] = m.predict_proba(frame(ev))[:, 1]

    rng = np.random.default_rng(42)
    boot = [rng.integers(0, len(life), len(life)) for _ in range(N_BOOT)]
    rows = []
    for seed, p in preds.items():
        d12 = [gini(y12[i], p[i]) - gini(y12[i], bench[i]) for i in boot]
        dlife = [gini(life[i], p[i]) - gini(life[i], bench[i]) for i in boot]
        rows.append({"seed": seed, "loans": len(ev), "gini_12m": gini(y12, p), "gini_lifetime": gini(life, p),
                     "benchmark_12m": gini(y12, bench), "benchmark_lifetime": gini(life, bench),
                     "diff_12m": gini(y12, p) - gini(y12, bench), "diff_12m_low": np.quantile(d12, 0.025),
                     "diff_12m_high": np.quantile(d12, 0.975), "diff_lifetime": gini(life, p) - gini(life, bench),
                     "diff_lifetime_low": np.quantile(dlife, 0.025), "diff_lifetime_high": np.quantile(dlife, 0.975)})
    out = pd.DataFrame(rows).round(5)
    out.to_csv(PROJECT_ROOT / "reports" / "tables" / "lifetime_temporal.csv", index=False)
    print(f"fit {len(fit):,}, early stopping {len(stop):,}, DR lifetime {life.mean():.4f}, DR 12m {y12.mean():.4f}")
    print(out.T.to_string())


if __name__ == "__main__":
    main()
