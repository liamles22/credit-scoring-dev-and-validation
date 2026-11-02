"""Reason codes for LightGBM monotone on 2017 loans declined at 80% approval.

The cut-off is the 80th percentile of PD on `test`. Checks: contributions add up to the model output,
reasons cover the declined loans, and reasons stay the same when the model is refitted with other
seeds. This is a prototype, not evidence of compliance. No model decision depends on it.

Run from the project root: python -m scripts.reason_code_check
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT, TARGET
from src.features_config import CATEGORICAL
from src.reason_codes import LENDER_SIDE_REASONS, top_reasons
from src.split import PROCESSED_PATH, SEED

TABLES = PROJECT_ROOT / "reports" / "tables"
MODEL = PROJECT_ROOT / "models" / "lightgbm_monotone.txt"
N_SAMPLE = 20_000
REFIT_SEEDS = [1, 2]


def main():
    samples = pd.read_parquet(PROCESSED_PATH)
    with open(PROJECT_ROOT / "models" / "boosting_meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    features = meta["features"]
    cats = {c: sorted(samples[c].dropna().astype(str).unique()) for c in CATEGORICAL}

    def frame(df):
        X = df[features].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype("string"), categories=cats[c])
        return X

    booster = lgb.Booster(model_file=str(MODEL))
    test = samples[samples["sample"] == "test"]
    oot = samples[samples["sample"] == "oot"]
    cutoff = np.quantile(booster.predict(frame(test)), 0.8)
    pd_oot = booster.predict(frame(oot))
    declined = oot[pd_oot > cutoff].sample(N_SAMPLE, random_state=42)
    X = frame(declined)

    def contributions(model):
        c = model.predict(X, pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=features, index=X.index), c

    frame_c, raw = contributions(booster)
    additivity = np.abs(raw.sum(axis=1) - booster.predict(X, raw_score=True)).max()
    reasons = top_reasons(frame_c)
    counts = pd.Series([len(r) for r in reasons])

    train = samples[samples["sample"] == "train"]
    fit_idx, es_idx = train_test_split(train.index, test_size=0.1, stratify=train[TARGET], random_state=SEED)
    fit, es = train.loc[fit_idx], train.loc[es_idx]
    first, overlap = [], []
    for seed in REFIT_SEEDS:
        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, subsample_freq=1, random_state=seed,
                               verbose=-1, n_jobs=12,
                               monotone_constraints=[meta["directions"].get(f, 0) for f in features],
                               **meta["lgb_params"])
        m.fit(frame(fit), fit[TARGET].to_numpy(), eval_X=(frame(es),), eval_y=(es[TARGET].to_numpy(),),
              eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        other = top_reasons(contributions(m.booster_)[0])
        first.append(np.mean([a[:1] == b[:1] for a, b in zip(reasons, other)]))
        overlap.append(np.mean([len(set(a) & set(b)) / max(len(set(a) | set(b)), 1) for a, b in zip(reasons, other)]))

    flat = pd.Series([r for row in reasons for r in row])
    freq = pd.DataFrame({"any_reason": flat.value_counts() / len(reasons),
                         "first_reason": pd.Series([r[0] for r in reasons if r]).value_counts() / len(reasons)}).fillna(0)
    freq = freq.sort_values("any_reason", ascending=False)
    freq.index.name = "reason"
    freq["lender_side"] = freq.index.isin(LENDER_SIDE_REASONS)
    freq.round(4).to_csv(TABLES / "reason_code_frequency.csv")
    checks = pd.Series({
        "declined_share_of_2017": float((pd_oot > cutoff).mean()), "loans_checked": len(reasons),
        "max_additivity_error": float(additivity), "share_with_4_reasons": float((counts == 4).mean()),
        "share_with_no_reason": float((counts == 0).mean()),
        "share_with_lender_side_reason": float(np.mean([bool(set(r) & LENDER_SIDE_REASONS) for r in reasons])),
        "first_reason_match_refit": float(np.mean(first)), "top4_overlap_refit": float(np.mean(overlap))})
    checks.round(5).to_csv(TABLES / "reason_code_checks.csv", header=["value"])
    pd.Series([" | ".join(r) for r in reasons[:5]], name="reasons").to_csv(TABLES / "reason_code_examples.csv", index=False)
    print(checks.round(4).to_string())
    print(freq.round(3).to_string())


if __name__ == "__main__":
    main()
