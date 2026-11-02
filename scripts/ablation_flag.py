"""2016-field ablation with an availability flag, as in docs/preregistration_ablation_flag.md.

OOT figures are diagnostics only; no decision depends on them.

Run from the project root: python -m scripts.ablation_flag
"""
import pickle

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT, TARGET
from src.features_config import CATEGORICAL, NEW_IN_2016, assert_no_leakage, model_features
from src.metrics import gini
from src.split import PROCESSED_PATH, SEED

SEEDS = [42, 1, 2, 3, 4]
FLAG = "has_2016_fields"
WITHOUT = [f for f in model_features() if f not in NEW_IN_2016]
VARIANTS = {"C0": model_features(), "C1": WITHOUT, "C2": WITHOUT + [FLAG]}
PATTERN_MIN, CONTENT_MAX = 0.7, 0.3


def conclusion(recovered: float) -> str:
    if recovered >= PATTERN_MIN:
        return "mainly the availability pattern"
    if recovered <= CONTENT_MAX:
        return "mainly the content of the fields"
    return "mixed"


def obs_pred(y, p):
    return y.mean() / p.mean()


def main():
    samples = pd.read_parquet(PROCESSED_PATH)
    samples[FLAG] = samples[NEW_IN_2016].notna().any(axis=1).astype("int8")
    train = samples[samples["sample"] == "train"]
    test = samples[samples["sample"] == "test"]
    oot = samples[samples["sample"] == "oot"]
    fit_idx, es_idx = train_test_split(train.index, test_size=0.1, stratify=train[TARGET], random_state=SEED)
    fit, es = train.loc[fit_idx], train.loc[es_idx]
    with open(PROJECT_ROOT / "models" / "boosting_meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    cats = {c: sorted(samples[c].dropna().astype(str).unique()) for c in CATEGORICAL}
    recent = (test["issue_d"] >= "2016-01-01").to_numpy()
    y_test, y_oot = test[TARGET].to_numpy(), oot[TARGET].to_numpy()
    print("flag share:", samples.groupby("sample")[FLAG].mean().round(4).to_dict(), flush=True)

    rows = []
    for name, features in VARIANTS.items():
        assert_no_leakage(features)

        def frame(df):
            X = df[features].copy()
            for c in CATEGORICAL:
                X[c] = pd.Categorical(X[c].astype("string"), categories=cats[c])
            return X

        for seed in SEEDS:
            m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, subsample_freq=1, random_state=seed,
                                   verbose=-1, n_jobs=12,
                                   monotone_constraints=[meta["directions"].get(f, 0) for f in features],
                                   **meta["lgb_params"])
            m.fit(frame(fit), fit[TARGET].to_numpy(), eval_X=(frame(es),), eval_y=(es[TARGET].to_numpy(),),
                  eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
            p, po = m.predict_proba(frame(test))[:, 1], m.predict_proba(frame(oot))[:, 1]
            rows.append({"variant": name, "seed": seed, "obs_pred_test_2016": obs_pred(y_test[recent], p[recent]),
                         "obs_pred_test_before_2016": obs_pred(y_test[~recent], p[~recent]),
                         "gini_test": gini(y_test, p), "obs_pred_oot_diagnostic": obs_pred(y_oot, po),
                         "gini_oot_diagnostic": gini(y_oot, po)})
    runs = pd.DataFrame(rows)
    runs.round(5).to_csv(PROJECT_ROOT / "reports" / "tables" / "ablation_flag.csv", index=False)
    mean = runs.groupby("variant").mean(numeric_only=True).drop(columns="seed")
    c = mean["obs_pred_test_2016"]
    recovered = (c["C1"] - c["C2"]) / (c["C1"] - c["C0"])
    print(mean.round(4).T.to_string())
    print(f"by seed C2: {runs[runs.variant == 'C2'].obs_pred_test_2016.round(3).tolist()}")
    print(f"recovered {recovered:.3f}: {conclusion(recovered)}")


if __name__ == "__main__":
    main()
