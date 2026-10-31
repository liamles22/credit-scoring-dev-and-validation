"""Sensitivity of LightGBM monotone to fields that may carry the lender's own decisions.

Lending Club chooses whose income to verify and how to list a loan, so these fields may
partly encode its risk model (a hypothesis, see the validation report). The model is
refitted without them, on the original split and parameters, and compared on the test
sample only. OOT rows are never loaded.

Run from the project root: python -m scripts.lender_fields_sensitivity
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT, TARGET
from src.features_config import CATEGORICAL, assert_no_leakage, model_features
from src.metrics import gini
from src.split import PROCESSED_PATH, SEED

TABLES = PROJECT_ROOT / "reports" / "tables"
VARIANTS = {
    "baseline": [],
    "no verification_status": ["verification_status"],
    "no initial_list_status": ["initial_list_status"],
    "no lender fields": ["verification_status", "initial_list_status"],
}
N_BOOT = 500
SEEDS = [42, 1, 2, 3]  # refits measure training noise, which the bootstrap does not


def fit_predict(train_fit, train_es, test, features, categories, params, directions, seed=SEED) -> np.ndarray:
    def frame(df):
        X = df[features].copy()
        for c in [c for c in CATEGORICAL if c in features]:
            X[c] = pd.Categorical(X[c].astype("string"), categories=categories[c])
        return X

    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, subsample_freq=1, random_state=seed,
                           verbose=-1, n_jobs=12, monotone_constraints=[directions.get(f, 0) for f in features],
                           **params)
    m.fit(frame(train_fit), train_fit[TARGET].to_numpy(), eval_X=(frame(train_es),),
          eval_y=(train_es[TARGET].to_numpy(),), eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(frame(test))[:, 1]


def main():
    samples = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "in", ["train", "test"])])
    assert set(samples["sample"].unique()) == {"train", "test"}
    train = samples[samples["sample"] == "train"]
    test = samples[samples["sample"] == "test"]
    fit_idx, es_idx = train_test_split(train.index, test_size=0.1, stratify=train[TARGET], random_state=SEED)
    train_fit, train_es = train.loc[fit_idx], train.loc[es_idx]

    with open(PROJECT_ROOT / "models" / "boosting_meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    categories = {c: sorted(samples[c].dropna().astype(str).unique()) for c in CATEGORICAL}
    y = test[TARGET].to_numpy()

    pd_ = {}
    for name, dropped in VARIANTS.items():
        features = [f for f in model_features() if f not in dropped]
        assert_no_leakage(features)
        pd_[name] = fit_predict(train_fit, train_es, test, features, categories, meta["lgb_params"],
                                meta["directions"])
        print(f"{name}: test Gini {gini(y, pd_[name]):.4f}", flush=True)

    # Paired bootstrap on the test sample: same resampled loans for both models.
    rng = np.random.default_rng(SEED)
    boot = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    rows = []
    for name in VARIANTS:
        diffs = [gini(y[i], pd_[name][i]) - gini(y[i], pd_["baseline"][i]) for i in boot]
        rows.append({"variant": name, "gini_test": gini(y, pd_[name]),
                     "diff_vs_baseline": gini(y, pd_[name]) - gini(y, pd_["baseline"]),
                     "ci_low": np.quantile(diffs, 0.025), "ci_high": np.quantile(diffs, 0.975)})
    pd.DataFrame(rows).round(5).to_csv(TABLES / "lender_fields_sensitivity.csv", index=False)

    # Refit with other seeds: does the difference survive training randomness?
    rows = []
    for seed in SEEDS:
        for name in ["baseline", "no lender fields"]:
            features = [f for f in model_features() if f not in VARIANTS[name]]
            p = fit_predict(train_fit, train_es, test, features, categories, meta["lgb_params"],
                            meta["directions"], seed)
            rows.append({"seed": seed, "variant": name, "gini_test": gini(y, p)})
    seeds = pd.DataFrame(rows).pivot(index="seed", columns="variant", values="gini_test")
    seeds["difference"] = seeds["no lender fields"] - seeds["baseline"]
    seeds.round(5).to_csv(TABLES / "lender_fields_seeds.csv")

    # Is verification decided along the lender's grade? Share of each status by grade, train only.
    by_grade = pd.crosstab(train["grade"], train["verification_status"].astype(str), normalize="index")
    by_grade.round(4).to_csv(TABLES / "verification_by_grade.csv")


if __name__ == "__main__":
    main()
