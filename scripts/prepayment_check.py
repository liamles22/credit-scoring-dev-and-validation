"""Early full repayment, as in docs/preregistration_prepayment.md. Uses `train` and `test` only.

Payment fields define the early-repaid group and never enter a model.

Run from the project root: python -m scripts.prepayment_check
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import GOOD_STATUSES, PROJECT_ROOT, TARGET, load_interim
from src.features_config import CATEGORICAL, assert_no_leakage, model_features
from src.metrics import gini
from src.scoring import Models
from src.split import PROCESSED_PATH, SEED

TABLES = PROJECT_ROOT / "reports" / "tables"
SEEDS = [42, 1, 2, 3, 4]
MIN_GAIN = 0.005


def early_repaid(df: pd.DataFrame) -> pd.Series:
    """Fully repaid less than 12 months after issue."""
    last = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y")
    months = (last.dt.year - df["issue_d"].dt.year) * 12 + last.dt.month - df["issue_d"].dt.month
    return df["loan_status"].isin(GOOD_STATUSES) & (months < 12)


def main():
    d = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "in", ["train", "test"])])
    assert set(d["sample"].unique()) == {"train", "test"}
    d = d.merge(load_interim(columns=["id", "last_pymnt_d"]), on="id", how="left", validate="one_to_one")
    d["early_repaid"] = early_repaid(d)
    features = model_features()
    assert_no_leakage(features)
    with open(PROJECT_ROOT / "models" / "boosting_meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    cats = {c: sorted(d[c].dropna().astype(str).unique()) for c in CATEGORICAL}

    def frame(df):
        X = df[features].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype("string"), categories=cats[c])
        return X

    train, test = d[d["sample"] == "train"], d[d["sample"] == "test"]
    fit_idx, es_idx = train_test_split(train.index, test_size=0.1, stratify=train[TARGET], random_state=SEED)
    y, early = test[TARGET].to_numpy(), test["early_repaid"].to_numpy()

    by_year = d.groupby("issue_year").agg(loans=("early_repaid", "size"), early_repaid_share=("early_repaid", "mean"))
    by_year["share_of_good"] = d[d[TARGET] == 0].groupby("issue_year")["early_repaid"].mean()
    by_year.round(4).to_csv(TABLES / "prepayment_by_year.csv")
    champion = Models(cats).predict(test)["LightGBM monotone"]
    deciles = pd.qcut(pd.Series(champion).rank(method="first"), 10, labels=False) + 1
    by_risk = pd.DataFrame({"decile": deciles.to_numpy(), "early": early, "y": y}).groupby("decile").agg(
        early_repaid_share=("early", "mean"), default_rate=("y", "mean"))
    by_risk.round(4).to_csv(TABLES / "prepayment_by_risk.csv")
    print(by_year.round(4).to_string())
    print(by_risk.round(4).to_string())

    samples = {"P0": (train.loc[fit_idx], train.loc[es_idx])}
    samples["P1"] = tuple(s[~s["early_repaid"]] for s in samples["P0"])
    rows = []
    for name, (fit, es) in samples.items():
        for seed in SEEDS:
            m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, subsample_freq=1, random_state=seed,
                                   verbose=-1, n_jobs=12,
                                   monotone_constraints=[meta["directions"].get(f, 0) for f in features],
                                   **meta["lgb_params"])
            m.fit(frame(fit), fit[TARGET].to_numpy(), eval_X=(frame(es),), eval_y=(es[TARGET].to_numpy(),),
                  eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
            p = m.predict_proba(frame(test))[:, 1]
            rows.append({"variant": name, "seed": seed, "gini_without_early": gini(y[~early], p[~early]),
                         "gini_full_test": gini(y, p), "obs_pred_without_early": y[~early].mean() / p[~early].mean()})
    runs = pd.DataFrame(rows)
    runs.round(5).to_csv(TABLES / "prepayment_effect.csv", index=False)
    print(runs.groupby("variant").mean(numeric_only=True).drop(columns="seed").round(4).T.to_string())
    a, b = runs[runs.variant == "P0"].set_index("seed"), runs[runs.variant == "P1"].set_index("seed")
    gains = b["gini_without_early"] - a["gini_without_early"]
    print(f"early share in test {early.mean():.4f}; mean gain {gains.mean():.4f}; min {gains.min():.4f}; "
          f"material: {bool(gains.mean() >= MIN_GAIN and gains.min() > 0)}")


if __name__ == "__main__":
    main()
