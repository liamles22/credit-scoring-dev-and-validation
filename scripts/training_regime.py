"""2016-regime training, as in docs/preregistration_2016_training.md. Uses `train` rows only.

Run from the project root: python -m scripts.training_regime
"""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, TARGET
from src.features_config import CATEGORICAL, assert_no_leakage, model_features
from src.metrics import gini
from src.split import PROCESSED_PATH, issued_between

SEEDS = [42, 1, 2, 3, 4]
WINDOWS = {"W0": ("2013-07", "2016-05"), "W1": ("2016-01", "2016-05"), "W2": ("2015-12", "2016-05")}
STOP, SLICE = ("2016-06", "2016-06"), ("2016-07", "2016-12")
MIN_CALIBRATION_GAIN, MAX_GINI_LOSS = 0.02, 0.005


def main():
    d = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "==", "train")])
    assert set(d["sample"].unique()) == {"train"}
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

    stop, slice_ = d[issued_between(d, STOP)], d[issued_between(d, SLICE)]
    y = slice_[TARGET].to_numpy()
    rows = []
    for name, window in WINDOWS.items():
        fit = d[issued_between(d, window)]
        for seed in SEEDS:
            m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, subsample_freq=1, random_state=seed,
                                   verbose=-1, n_jobs=12,
                                   monotone_constraints=[meta["directions"].get(f, 0) for f in features],
                                   **meta["lgb_params"])
            m.fit(frame(fit), fit[TARGET].to_numpy(), eval_X=(frame(stop),), eval_y=(stop[TARGET].to_numpy(),),
                  eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
            p = m.predict_proba(frame(slice_))[:, 1]
            rows.append({"window": name, "seed": seed, "fit_loans": len(fit), "gini": gini(y, p),
                         "obs_pred": y.mean() / p.mean(), "gap": abs(y.mean() / p.mean() - 1)})
    runs = pd.DataFrame(rows)
    runs.round(5).to_csv(PROJECT_ROOT / "reports" / "tables" / "training_regime.csv", index=False)
    mean = runs.groupby("window")[["fit_loans", "gini", "obs_pred", "gap"]].mean()
    print(mean.round(4).to_string())
    best = mean.loc[["W1", "W2"], "gap"].idxmin()
    a, b = runs[runs.window == "W0"].set_index("seed"), runs[runs.window == best].set_index("seed")
    gains, loss = a["gap"] - b["gap"], mean.loc["W0", "gini"] - mean.loc[best, "gini"]
    ok = bool((gains >= MIN_CALIBRATION_GAIN).all() and loss <= MAX_GINI_LOSS)
    print(f"best {best}; min calibration gain {gains.min():.4f}; mean Gini loss {loss:.4f}; recommended: {ok}")


if __name__ == "__main__":
    main()
