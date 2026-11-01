"""Model v2 experiment, as pre-registered in docs/preregistration_model_v2.md.

Decisions use the Selection slice of `train` only. `test` is reported for information.
OOT rows are never loaded.

Run from the project root: python -m scripts.model_v2_experiment
"""
import json
import pickle

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.binning import ScorecardBinning
from src.data import PROJECT_ROOT, TARGET
from src.features import AFFORDABILITY, AFFORDABILITY_DIRECTIONS, add_affordability
from src.features_config import CATEGORICAL, NEW_IN_2016, assert_no_leakage, model_features
from src.metrics import gini
from src.split import PROCESSED_PATH, issued_between

TABLES = PROJECT_ROOT / "reports" / "tables"
FIT = ("2013-07", "2016-03")
TUNING = ("2016-04", "2016-06")
SELECTION = ("2016-07", "2016-12")
SEEDS = [42, 1, 2, 3, 4]
N_TRIALS = 25
N_BOOT = 500
MIN_GAIN, MAX_CALIBRATION_LOSS, TIE_GAP = 0.005, 0.02, 0.002
MIN_IV, MAX_CORR, MAX_VIF = 0.02, 0.7, 5


def obs_to_pred_gap(y, p):
    return abs(np.mean(y) / np.mean(p) - 1)


def paired_ci(y, base, new):
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        diffs.append(gini(y[i], new[i]) - gini(y[i], base[i]))
    return np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)


class Lightgbm:
    def __init__(self, data, categories, directions):
        self.data, self.categories, self.directions = data, categories, directions

    def frame(self, part, features):
        X = self.data[part][features].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype("string"), categories=self.categories[c])
        return X

    def fit(self, features, params, seed):
        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, subsample_freq=1, random_state=seed,
                               verbose=-1, n_jobs=12,
                               monotone_constraints=[self.directions.get(f, 0) for f in features], **params)
        m.fit(self.frame("fit", features), self.data["fit"][TARGET].to_numpy(),
              eval_X=(self.frame("tuning", features),), eval_y=(self.data["tuning"][TARGET].to_numpy(),),
              eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        return m

    def tune(self, features):
        def objective(trial):
            params = dict(
                num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 100, 3000, log=True),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
            )
            m = self.fit(features, params, 42)
            return gini(self.data["tuning"][TARGET], m.predict_proba(self.frame("tuning", features))[:, 1])

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_TRIALS)
        return study.best_params


def run_lightgbm(data, categories, meta):
    base = model_features()
    extended = base + AFFORDABILITY
    assert_no_leakage(extended)
    g = Lightgbm(data, categories, {**meta["directions"], **AFFORDABILITY_DIRECTIONS})
    params_a2 = g.tune(base)
    print("A2 tuned", params_a2, flush=True)
    params_a3 = g.tune(extended)
    print("A3 tuned", params_a3, flush=True)
    variants = {"A0": (base, meta["lgb_params"]), "A1": (extended, meta["lgb_params"]),
                "A2": (base, params_a2), "A3": (extended, params_a3)}
    rows, seed42 = [], {}
    y_sel, y_test = data["selection"][TARGET].to_numpy(), data["test"][TARGET].to_numpy()
    for name, (features, params) in variants.items():
        for seed in SEEDS:
            m = g.fit(features, params, seed)
            p = m.predict_proba(g.frame("selection", features))[:, 1]
            p_test = m.predict_proba(g.frame("test", features))[:, 1]
            rows.append({"variant": name, "seed": seed, "gini_selection": gini(y_sel, p),
                         "calibration_gap": obs_to_pred_gap(y_sel, p), "gini_test_info": gini(y_test, p_test)})
            if seed == 42:
                seed42[name] = p
            print(rows[-1], flush=True)
    runs = pd.DataFrame(rows)
    runs.round(5).to_csv(TABLES / "v2_lightgbm_runs.csv", index=False)
    with open(TABLES / "v2_best_params.json", "w") as fh:
        json.dump({"A2": params_a2, "A3": params_a3}, fh, indent=2)

    base_runs = runs[runs["variant"] == "A0"].set_index("seed")
    decisions = []
    for name in ["A1", "A2", "A3"]:
        r = runs[runs["variant"] == name].set_index("seed")
        gains = r["gini_selection"] - base_runs["gini_selection"]
        low, high = paired_ci(y_sel, seed42["A0"], seed42[name])
        loss = r["calibration_gap"].mean() - base_runs["calibration_gap"].mean()
        decisions.append({"variant": name, "mean_gain": gains.mean(), "min_gain": gains.min(), "ci_low": low,
                          "ci_high": high, "calibration_loss": loss, "mean_gini": r["gini_selection"].mean(),
                          "passes": bool(gains.mean() >= MIN_GAIN and gains.min() > 0 and low > 0
                                         and loss <= MAX_CALIBRATION_LOSS)})
    d = pd.DataFrame(decisions)
    passing = d[d["passes"]]
    winner = "A0"
    if len(passing):
        best = passing["mean_gini"].max()
        winner = passing[passing["mean_gini"] >= best - TIE_GAP].iloc[0]["variant"]  # A1, A2, A3 order
    return d, winner


def select_scorecard(W, y, candidates, iv):
    """Existing notebook 03 and 04 rules: correlation, VIF, then sign and p-value."""
    corr = W.loc[W.index[W.attrs["recent"]], candidates].corr()
    kept = []
    for f in sorted(candidates, key=lambda c: -iv[c]):
        if not any(abs(corr.loc[f, k]) > MAX_CORR for k in kept):
            kept.append(f)
    recent = W.loc[W.index[W.attrs["recent"]], kept].to_numpy()
    while True:
        X = np.column_stack([np.ones(len(recent)), recent])
        vif = [variance_inflation_factor(X, i + 1) for i in range(len(kept))]
        if max(vif) <= MAX_VIF:
            break
        i = int(np.argmax(vif))
        kept.pop(i)
        recent = np.delete(recent, i, axis=1)
    while True:
        lr = sm.Logit(y, sm.add_constant(W[kept])).fit(disp=0)
        coef, pval = lr.params.drop("const"), lr.pvalues.drop("const")
        if (coef <= 0).any():
            kept.remove(coef.idxmin())
        elif (pval > 0.05).any():
            kept.remove(pval.idxmax())
        else:
            return kept, lr


def run_scorecard(data):
    with open(PROJECT_ROOT / "models" / "scorecard_logit.pkl", "rb") as fh:
        current = pickle.load(fh)["features"]
    features = current + AFFORDABILITY
    numeric = [f for f in features if f not in CATEGORICAL]
    trends = {f: "auto_asc_desc" for f in numeric}
    binning = ScorecardBinning(features, CATEGORICAL, late_features=NEW_IN_2016, trends=trends,
                               n_jobs=6).fit(data["fit"])
    W = {k: binning.transform(data[k]) for k in ["fit", "selection", "test"]}
    y = {k: data[k][TARGET].to_numpy() for k in W}
    W["fit"].attrs["recent"] = (data["fit"]["issue_d"] >= "2016-01-01").to_numpy()
    iv = binning.summary()["iv"].to_dict()

    s0 = sm.Logit(y["fit"], sm.add_constant(W["fit"][current])).fit(disp=0)
    eligible = current + [f for f in AFFORDABILITY if iv[f] >= MIN_IV]
    s1_features, s1 = select_scorecard(W["fit"], y["fit"], eligible, iv)

    def evaluate(name, model, feats):
        p = np.asarray(model.predict(sm.add_constant(W["selection"][feats], has_constant="add")))
        p_test = np.asarray(model.predict(sm.add_constant(W["test"][feats], has_constant="add")))
        return p, {"variant": name, "features": len(feats), "gini_selection": gini(y["selection"], p),
                   "calibration_gap": obs_to_pred_gap(y["selection"], p), "gini_test_info": gini(y["test"], p_test)}

    p0, r0 = evaluate("S0", s0, current)
    p1, r1 = evaluate("S1", s1, s1_features)
    low, high = paired_ci(y["selection"], p0, p1)
    gain = r1["gini_selection"] - r0["gini_selection"]
    loss = r1["calibration_gap"] - r0["calibration_gap"]
    r1.update(gain=gain, ci_low=low, ci_high=high, calibration_loss=loss,
              passes=bool(gain >= MIN_GAIN and low > 0 and loss <= MAX_CALIBRATION_LOSS))
    out = pd.DataFrame([r0, r1])
    out.round(5).to_csv(TABLES / "v2_scorecard.csv", index=False)
    pd.Series({f: iv[f] for f in AFFORDABILITY}, name="iv").round(5).to_csv(TABLES / "v2_affordability_iv.csv")
    with open(TABLES / "v2_scorecard_features.json", "w") as fh:
        json.dump({"S1": s1_features, "added": [f for f in s1_features if f in AFFORDABILITY],
                   "removed": [f for f in current if f not in s1_features]}, fh, indent=2)
    return out


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    samples = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "in", ["train", "test"])])
    assert set(samples["sample"].unique()) == {"train", "test"}
    samples = add_affordability(samples)
    train = samples[samples["sample"] == "train"]
    data = {"fit": train[issued_between(train, FIT)], "tuning": train[issued_between(train, TUNING)],
            "selection": train[issued_between(train, SELECTION)], "test": samples[samples["sample"] == "test"]}
    assert sum(len(data[k]) for k in ["fit", "tuning", "selection"]) == len(train)
    print({k: (len(v), round(v[TARGET].mean(), 4)) for k, v in data.items()}, flush=True)
    categories = {c: sorted(samples[c].dropna().astype(str).unique()) for c in CATEGORICAL}
    with open(PROJECT_ROOT / "models" / "boosting_meta.pkl", "rb") as fh:
        meta = pickle.load(fh)

    scorecard = run_scorecard(data)
    print(scorecard.to_string(), flush=True)
    decisions, winner = run_lightgbm(data, categories, meta)
    decisions.round(5).to_csv(TABLES / "v2_decision.csv", index=False)
    print(decisions.round(4).to_string(), flush=True)
    print("LightGBM winner:", winner, flush=True)


if __name__ == "__main__":
    main()
