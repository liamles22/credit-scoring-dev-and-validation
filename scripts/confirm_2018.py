"""Part B of docs/preregistration_confirmation_2018.md: one-time check on January-February 2018 loans.

Models and recalibration are loaded unchanged. The cohort stays in memory only. The script refuses
to run again once its result table exists.

Run from the project root: python -m scripts.confirm_2018
"""
import pickle

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, TARGET, load_interim, make_modelling_base
from src.features_config import CATEGORICAL
from src.metrics import gini, hosmer_lemeshow
from src.scoring import Models
from src.split import PROCESSED_PATH, select_scope

OUT = PROJECT_ROOT / "reports" / "tables" / "confirm_2018.csv"
METHODS = {"LightGBM monotone": "intercept shift", "CatBoost monotone": "logistic", "Scorecard": "logistic"}
N_BOOT = 1000


def main():
    if OUT.exists():
        raise SystemExit("confirm_2018.csv exists: this check runs once")
    cohort = select_scope(make_modelling_base(load_interim(), years=(2018, 2018), buffer=1))
    assert cohort["issue_d"].dt.strftime("%Y-%m").isin(["2018-01", "2018-02"]).all()
    print(f"Cohort: {len(cohort):,} loans", flush=True)
    cohort["credit_hist_months"] = ((cohort["issue_d"].dt.year - cohort["earliest_cr_line"].dt.year) * 12
                                    + cohort["issue_d"].dt.month - cohort["earliest_cr_line"].dt.month)
    ref = pd.read_parquet(PROCESSED_PATH, columns=CATEGORICAL)
    models = Models({c: sorted(ref[c].dropna().astype(str).unique()) for c in CATEGORICAL})
    pred = models.predict(cohort)
    with open(PROJECT_ROOT / "models" / "recalibration.pkl", "rb") as fh:
        calibrators = pickle.load(fh)

    y = cohort[TARGET].to_numpy()
    bench = cohort["sub_grade"].astype(str).rank(method="dense").to_numpy()
    rng = np.random.default_rng(42)
    boot = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    rows = [{"model": "Lending Club sub-grade", "gini": gini(y, bench)}]
    for name, method in METHODS.items():
        p = pred[name]
        recal = calibrators[(name, method)].transform(p)
        d = [gini(y[i], p[i]) - gini(y[i], bench[i]) for i in boot]
        rows.append({"model": name, "method": method, "gini": gini(y, p), "diff_vs_benchmark": gini(y, p) - gini(y, bench),
                     "ci_low": np.quantile(d, 0.025), "ci_high": np.quantile(d, 0.975),
                     "obs_pred_original": y.mean() / p.mean(), "obs_pred_recalibrated": y.mean() / recal.mean(),
                     "hl_p_recalibrated": hosmer_lemeshow(y, recal, out_of_sample=True)[1]})
    out = pd.DataFrame(rows)
    out.insert(1, "loans", len(y))
    out.insert(2, "defaults", int(y.sum()))
    out.round(5).to_csv(OUT, index=False)
    r = out.set_index("model").loc["LightGBM monotone"]
    print(out.round(4).T.to_string())
    print("criterion 1:", bool(r["ci_low"] > 0), "| criterion 2:", bool(0.9 <= r["obs_pred_recalibrated"] <= 1.1))


if __name__ == "__main__":
    main()
