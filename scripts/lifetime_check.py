"""Ranking of lifetime defaults on matured loans, for a product with a 36-month term.

The models are trained on the 12-month target. This check scores test loans issued before 2016,
which are almost all closed. It uses the development period, so it is not out-of-time evidence.
OOT rows are never loaded.

Run from the project root: python -m scripts.lifetime_check
"""
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, TARGET, TARGET_LIFETIME
from src.features_config import CATEGORICAL
from src.metrics import gini
from src.scoring import Models
from src.split import PROCESSED_PATH

N_BOOT = 500


def main():
    samples = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "in", ["train", "test"])])
    assert set(samples["sample"].unique()) == {"train", "test"}
    test = samples[(samples["sample"] == "test") & (samples["issue_d"] < "2016-01-01")]
    test = test[test[TARGET_LIFETIME].notna()]
    models = Models({c: sorted(samples[c].dropna().astype(str).unique()) for c in CATEGORICAL})
    pred = models.predict(test)
    y12, life = test[TARGET].to_numpy(), test[TARGET_LIFETIME].to_numpy()
    bench = test["sub_grade"].astype(str).rank(method="dense").to_numpy()
    scores = {"Lending Club sub-grade": bench, "Scorecard": pred["Scorecard"],
              "LightGBM monotone": pred["LightGBM monotone"], "CatBoost monotone": pred["CatBoost monotone"]}
    # Defaults after month 12 against loans that never defaulted.
    later = (life == 1) & (y12 == 0)
    keep = later | (life == 0)

    rng = np.random.default_rng(42)
    boot = [rng.integers(0, len(life), len(life)) for _ in range(N_BOOT)]
    rows = []
    for name, s in scores.items():
        d = [gini(life[i], s[i]) - gini(life[i], bench[i]) for i in boot]
        rows.append({"score": name, "gini_12m": gini(y12, s), "gini_lifetime": gini(life, s),
                     "gini_later_defaults_only": gini(later[keep].astype(int), s[keep]),
                     "lifetime_vs_benchmark": gini(life, s) - gini(life, bench),
                     "ci_low": np.quantile(d, 0.025), "ci_high": np.quantile(d, 0.975)})
    out = pd.DataFrame(rows).round(5)
    out.to_csv(PROJECT_ROOT / "reports" / "tables" / "lifetime_gini.csv", index=False)
    print(f"{len(test):,} loans, 12-month DR {y12.mean():.4f}, lifetime DR {life.mean():.4f}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
