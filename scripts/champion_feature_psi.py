"""PSI of all champion inputs, train to OOT 2017. A monitoring report, not a selection step.

Numeric features use train deciles, with missing as its own bin. The 2016 bureau fields are
compared with train loans issued from 2016, because older loans did not have them.

Run from the project root: python -m scripts.champion_feature_psi
"""
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.features_config import CATEGORICAL, NEW_IN_2016, model_features
from src.metrics import psi
from src.split import PROCESSED_PATH


def bins(reference: pd.Series, new: pd.Series):
    """Same bins for both samples: deciles of `reference`, missing kept as a label."""
    if reference.name in CATEGORICAL:
        return reference.astype(object).fillna("missing"), new.astype(object).fillna("missing")
    edges = np.unique(np.nanquantile(reference.to_numpy(dtype=float), np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    cut = lambda s: pd.cut(s, edges).astype(object).astype(str)
    return cut(reference), cut(new)


def main():
    features = model_features()
    samples = pd.read_parquet(PROCESSED_PATH, columns=features + ["sample", "issue_d"])
    train, oot = samples[samples["sample"] == "train"], samples[samples["sample"] == "oot"]
    rows = []
    for f in features:
        ref = train.loc[train["issue_d"] >= "2016-01-01", f] if f in NEW_IN_2016 else train[f]
        a, b = bins(ref, oot[f])
        rows.append({"feature": f, "psi_to_oot": psi(a, b),
                     "reference": "train from 2016" if f in NEW_IN_2016 else "train"})
    out = pd.DataFrame(rows).sort_values("psi_to_oot", ascending=False)
    out["status"] = pd.cut(out["psi_to_oot"], [-1, 0.1, 0.25, 99], labels=["stable", "monitor", "significant shift"])
    out.round(5).to_csv(PROJECT_ROOT / "reports" / "tables" / "champion_feature_psi.csv", index=False)
    print(out["status"].value_counts().to_string())
    print(out.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
