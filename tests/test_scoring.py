"""Saved models (src/scoring.py). Needs local models and data, so CI skips it."""
import numpy as np
import pandas as pd
import pytest

from src.data import PROJECT_ROOT, TARGET
from src.features_config import CATEGORICAL
from src.metrics import gini
from src.scoring import MODEL_NAMES, MODELS
from src.split import PROCESSED_PATH

COMPARISON = PROJECT_ROOT / "reports" / "tables" / "04_model_comparison.csv"
REPORTED_NAMES = {"Scorecard": "Scorecard (logistic WoE)"}

pytestmark = pytest.mark.skipif(
    not (PROCESSED_PATH.exists() and (MODELS / "boosting_meta.pkl").exists()),
    reason="models and processed data are not in git")


@pytest.fixture(scope="module")
def scored():
    from src.scoring import Models

    samples = pd.read_parquet(PROCESSED_PATH, filters=[("sample", "in", ["train", "test"])])
    models = Models({c: sorted(samples[c].dropna().astype(str).unique()) for c in CATEGORICAL})
    test = samples[samples["sample"] == "test"].reset_index(drop=True)
    return models, test, models.predict(test)


def test_every_model_scores_every_row(scored):
    _, test, pd_ = scored
    assert list(pd_) == MODEL_NAMES
    for name, p in pd_.items():
        assert len(p) == len(test), name
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), name


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_test_gini_matches_notebook_04(scored, name):
    _, test, pd_ = scored
    reported = pd.read_csv(COMPARISON).set_index("model").loc[REPORTED_NAMES.get(name, name), "gini_test"]
    assert gini(test[TARGET].to_numpy(), pd_[name]) == pytest.approx(reported, abs=5e-4)


def test_column_and_row_order_do_not_matter(scored):
    models, test, pd_ = scored
    part = test.iloc[:2000]
    shuffled = part.sample(frac=1, random_state=0)[part.columns[::-1]]
    again = models.predict(shuffled)
    for name in MODEL_NAMES:
        np.testing.assert_allclose(again[name], pd_[name][shuffled.index], rtol=1e-6, err_msg=name)
