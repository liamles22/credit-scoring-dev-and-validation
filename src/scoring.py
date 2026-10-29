"""Load the models saved by notebook 04 and score any sample."""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostClassifier

from src.data import PROJECT_ROOT

MODELS = PROJECT_ROOT / "models"
MODEL_NAMES = ["Scorecard", "LightGBM", "LightGBM monotone", "CatBoost", "CatBoost monotone"]


def _file(name, ext):
    return MODELS / f"{name.lower().replace(' ', '_')}.{ext}"


class Models:
    def __init__(self, categories: dict):
        """`categories`: category list per categorical feature, as used in training."""
        with open(MODELS / "scorecard_logit.pkl", "rb") as fh:
            self.scorecard = pickle.load(fh)
        with open(MODELS / "boosting_meta.pkl", "rb") as fh:
            meta = pickle.load(fh)
        self.features, self.categorical = meta["features"], meta["categorical"]
        self.categories = categories
        self.lgb = {n: lgb.Booster(model_file=str(_file(n, "txt"))) for n in ["LightGBM", "LightGBM monotone"]}
        self.cb = {n: CatBoostClassifier().load_model(str(_file(n, "cbm"))) for n in ["CatBoost", "CatBoost monotone"]}

    def _lgb_frame(self, df):
        X = df[self.features].copy()
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype("string"), categories=self.categories[c])
        return X

    def _cb_frame(self, df):
        X = df[self.features].copy()
        for c in self.categorical:
            X[c] = X[c].astype(str).fillna("missing")
        return X

    def predict(self, df: pd.DataFrame) -> dict:
        """PD of every model for every row of `df`."""
        W = self.scorecard["binning"].transform(df)[self.scorecard["features"]]
        out = {"Scorecard": np.asarray(self.scorecard["model"].predict(sm.add_constant(W, has_constant="add")))}
        Xl, Xc = self._lgb_frame(df), self._cb_frame(df)
        out.update({n: b.predict(Xl) for n, b in self.lgb.items()})
        out.update({n: m.predict_proba(Xc)[:, 1] for n, m in self.cb.items()})
        return out
