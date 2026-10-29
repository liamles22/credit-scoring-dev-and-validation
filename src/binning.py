"""WoE binning for the logistic scorecard.

Wraps optbinning's BinningProcess with one project-specific rule: fields that Lending Club
only started reporting in 2016 are binned on 2016+ loans, and older loans, for which the
field did not exist, get a neutral WoE of 0. Otherwise the "missing" bin would encode the
issue period ("missing = older loan") instead of borrower risk.
"""
import pandas as pd
from optbinning import BinningProcess

from src.data import TARGET


class ScorecardBinning:
    def __init__(self, features, categorical, late_features=(), late_start="2016-01-01",
                 min_bin_size=0.05, trends=None, n_jobs=1):
        self.features = list(features)
        self.categorical = [c for c in categorical if c in self.features]
        self.late = [f for f in late_features if f in self.features]
        self.early = [f for f in self.features if f not in self.late]
        self.late_start = pd.Timestamp(late_start)
        self.min_bin_size = min_bin_size
        self.trends = trends or {}
        self.n_jobs = n_jobs

    def _process(self, names):
        params = {v: {"monotonic_trend": self.trends[v]} for v in names if v in self.trends}
        return BinningProcess(names, categorical_variables=[c for c in self.categorical if c in names],
                              min_bin_size=self.min_bin_size, binning_fit_params=params,
                              n_jobs=self.n_jobs)

    def _is_late(self, df):
        return df["issue_d"] >= self.late_start

    def fit(self, df: pd.DataFrame):
        self.bp_early_ = self._process(self.early).fit(df[self.early], df[TARGET].to_numpy())
        self.bp_late_ = None
        if self.late:
            d = df[self._is_late(df)]
            self.bp_late_ = self._process(self.late).fit(d[self.late], d[TARGET].to_numpy())
        return self

    def binning(self, name):
        bp = self.bp_late_ if name in self.late else self.bp_early_
        return bp.get_binned_variable(name)

    def transform(self, df: pd.DataFrame, metric: str = "woe") -> pd.DataFrame:
        """WoE (or bin labels with metric="bins") for every feature.

        For late features, loans issued before `late_start` get WoE 0 (label "Not reported").
        """
        out = pd.DataFrame(index=df.index)
        for name in self.features:
            out[name] = self.binning(name).transform(df[name].to_numpy(), metric=metric)
        if self.late:
            before = ~self._is_late(df)
            out.loc[before, self.late] = 0.0 if metric == "woe" else "Not reported"
        if metric == "woe":
            # optbinning's WoE is ln(good/bad); flip the sign so that higher WoE = higher risk.
            out = -out
        return out

    def summary(self) -> pd.DataFrame:
        s = [self.bp_early_.summary()]
        if self.bp_late_ is not None:
            s.append(self.bp_late_.summary())
        s = pd.concat(s).set_index("name")[["dtype", "n_bins", "iv", "gini"]]
        s["fitted_on"] = ["2016+ loans" if n in self.late else "all train" for n in s.index]
        return s.loc[self.features]
