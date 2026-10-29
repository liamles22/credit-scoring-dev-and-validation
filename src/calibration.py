"""PD recalibration: a monotone transform of the model's log-odds, so ranking (Gini) is unchanged.

* Intercept shift (calibration in the large): logit(PD') = logit(PD) + shift, with the shift
  chosen so that the mean PD on the calibration sample equals its observed default rate.
* Logistic (Platt) recalibration: logit(PD') = a + b * logit(PD), fitted by logistic
  regression on the calibration sample; b also corrects PDs that are too steep or too flat.
"""
import numpy as np
import statsmodels.api as sm
from scipy.optimize import brentq


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


class InterceptShift:
    def fit(self, pd_, y):
        z, target = logit(pd_), float(np.mean(y))
        self.shift_ = brentq(lambda s: expit(z + s).mean() - target, -5, 5)
        return self

    def transform(self, pd_):
        return expit(logit(pd_) + self.shift_)


class LogisticRecalibration:
    def fit(self, pd_, y):
        res = sm.Logit(np.asarray(y), sm.add_constant(logit(pd_))).fit(disp=0)
        self.a_, self.b_ = res.params
        return self

    def transform(self, pd_):
        return expit(self.a_ + self.b_ * logit(pd_))
