"""PD recalibration (src/calibration.py)."""
import numpy as np
import pytest

from src.calibration import InterceptShift, LogisticRecalibration, expit, logit
from src.metrics import gini


@pytest.fixture
def under_predicted():
    rng = np.random.default_rng(3)
    z = rng.normal(-3.2, 0.6, 100_000)
    y = rng.random(100_000) < expit(z)
    return expit(z - 0.3), y  # model PD is too low


def test_intercept_shift_matches_observed_rate(under_predicted):
    pd_, y = under_predicted
    out = InterceptShift().fit(pd_, y).transform(pd_)
    assert out.mean() == pytest.approx(y.mean(), rel=1e-6)


def test_recalibration_keeps_ranking(under_predicted):
    pd_, y = under_predicted
    for cal in [InterceptShift(), LogisticRecalibration()]:
        out = cal.fit(pd_, y).transform(pd_)
        assert gini(y, out) == pytest.approx(gini(y, pd_))


def test_logistic_recalibration_recovers_shift(under_predicted):
    pd_, y = under_predicted
    cal = LogisticRecalibration().fit(pd_, y)
    assert cal.b_ == pytest.approx(1.0, abs=0.05)
    assert cal.a_ == pytest.approx(0.3, abs=0.1)


def test_logit_expit_roundtrip():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    assert expit(logit(p)) == pytest.approx(p)
