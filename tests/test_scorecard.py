"""Points scaling (src/scorecard.py)."""
import numpy as np
import pytest

from src.scorecard import BASE_ODDS, BASE_SCORE, PDO, score_from_logit


def test_base_odds_give_base_score():
    # logit of default = ln(bad / good) = -ln(BASE_ODDS)
    assert score_from_logit(-np.log(BASE_ODDS)) == pytest.approx(BASE_SCORE)


def test_doubling_the_odds_adds_pdo_points():
    s1 = score_from_logit(-np.log(BASE_ODDS))
    s2 = score_from_logit(-np.log(2 * BASE_ODDS))
    assert s2 - s1 == pytest.approx(PDO)


def test_higher_risk_means_lower_score():
    assert score_from_logit(-1.0) > score_from_logit(0.0)
