"""Applicant-level reasons (src/reason_codes.py)."""
import pandas as pd
import pytest

from src.features_config import model_features
from src.reason_codes import FEATURE_TO_REASON, REASON_GROUPS, group_contributions, top_reasons


def test_every_model_feature_has_exactly_one_reason():
    listed = [f for features in REASON_GROUPS.values() for f in features]
    assert len(listed) == len(set(listed))
    assert set(listed) == set(model_features())


def test_contributions_are_summed_within_a_reason():
    c = pd.DataFrame({"fico_range_low": [0.2], "fico_range_high": [0.1], "dti": [0.25]})
    assert group_contributions(c)["Credit score"].iloc[0] == pytest.approx(0.3)
    assert top_reasons(c, 1)[0] == ["Credit score"]


def test_only_risk_raising_reasons_are_returned():
    c = pd.DataFrame({"fico_range_low": [-0.2], "dti": [0.1], "annual_inc": [0.0]})
    assert top_reasons(c)[0] == ["Debt payments relative to income"]


def test_k_limits_reasons():
    c = pd.DataFrame({f: [0.1 + i / 100] for i, f in enumerate(["dti", "annual_inc", "loan_amnt", "purpose", "emp_length"])})
    assert len(top_reasons(c, 4)[0]) == 4


def test_mapping_values_are_known_reasons():
    assert set(FEATURE_TO_REASON.values()) == set(REASON_GROUPS)
