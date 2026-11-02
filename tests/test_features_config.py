"""Leakage control (src/features_config.py)."""
import pytest

from src.features import AFFORDABILITY
from src.features_config import (CANDIDATE, DERIVED, GROUPS, JOINT_ONLY, LEAKAGE, NEW_IN_2016,
                                 TARGET_ONLY_COLS, LeakageError, assert_no_leakage, check_coverage,
                                 model_features)

ALL_COLUMNS = [c for cols in GROUPS.values() for c in cols]


def test_every_raw_column_classified_exactly_once():
    assert len(ALL_COLUMNS) == 151
    check_coverage(ALL_COLUMNS)


def test_coverage_fails_on_unclassified_column():
    with pytest.raises(LeakageError, match="unclassified"):
        check_coverage(ALL_COLUMNS + ["new_column"])


def test_coverage_fails_on_missing_column():
    with pytest.raises(LeakageError, match="not in data"):
        check_coverage(ALL_COLUMNS[1:])


@pytest.mark.parametrize("column", ["recoveries", "installment", "last_pymnt_d", "loan_status",
                                    "target", "target_lifetime", "total_pymnt"])
def test_forbidden_columns_are_caught(column):
    with pytest.raises(LeakageError, match="forbidden"):
        assert_no_leakage(["fico_range_low", column])


@pytest.mark.parametrize("column", ["grade", "sub_grade", "int_rate", "funded_amnt", "zip_code",
                                    "addr_state", "id", "issue_d", "unknown_column"])
def test_allow_list_rejects_everything_outside_candidates(column):
    with pytest.raises(LeakageError, match="forbidden"):
        assert_no_leakage(["fico_range_low", column])


@pytest.mark.parametrize("group", ["LENDER_SCORE", "PROXY", "LEAKAGE", "TECHNICAL", "KEY"])
def test_every_excluded_group_is_rejected(group):
    for column in GROUPS[group]:
        with pytest.raises(LeakageError):
            assert_no_leakage([column])


def test_leakage_error_is_not_an_assertion():
    """`python -O` strips `assert`; the check must raise regardless."""
    assert not issubclass(LeakageError, AssertionError)


def test_derived_features_are_declared_and_not_raw():
    assert set(AFFORDABILITY) <= set(DERIVED)
    assert not set(DERIVED) & set(ALL_COLUMNS)
    assert_no_leakage(list(DERIVED))


def test_model_features_are_clean():
    feats = model_features()
    assert_no_leakage(feats)
    assert not set(feats) & set(LEAKAGE)
    assert not set(feats) & set(TARGET_ONLY_COLS)
    assert not set(feats) & set(JOINT_ONLY)
    assert "credit_hist_months" in feats
    assert set(feats) - {"credit_hist_months"} <= set(CANDIDATE)


def test_model_features_without_2016_fields():
    assert not set(model_features(include_2016=False)) & set(NEW_IN_2016)
