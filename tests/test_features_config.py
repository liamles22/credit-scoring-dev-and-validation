"""Leakage control (src/features_config.py)."""
import pytest

from src.features_config import (CANDIDATE, GROUPS, JOINT_ONLY, LEAKAGE, NEW_IN_2016,
                                 TARGET_ONLY_COLS, assert_no_leakage, check_coverage,
                                 model_features)

ALL_COLUMNS = [c for cols in GROUPS.values() for c in cols]


def test_every_raw_column_classified_exactly_once():
    assert len(ALL_COLUMNS) == 151
    check_coverage(ALL_COLUMNS)


def test_coverage_fails_on_unclassified_column():
    with pytest.raises(AssertionError, match="unclassified"):
        check_coverage(ALL_COLUMNS + ["new_column"])


def test_coverage_fails_on_missing_column():
    with pytest.raises(AssertionError, match="not in data"):
        check_coverage(ALL_COLUMNS[1:])


@pytest.mark.parametrize("column", ["recoveries", "installment", "last_pymnt_d", "loan_status",
                                    "target", "target_lifetime", "total_pymnt"])
def test_forbidden_columns_are_caught(column):
    with pytest.raises(AssertionError, match="forbidden"):
        assert_no_leakage(["fico_range_low", column])


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
