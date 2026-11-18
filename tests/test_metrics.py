import numpy as np
from common import add_features, best_threshold, ece_score, get_splits


def test_ece_is_zero_for_perfect_probabilities():
    y = np.array([0] * 50 + [1] * 50)
    assert ece_score(y, np.where(y == 1, 1.0, 0.0)) == 0.0


def test_ece_is_large_for_overconfident_probabilities():
    y = np.array([0, 1] * 50)
    assert ece_score(y, np.full(100, 0.95)) > 0.4


def test_best_threshold_separates_two_clean_groups():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = best_threshold(y, proba)
    assert 0.3 < threshold <= 0.7


def test_add_features_creates_the_new_columns():
    X_train, *_ = get_splits()
    out = add_features(X_train.head(200))
    for column in ["log_capital_net", "has_capital", "is_married", "overtime_hours", "is_us"]:
        assert column in out.columns
