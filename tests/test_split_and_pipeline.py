import numpy as np

from common import build_pipeline, get_splits


def test_split_sizes_and_stratification():
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
    total = len(X_train) + len(X_val) + len(X_test)
    assert abs(len(X_train) / total - 0.6) < 0.001
    assert abs(len(X_val) / total - 0.2) < 0.001
    assert abs(len(X_test) / total - 0.2) < 0.001
    rates = [y_train.mean(), y_val.mean(), y_test.mean()]
    assert max(rates) - min(rates) < 0.001


def test_no_row_in_two_splits():
    X_train, X_val, X_test, *_ = get_splits()
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_val.index) & set(X_test.index)


def test_pipeline_contains_the_preprocessing():
    pipe = build_pipeline("logreg")
    assert [name for name, _ in pipe.steps] == ["features", "prep", "model"]


def test_pipeline_predicts_probabilities():
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
    pipe = build_pipeline("logreg").fit(X_train.iloc[:3000], y_train.iloc[:3000])
    proba = pipe.predict_proba(X_val.iloc[:200])[:, 1]
    assert proba.shape == (200,)
    assert np.all((proba >= 0) & (proba <= 1))
