"""Train/test split strategies for :class:`TripletDataset`.

Private helpers orchestrated by
:meth:`ggml_ot.data.TripletDataset._train_test_split`. Each strategy
returns a list of ``(train_idx, test_idx)`` tuples over distribution
indices.
"""

from __future__ import annotations

import numpy as np

from ggml_ot.settings import settings


def _infer_test_train_size(train_size, test_size):
    """Resolve ``train_size`` / ``test_size`` for the sklearn shuffle splitters.

    Complementary fractions (one side ``None``, or both summing to 1) collapse to
    a single fraction with the other left ``None``, so sklearn derives the integer
    complement and the two counts sum to ``n_samples``. An explicit partial split
    (both given, summing to < 1) is returned unchanged.
    """
    if test_size is None or train_size is None:
        return train_size, test_size

    total = train_size + test_size
    if round(total, 2) > 1.00:
        raise ValueError(f"train_size and test_size must sum to at most 1. Got {train_size} and {test_size}.")
    if abs(total - 1.0) < 1e-9:
        return train_size, None
    return train_size, test_size


def _split_shuffle(labels, n_splits, train_size, test_size):
    """Random distribution-level split via ``ShuffleSplit``."""
    from sklearn.model_selection import ShuffleSplit

    train_size, test_size = _infer_test_train_size(train_size, test_size)
    ss = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        train_size=train_size,
        random_state=settings.random_seed,
    )
    return list(ss.split(np.zeros(len(labels))))


def _split_stratified(labels, n_splits, train_size, test_size):
    """Class-balanced split via ``StratifiedShuffleSplit``."""
    from sklearn.model_selection import StratifiedShuffleSplit

    train_size, test_size = _infer_test_train_size(train_size, test_size)
    skf = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        train_size=train_size,
        random_state=settings.random_seed,
    )
    return list(skf.split(np.zeros(len(labels)), labels))


def _split_auto(labels, n_splits, train_size, test_size):
    """Stratified if class sizes permit, else shuffle.

    Attempts :func:`_split_stratified`; on ``ValueError`` (sklearn
    raises when the least-populated class is too small for the
    requested train/test partition) falls back to
    :func:`_split_shuffle`.
    """
    try:
        return _split_stratified(labels, n_splits, train_size, test_size)
    except ValueError:
        return _split_shuffle(labels, n_splits, train_size, test_size)
