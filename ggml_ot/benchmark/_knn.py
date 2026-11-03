"""k-NN scoring for benchmark evaluation."""

import warnings

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Catch internally triggered future deprecation warning.
warnings.simplefilter("ignore", category=FutureWarning)


def validate_n_neighbors(n_neighbors: int, n_train: int) -> None:
    """Fail fast on invalid k-NN configurations."""
    if n_neighbors < 1:
        raise ValueError(f"`n_neighbors` must be >= 1, got {n_neighbors}.")
    if n_neighbors > n_train:
        raise ValueError(
            f"`n_neighbors` ({n_neighbors}) exceeds the train split size ({n_train}). "
            "Lower `knn_k` / `n_neighbors` or increase the number of training distributions."
        )


def knn_classification(distances, labels, train_index, test_index, n_neighbors=5, weights="uniform"):
    validate_n_neighbors(n_neighbors, len(train_index))

    train_dists = distances[np.ix_(train_index, train_index)]
    test_to_train_dists = distances[np.ix_(test_index, train_index)]

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed", weights=weights)
    neigh.fit(train_dists, [labels[t] for t in train_index])

    predicted_labels = neigh.predict(test_to_train_dists)
    score = neigh.score(test_to_train_dists, [labels[t] for t in test_index])

    return score, predicted_labels
