import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Catch internally triggered future deprecation warning
import warnings

warnings.simplefilter("ignore", category=FutureWarning)


def knn(distances, labels, train_index, test_index, n_neighbors=5, weights=None):
    if n_neighbors < 1:
        raise ValueError(f"`n_neighbors` must be >= 1, got {n_neighbors}.")

    n_train = len(train_index)
    if n_neighbors > n_train:
        raise ValueError(
            f"`n_neighbors` ({n_neighbors}) exceeds the train split size ({n_train}). "
            "Lower `knn_k` / `n_neighbors` or increase the number of training distributions."
        )

    train_dists = distances[np.ix_(train_index, train_index)]
    test_to_train_dists = distances[np.ix_(test_index, train_index)]

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed", weights=weights)
    neigh.fit(train_dists, [labels[t] for t in train_index])

    predicted_labels = neigh.predict(test_to_train_dists)

    score = neigh.score(test_to_train_dists, [labels[t] for t in test_index])

    return score, predicted_labels
