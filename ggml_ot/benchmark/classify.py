import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Catch internally triggered future deprecation warning
import warnings

warnings.simplefilter("ignore", category=FutureWarning)


def knn(distances, labels, train_index, test_index, n_neighbors=5, weights=None):
    train_dists = distances[np.ix_(train_index, train_index)]
    test_to_train_dists = distances[np.ix_(test_index, train_index)]

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed", weights=weights)
    neigh.fit(train_dists, [labels[t] for t in train_index])

    predicted_labels = neigh.predict(test_to_train_dists)

    score = neigh.score(test_to_train_dists, [labels[t] for t in test_index])

    return score, predicted_labels


'''
def knn_from_distances(
    dists,
    labels,
    n_splits=20,
    distribution_labels=None,
    n_neighbors=5,
    weights=None,
    test_size=0.2,
    train_size=None,
):
    """Does a K-Nearest Neighbor classification on a given distance matrix.

    :param dists: distance matrix of shape (n_samples, n_samples)
    :type dists: array-like
    :param labels: labels corresponding to distributions of shape (num_distributions)
    :type labels: array-like
    :param n_splits: number of re-shuffling and splitting iterations, defaults to 20
    :type n_splits: int, optional
    :param distribution_labels: distribution labels for group shluffle split, defaults to None
    :type distribution_labels: array-like, optional
    :param n_neighbors: number of neighbors to use for classification, defaults to 5
    :type n_neighbors: int, optional
    :param weights: weight function to use for classification, defaults to None
    :type weights: "uniform", "distance" or callable, optional
    :param test_size: proportion of dataset to include in the test split, defaults to 0.2
    :type test_size: float, optional
    :param train_size: proportion of dataset to include in the train split, defaults to None
    :type train_size: int, optional
    :return: return predicted labels, true labels, mean accuracy on test data and labels, Rand Index and indices of test data
    :rtype: tuple of array-like
    """
    if type(dists) is not np.ndarray:
        dists = np.array(dists)
    predicted_labels, true_labels, test_indices, scores = [], [], [], []

    train_test_inds = ShuffleSplit(
        labels,
        n_splits,
        train_size,
        test_size,
        validation_size=0,
        distribution_labels=distribution_labels,
    )
    # np.arange(len(labels)))

    for i, (train_index, test_index) in enumerate(train_test_inds):
        train_dists = dists[np.ix_(train_index, train_index)]
        test_to_train_dists = dists[np.ix_(test_index, train_index)]

        neigh = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric="precomputed", weights=weights
        )
        neigh.fit(train_dists, [labels[t] for t in train_index])

        predicted_labels.append(neigh.predict(test_to_train_dists))
        true_labels.append(np.asarray([labels[t] for t in test_index]))
        scores.append(neigh.score(test_to_train_dists, true_labels[-1]))
        test_indices.append(test_index)
        # ari.append(adjusted_rand_score(predicted_labels[-1],true_labels[-1]))

    ari = adjusted_rand_score(
        np.concatenate(true_labels), np.concatenate(predicted_labels)
    )

    return predicted_labels, true_labels, scores, ari, test_indices
'''
'''
def get_dist_precomputed(precomputed_dists, ind1, ind2):
    """Retrieves a submatrix from a precomputed distance matrix.

    :param precomputed_dists: distance matrix of shape (n_samples, n_samples)
    :type precomputed_dists: array-like
    :param ind1: Row indices of the submatrix to extract
    :type ind1: array-like
    :param ind2: Column indices of the submatrix to extract
    :type ind2: array-like
    :return: Submatrix of given distance matrix
    :rtype: array-like
    """
    return precomputed_dists[ind1, :][:, ind2]
'''

'''
def ShuffleSplit(
    labels,
    n_splits=10,
    train_size=0.4,
    test_size=0.4,
    validation_size=0.2,
    distribution_labels=None,
):
    """Generates a stratified or grouped train-test(-validation) split.

    :param labels: class labels to stratify splits
    :type labels: array-like
    :param n_splits: number of re-shuffling and splitting iterations, defaults to 10
    :type n_splits: int, optional
    :param train_size: proportion of the dataset to include in the train split, defaults to 0.4
    :type train_size: float, optional
    :param test_size: proportion of dataset to include in the test split, defaults to 0.4
    :type test_size: float, optional
    :param validation_size: proportion of dataset to include in the validation split, defaults to 0.2
    :type validation_size: float, optional
    :param distribution_labels: distribution labels for group shluffle split, defaults to None
    :type distribution_labels: array-like, optional
    :return: indices of train, test data of each split
    :rtype: array-like of tuples
    """
    if validation_size > 0:
        # draw validation inds in test split and later split into two test sets
        test_size = test_size + validation_size
    if n_splits > 0:
        if distribution_labels is None:
            sss = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                train_size=train_size,
                random_state=0,
            )
            train_test_inds = sss.split(np.zeros(len(labels)), labels)
        else:
            # split patients into test and train,should be stratified
            # distribution_labels

            # unique_distribution_labels = np.unique(distribution_labels)
            # distribution_train_test_inds = sss.split(np.zeros(len(unique_distribution_labels)), unique_distribution_labels)

            gss = GroupShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                train_size=train_size,
                random_state=0,
            )
            train_test_inds = gss.split(
                np.zeros(len(labels)), labels, distribution_labels
            )

            # train_test_inds = []
            # for i, (train_distribution_index, test_distribution_index) in enumerate(distribution_train_test_inds):
            #    train_label_ind = [distribution_labels == unique_distribution_labels[train_distribution_index]]
            #    test_label_ind = [distribution_labels == unique_distribution_labels[test_distribution_index]]
            #    #for each patient split, do test and train splits between cells from both splits
            #    train = sss.split(np.zeros(len(labels)), labels)
            #    test =
    else:
        # Train = Test
        train_test_inds = [(np.arange(len(labels)), np.arange(len(labels)))]

    if validation_size > 0:
        test_size = test_size - validation_size

    return train_test_inds
'''
