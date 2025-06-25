from sklearn.metrics import (
    confusion_matrix,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import numpy as np
import seaborn as sns
import numpy.typing as npt
import matplotlib.pyplot as plt
from IPython.display import display
import warnings

from ggml_ot.distances import Computed_Distances, compute_OT


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
            # print("shapes")
            # print(np.zeros(len(labels)).shape)
            # print(labels.shape)
            # print(distribution_labels.shape)
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


def knn_from_dists(
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
        # print(train_index)
        # print(test_index)
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


def silhouette_score_wrapper(dists, labels):
    """Compute the mean Silhouette score of the given distance matrix. The silhouette score is a measure
    that tells how well each point fits in its own cluster compared to other clusters (1 best to -1 worst score).
    It is computed by: (b - a) / max(a, b) where a is the intra-cluster distance and b is the nearest-cluster
    distance.

    :param dists: distance matrix of shape (n_samples, n_samples)
    :type dists: array-like
    :param labels: labels corresponding to distributions of shape (num_distributions)
    :type labels: array-like
    :return: silhouette score
    :rtype: float
    """
    # wrapper function as fill diagonal is only available as inplace operation. It is needed
    # to catch cases where due to numerical errors the distance of a graph to itself may be
    # very close to zero, but not zero which is required by sklearn silhoute score method
    zero_dia_dists = np.copy(dists)
    np.fill_diagonal(zero_dia_dists, 0)
    return silhouette_score(zero_dia_dists, labels, metric="precomputed")


# def compute_dists(Graphs, Graphs2=None, method="TiedOT"):
#     # TODO: can we remove this?
#     dist, plan = methods[method](Graphs, Graphs2)
#     dist[dist < 0] = 0
#     return dist


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


def plot_1split(predicted, true, title=None, ax=None):
    """Plots a heatmap of the confusion matrix of given predicted and true labels.

    :param predicted: predicted labels
    :type predicted: array-like
    :param true: labels of ground truth
    :type true: array-like
    :param title: title of the plot, defaults to None
    :type title: str, optional
    :param ax: axis to plot on, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    """
    annot_labels_ind = np.unique(true, return_index=True)[1]
    annot_labels = true[annot_labels_ind]
    # ind
    cf_matrix = confusion_matrix(true, predicted, labels=annot_labels)
    if ax is None:
        plt.figure()
    ax = sns.heatmap(
        cf_matrix,
        annot=True,  # fmt='.0',
        cmap="Blues",
        xticklabels=annot_labels,
        yticklabels=annot_labels,
        ax=ax,
        fmt="g",
    )
    ax.set(xlabel="Predicted Label", ylabel="True Label")
    ax.set_title(title)


def plot_table(df, tranpose=False, print_latex=True):
    """Displays a DataFrame as a formatted LaTeX table.

    :param df: data to display
    :type df: DataFrame
    :param tranpose: whether to transpose the table, defaults to False
    :type tranpose: bool, optional
    """
    format_df = df
    format_df.set_index("method", inplace=True)
    if tranpose:
        format_df = format_df.transpose()
    display(format_df)
    if print_latex:
        print(
            format_df.to_latex(
                index=True,
                # formatters={"name": str.upper},
                float_format="{:.2f}".format,
            )
        )


def VI(
    labels1: npt.NDArray[np.int32],
    labels2: npt.NDArray[np.int32],
    # torch: bool = True,
    # device: str = "cpu",
    return_split_merge: bool = False,
):
    """
    Calculates the Variation of Information between two clusterings.

    Arguments:
    labels1: flat int32 array of labels for the first clustering
    labels2: flat int32 array of labels for the second clustering
    return_split_merge: whether to return split and merge terms, default:False

    Returns:
    vi: variation of information
    vi_split: split term of variation of information
    vi_merge: merge term of variation of information
    splitters(optional): labels of labels2 which are split by labels1. splitters[i,0] is the contribution of the i-th splitter to the VI and splitters[i,1] is the corresponding label of the splitter
    mergers(optional): labels of labels1 which are merging labels from labels2. mergers[i,0] is the contribution of the i-th merger to the VI and mergers[i,1] is the corresponding label of the merger
    """
    if labels1.ndim > 1 or labels2.ndim > 1:
        warnings.warn(
            f"Inputs of shape {labels1.shape}, {labels2.shape} are not one-dimensional -- inputs will be flattened."
        )
        labels1 = labels1.flatten()
        labels2 = labels2.flatten()

    # if torch:
    #     # TODO soll torch Version noch rein?
    #     return VI_torch(
    #         labels1, labels2, device=device, return_split_merge=return_split_merge
    #     )
    else:
        return VI_np(labels1, labels2, return_split_merge=return_split_merge)


def VI_np(labels1, labels2, return_split_merge=False):
    assert len(labels2) == len(labels1)
    size = len(labels2)

    mutual_labels = (labels1.astype(np.uint64) << 32) + labels2.astype(np.uint64)

    sm_unique, sm_inverse, sm_counts = np.unique(
        labels2, return_inverse=True, return_counts=True
    )
    fm_unique, fm_inverse, fm_counts = np.unique(
        labels1, return_inverse=True, return_counts=True
    )
    _, mutual_inverse, mutual_counts = np.unique(
        mutual_labels, return_inverse=True, return_counts=True
    )

    terms_mutual = -np.log(mutual_counts / size) * mutual_counts / size
    terms_mutual_per_count = (
        terms_mutual[mutual_inverse] / mutual_counts[mutual_inverse]
    )
    terms_sm = -np.log(sm_counts / size) * sm_counts / size
    terms_fm = -np.log(fm_counts / size) * fm_counts / size
    if not return_split_merge:
        terms_mutual_sum = np.sum(terms_mutual_per_count)
        vi_split = terms_mutual_sum - terms_sm.sum()
        vi_merge = terms_mutual_sum - terms_fm.sum()
        vi = vi_split + vi_merge
        return vi, vi_split, vi_merge

    vi_split_each = np.zeros(len(sm_unique))
    np.add.at(vi_split_each, sm_inverse, terms_mutual_per_count)
    vi_split_each -= terms_sm
    vi_merge_each = np.zeros(len(fm_unique))
    np.add.at(vi_merge_each, fm_inverse, terms_mutual_per_count)
    vi_merge_each -= terms_fm

    vi_split = np.sum(vi_split_each)
    vi_merge = np.sum(vi_merge_each)
    vi = vi_split + vi_merge

    i_splitters = np.argsort(vi_split_each)[::-1]
    i_mergers = np.argsort(vi_merge_each)[::-1]

    vi_split_sorted = vi_split_each[i_splitters]
    vi_merge_sorted = vi_merge_each[i_mergers]

    splitters = np.stack([vi_split_sorted, sm_unique[i_splitters]], axis=1)
    mergers = np.stack([vi_merge_sorted, fm_unique[i_mergers]], axis=1)
    return vi, vi_split, vi_merge, splitters, mergers


def evaluate_generalizability(
    dataset, w_theta, print_latex=False, method="", verbose=True
):
    """
    Evaluates how well a learned weight matrix w_theta generalizes to unseen data. It uses k-NN classification
    and Agglometarive Clustering for evaluation. The function returns the accuracy score of the classification results,
    the Mutual Information score (measures mutual dependence between predicted clusters and true labels), the Adjusted
    Rand score (quantifies similarity between predicted clusters and true labels based on pairwise decisions) and the
    Variation of Information (measures the difference between clusterings using entropy). It scales the scores such that they are between
    0 and 1 (1 is the best, 0 the worst).

    :param dataset: input dataset that holds the train and test datasets
    :type dataset: Dataset
    :param w_theta: trained weight matrix to check
    :type w_theta: array-like
    :param print_latex: whether to print the outcome table in latex format, defaults to False
    :type print_latex: bool, optional
    :param method: what was used to train w_theta, defaults to ""
    :type method: str, optional
    :return: the accuracy score of the knn classification, the NMI score, the ARI score and the VI score
    :rtype: tuple of floats
    """

    # if anndata object is given instead of only w_theta, extract the matrix
    if not isinstance(w_theta, np.ndarray):
        w_theta = w_theta.uns["W_ggml"]

    # extract and concatenate the train and test data
    all_datapoints = np.concatenate((dataset.datapoints, dataset.test_datapoints))
    all_distributions_labels = np.concatenate(
        (dataset.distributions_labels, dataset.test_distributions_labels)
    )
    all_distributions = np.concatenate(
        (dataset.distributions, dataset.test_distributions)
    )
    # compute mahalanobis and wasserstein distances on whole dataset
    distances = Computed_Distances(np.asarray(all_datapoints, dtype="f"), theta=w_theta)
    ot_distances = compute_OT(
        distributions=all_distributions, precomputed_distances=distances[:, :]
    )

    results = {}
    results[method] = {}

    # define train and test indices and matrices with distances between the train data and
    # distances between test and train data
    n_train = len(dataset.distributions)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(all_distributions))

    D_train_train = ot_distances[np.ix_(train_idx, train_idx)]
    D_test_train = ot_distances[np.ix_(test_idx, train_idx)]

    # use the train-train distance matrix for fitting the model and the train-test distance matrix
    # for predicting
    knn_clf = KNeighborsClassifier(
        n_neighbors=5, metric="precomputed", weights="uniform"
    )
    knn_clf.fit(D_train_train, dataset.distributions_labels)
    y_pred = knn_clf.predict(D_test_train)

    # compute the classification accuracy
    knn_acc = accuracy_score(dataset.test_distributions_labels, y_pred)

    # use the concatenation of all datapoints for the clustering
    clustering = AgglomerativeClustering(
        n_clusters=len(np.unique(all_distributions_labels)),
        metric="precomputed",
        linkage="average",
    )
    pred_cluster = clustering.fit_predict(ot_distances)

    # compute the MI, ARI and VI scores
    mi = normalized_mutual_info_score(all_distributions_labels, pred_cluster)
    ari = adjusted_rand_score(all_distributions_labels, pred_cluster)
    ari = (ari + 1) / 2
    vi, _, _ = VI(pred_cluster, all_distributions_labels)
    vi = (vi / np.log(2)) / np.log2(len(all_distributions_labels))
    vi = 1 - vi

    # store results
    results = {method: dict(knn_acc=knn_acc, mi=mi, ari=ari, vi=vi)}
    df = pd.DataFrame([results[method]])
    df["method"] = method

    if verbose:
        plot_table(df, print_latex=print_latex)

    return knn_acc, mi, ari, vi
