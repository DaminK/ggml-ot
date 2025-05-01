from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

import numpy as np
import seaborn as sns
import numpy.typing as npt
import matplotlib.pyplot as plt
from IPython.display import display
import warnings


def ShuffleSplit(
    labels,
    n_splits=10,
    train_size=0.4,
    test_size=0.4,
    validation_size=0.2,
    distribution_labels=None,
):
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
            print('shapes')
            print(np.zeros(len(labels)).shape)
            print(labels.shape)
            print(distribution_labels.shape)
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
    predicted_labels, true_labels, test_indices, scores = [], [], [], []

    train_test_inds = ShuffleSplit(
        labels,
        n_splits,
        train_size,
        test_size,
        validation_size=0,
        distribution_labels=distribution_labels,
    )
    # , np.arange(len(labels)))

    for i, (train_index, test_index) in enumerate(train_test_inds):
        print(train_index)
        print(test_index)
        train_dists = dists[np.ix_(train_index, train_index)]
        test_to_train_dists = dists[np.ix_(test_index, train_index)]

        neigh = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric='precomputed', weights=weights
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
    # wrapper function as fill diagonal is only available as inplace operation. It is needed to catch cases where due to numerical errors the distance of a graph to itself may be very close to zero, but not zero which is required by sklearn silhoute score method
    zero_dia_dists = np.copy(dists)
    np.fill_diagonal(zero_dia_dists, 0)
    return silhouette_score(zero_dia_dists, labels, metric='precomputed')


def compute_dists(Graphs, Graphs2=None, method='TiedOT'):
    dist, plan = methods[method](Graphs, Graphs2)
    dist[dist < 0] = 0
    return dist


def get_dist_precomputed(precomputed_dists, ind1, ind2):
    return precomputed_dists[ind1, :][:, ind2]


def plot_1split(predicted, true, title=None, ax=None):
    annot_labels_ind = np.unique(true, return_index=True)[1]
    annot_labels = true[annot_labels_ind]
    # ind
    cf_matrix = confusion_matrix(true, predicted, labels=annot_labels)
    if ax is None:
        plt.figure()
    ax = sns.heatmap(
        cf_matrix,
        annot=True,  # fmt='.0',
        cmap='Blues',
        xticklabels=annot_labels,
        yticklabels=annot_labels,
        ax=ax,
        fmt='g',
    )
    ax.set(xlabel='Predicted Label', ylabel='True Label')
    ax.set_title(title)


def plot_table(df, tranpose=False):
    format_df = df
    format_df.set_index('method', inplace=True)
    if tranpose:
        format_df = format_df.transpose()
    display(format_df)
    print(
        format_df.to_latex(
            index=True,
            # formatters={"name": str.upper},
            float_format='{:.2f}'.format,
        )
    )


def VI(
    labels1: npt.NDArray[np.int32],
    labels2: npt.NDArray[np.int32],
    torch: bool = True,
    device: str = 'cpu',
    return_split_merge: bool = False,
):
    """
    Calculates the Variation of Information between two clusterings.

    Arguments:
    labels1: flat int32 array of labels for the first clustering
    labels2: flat int32 array of labels for the second clustering
    torch: whether to use torch, default:True
    device: device to use for torch, default:"cpu"
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

    if torch:
        return VI_torch(
            labels1, labels2, device=device, return_split_merge=return_split_merge
        )
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
