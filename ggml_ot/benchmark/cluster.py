import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# Catch internally triggered future deprecation warning
import warnings

warnings.simplefilter("ignore", category=FutureWarning)


def hierachical_clustering(ot_distances, labels, n_cluster=None):
    if n_cluster is None:
        n_cluster = len(np.unique(labels))

    clustering = AgglomerativeClustering(
        n_clusters=n_cluster,
        metric="precomputed",
        linkage="average",
    )
    pred_cluster = clustering.fit_predict(ot_distances)

    mi = normalized_mutual_info_score(labels, pred_cluster)
    ari = adjusted_rand_score(labels, pred_cluster)
    vi = variation_of_info(labels, pred_cluster)

    return mi, ari, vi


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


# Package import is broken,
"""
import pyvoi
def variation_of_info(*args,torch = False,**kwargs):
    return pyvoi.VI(*args,torch,**kwargs)[0]
"""


# Thus, we directly use the relevant code under MIT license https://pypi.org/project/python-voi/
def variation_of_info(
    labels1: np.typing.NDArray[np.int32],
    labels2: np.typing.NDArray[np.int32],
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
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    if labels1.ndim > 1 or labels2.ndim > 1:
        warnings.warn(
            f"Inputs of shape {labels1.shape}, {labels2.shape} are not one-dimensional -- inputs will be flattened."
        )
        labels1 = labels1.flatten()
        labels2 = labels2.flatten()

    vi_scores = VI_np(labels1, labels2, return_split_merge=return_split_merge)
    return vi_scores[0]


def VI_np(labels1, labels2, return_split_merge=False):
    assert len(labels2) == len(labels1)
    size = len(labels2)

    mutual_labels = (labels1.astype(np.uint64) << 32) + labels2.astype(np.uint64)

    sm_unique, sm_inverse, sm_counts = np.unique(labels2, return_inverse=True, return_counts=True)
    fm_unique, fm_inverse, fm_counts = np.unique(labels1, return_inverse=True, return_counts=True)
    _, mutual_inverse, mutual_counts = np.unique(mutual_labels, return_inverse=True, return_counts=True)

    terms_mutual = -np.log(mutual_counts / size) * mutual_counts / size
    terms_mutual_per_count = terms_mutual[mutual_inverse] / mutual_counts[mutual_inverse]
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
