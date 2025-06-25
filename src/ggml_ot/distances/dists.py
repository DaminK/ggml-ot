import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy
import ot
import scipy.spatial as sp


class Computed_Distances:
    """Computes and caches mahalanobis distance on-demand.

    :param points: points the distances are computed from
    :type points: array-like
    :param theta: weight vector for the mahalanobis distance
    :type theta: array-like
    :param n_threads: number of threads to use for the computation of the mahalanobis distance
    :type n_threads: int

    :ivar data: holds the computed distances
    :vartype data: numpy.ndarray
    :ivar ndim: dimension of the data matrix
    :vartype ndim: int
    :ivar shape: shape of the data matrix
    :vartype shape: tuple
    """

    def __init__(self, points, theta, n_threads=60):
        self.points = points
        self.theta = theta
        self.n_threads = n_threads

        self.data = np.full((len(points), len(points)), np.nan)

        self.ndim = self.data.ndim
        self.shape = self.data.shape

    def __getitem__(self, slice_):
        # print(f"Triggered __getitem__ with slice: {slice_}")
        if np.isnan(self.data[slice_]).any():
            ranges = [
                np.squeeze(np.arange(len(self.data))[slice_[i]])
                for i in range(len(slice_))
            ]
            # ranges = []
            # for i in range(len(slice_)):
            #     s = slice_[i]
            #     if isinstance(s, int):
            #         ranges.append(np.array([s]))
            #     else:
            #         ranges.append(np.arange(len(self.data))[s])
            # find the nan entries in the distance matrix
            entry_nan_index = ([], [])
            for entry in ranges[0]:
                check = np.isnan(self.data[entry, :])
                if (
                    check.ndim == 2
                    and np.isnan(self.data[entry, :][:, slice_[1]]).any()
                ):
                    entry_nan_index[0].append(entry)
                elif check.ndim == 1 and np.isnan(self.data[entry, :][slice_[1]]).any():
                    entry_nan_index[0].append(entry)
            for entry in ranges[1]:
                if np.isnan(self.data[slice_[0], entry]).any():
                    entry_nan_index[1].append(entry)

            # check for elements with nan entries and compute mahalanobis distances
            dist = pairwise_mahalanobis_distance_npy(
                self.points[entry_nan_index[0], :],
                self.points[entry_nan_index[1], :],
                w=self.theta,
                numThreads=self.n_threads,
            )
            self.data[np.ix_(entry_nan_index[0], entry_nan_index[1])] = dist

            return self.data[slice_]

        else:
            return self.data[slice_]


def compute_OT(
    distributions,
    precomputed_distances=None,
    ground_metric=None,
    w=None,
    cluster_centers=None,
    n_threads=32,
):
    """Compute the Optimal Transport between distributions using precomputed distances, the mahalanobis
    distance or a different ground metric.

    :param distributions: distributions to compute the OT on of shape (num_distributions, num_points, num_features)
    :type distributions: array-like
    :param precomputed_distances: precomputed distances to use as ground metric, defaults to None
    :type precomputed_distances: array-like, optional
    :param ground_metric: ground metric to use, defaults to None
    :type ground_metric: "euclidean", "cosine", "cityblock", optional
    :param w: weight matrix for the mahalanobis distance, defaults to None
    :type w: array-like, optional
    :param n_threads: number of threads to use for the computation of the OT, defaults to 32
    :type n_threads: int, optional
    :return: OT matrix
    :rtype: numpy.ndarray
    """
    if cluster_centers is None:
        cluster_centers = []
    # initialize matrix that stores the OT distance for each distribution pair
    D = np.zeros((len(distributions), len(distributions)))
    for i, distribution_i in enumerate(distributions):
        for j, distribution_j in enumerate(distributions):
            if i < j:
                # if precomputed distances are provided, use them
                if precomputed_distances is not None:
                    start_i = int(np.sum([len(dist) for dist in distributions[:i]]))
                    start_j = int(np.sum([len(dist) for dist in distributions[:j]]))
                    if precomputed_distances.ndim == 1:
                        precomputed_distances = scipy.spatial.distance.squareform(
                            precomputed_distances
                        )
                    M = precomputed_distances[
                        start_i : start_i + len(distribution_i),
                        start_j : start_j + len(distribution_j),
                    ]
                # if w is given, compute the mahalanobis distance
                elif w is not None:
                    if len(cluster_centers) == 0:
                        M = pairwise_mahalanobis_distance_npy(
                            distribution_i, distribution_j, w
                        )
                    else:
                        M = pairwise_mahalanobis_distance_npy(
                            cluster_centers, cluster_centers, w
                        )
                # if a ground metric is given, compute the distance using that metric
                elif ground_metric == "euclidean":
                    M = sp.distance.cdist(
                        distribution_i, distribution_j, metric="euclidean"
                    )
                    print("finished computing")
                elif ground_metric == "cosine":
                    M = sp.distance.cdist(
                        distribution_i, distribution_j, metric="cosine"
                    )
                # the Manhattan distance
                elif ground_metric == "cityblock":
                    M = sp.distance.cdist(
                        distribution_i, distribution_j, metric="cityblock"
                    )
                # compute the Earth Mover's Distance (OT)
                if len(cluster_centers) == 0:
                    D[i, j] = ot.emd2([], [], M, numThreads=n_threads)
                else:
                    D[i, j] = ot.emd2(
                        distribution_i, distribution_j, M, numThreads=n_threads
                    )
            else:
                D[i, j] = D[j, i]

    return D


def pairwise_mahalanobis_distance(X_i, X_j, w):
    """Compute the Mahalanobis distance between two distributions using w (with torch).

    :param X_i: distribution of shape (num_points n, num_features)
    :type X_i: torch.Tensor
    :param X_j: distribution of shape (num_points m, num_features)
    :type X_j: torch.Tensor
    :param w: weight tensor defining the mahalanobis distance of shape (rank k, num_features)
    :type w: torch.Tensor
    :return: Mahalanobis distance between X_i and X_j of shape (num_points n, num_points m)
    :rtype: torch.Tensor
    """
    # Transform poins of X_i,X_j according to W
    if w.dim() == 1:
        # assume cov=0, scale dims by diagonal
        proj_X_i = X_i * w[None, :]
        proj_X_j = X_j * w[None, :]

    else:
        w = torch.transpose(w, 0, 1)
        proj_X_i = torch.matmul(X_i, w)
        proj_X_j = torch.matmul(X_j, w)

    return torch.linalg.norm(
        proj_X_i[:, torch.newaxis, :] - proj_X_j[torch.newaxis, :, :], dim=-1
    )


def pairwise_mahalanobis_distance_npy(X_i, X_j=None, w=None, numThreads=32):
    """Compute the Mahalanobis distance between two distributions X_i and X_j using w which can be a weight tensor
    or a ground metric. If only X_i is given, the distance is computed between all pairs of X_i.

    :param X_i: distribution of shape (num_points n, num_features)
    :type X_i: array-like
    :param X_j: distribution of shape (num_points m, num_features), defaults to None
    :type X_j: array-like, optional
    :param w: weight tensor defining the mahalanobis distance of shape (rank k, num_features) or a string defining the metric to use, defaults to None
    :type w: array-like or str, optional
    :return: Mahalanobis distance (or distance of given metric) between X_i and X_j or all pairs of X_i of shape (num_points n, num_points m)
    :rtype: array-like
    """
    # if X_j is not provided, compute the distance between all pairs of X_i
    if X_j is None:
        # if w is a string, compute the distance using that metric
        if w is None or isinstance(w, str):
            return pairwise_distances(X_i, metric=w, n_jobs=numThreads)
        # else, compute the mahalanobis distance
        else:
            if w.ndim == 2 and w.shape[0] == w.shape[1]:
                return pairwise_distances(
                    X_i, metric="mahalanobis", n_jobs=numThreads, VI=w
                )
            else:
                X_j = X_i
    # Transform points of X_i,X_j according to W
    if w is None or isinstance(w, str):
        return scipy.spatial.distance.cdist(X_i, X_j, metric=w)
    # Assume w is cov matrix of mahalanobis distance
    elif w.ndim == 1:
        # assume cov=0, scale dims by diagonal
        w = np.diag(w)
        proj_X_i = np.matmul(X_i, w)
        proj_X_j = np.matmul(X_j, w)

        # proj_X_i = X_i * w[None,:]
        # proj_X_j = X_j * w[None,:]
    else:
        w = np.transpose(w)
        proj_X_i = np.matmul(X_i, w)
        proj_X_j = np.matmul(X_j, w)

    return np.linalg.norm(
        proj_X_i[:, np.newaxis, :] - proj_X_j[np.newaxis, :, :], axis=-1
    )
