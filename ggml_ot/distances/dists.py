import numpy as np
import torch
import ot
import scipy.spatial as sp

from ggml_ot.distances.gaussian_emd import pairwise_gaussian_distance
from ggml_ot.distances.mahalanobis import pairwise_mahalanobis_distance

# Catch internally triggered future deprecation warning
import warnings

warnings.simplefilter("ignore", category=FutureWarning)


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
        if np.isnan(self.data[slice_]).any():
            ranges = [np.squeeze(np.arange(len(self.data))[slice_[i]]) for i in range(len(slice_))]
            # find the nan entries in the distance matrix
            entry_nan_index = ([], [])
            for entry in ranges[0]:
                check = np.isnan(self.data[entry, :])
                if check.ndim == 2 and np.isnan(self.data[entry, :][:, slice_[1]]).any():
                    entry_nan_index[0].append(entry)
                elif check.ndim == 1 and np.isnan(self.data[entry, :][slice_[1]]).any():
                    entry_nan_index[0].append(entry)
            for entry in ranges[1]:
                if np.isnan(self.data[slice_[0], entry]).any():
                    entry_nan_index[1].append(entry)

            # check for elements with nan entries and compute mahalanobis distances
            dist = pairwise_mahalanobis_distance(
                self.points[entry_nan_index[0], :],
                self.points[entry_nan_index[1], :],
                w=self.theta,
            )
            self.data[np.ix_(entry_nan_index[0], entry_nan_index[1])] = dist

            return self.data[slice_]

        else:
            return self.data[slice_]


def compute_OT(
    supports,
    covariances=None,
    weights=None,  # TODO comment
    identical_supports=False,  # TODO comment
    hellinger_approx=False,  # TODO comment
    precomputed_distances=None,
    ground_metric=None,
    n_threads=32,
    **kwargs,
):
    """
    Compute the Optimal Transport between distributions using precomputed distances, the mahalanobis
    distance or a different ground metric.

    :param supports: supports to compute the OT on of shape (num_distributions, num_points, num_features)
    :type supports: array-like
    :param precomputed_distances: precomputed distances to use as ground metric, defaults to None
    :type precomputed_distances: array-like, optional
    :param ground_metric: weight matrix for the mahalanobis distance or string refering to other ground metrics (see scipy.spatial.distance.cdist), defaults to None (euclidean).
       Currently only support matrix or None (Euclidean) for GMMs.
    :type ground_metric: array-like, str, optional
    :param n_threads: number of threads to use for the computation of the OT, defaults to 32
    :type n_threads: int, optional
    :return: OT matrix
    :rtype: numpy.ndarray
    """

    # Setup ground metric for identical supports
    if identical_supports:
        assert weights is not None, (
            "identical_supports == true and weights is None: OT distance will be zero between all distributions for identical supports and weights"
        )

        if precomputed_distances is None:
            if covariances is None:
                # TODO support ground_metric str from scipy.dist in pairwise_mahalanobis
                M = pairwise_mahalanobis_distance(supports, supports, ground_metric, as_numpy=True)
            else:
                if isinstance(ground_metric, str):
                    assert ground_metric == "euclidean", (
                        "ground_metric for GMMs only supports Mahalanobis (ground_metric: array_like) or Euclidean (ground_metric: 'euclidean' | None)"
                    )
                M = pairwise_gaussian_distance(
                    supports,
                    covariances,
                    ground_metric,
                    hellinger_approx=hellinger_approx,
                ).numpy()
        else:
            M = precomputed_distances

        D = np.zeros((len(weights), len(weights)))
    else:
        D = np.zeros((len(supports), len(supports)))

    # Iterate over distribution pairs
    for i, distribution_i in enumerate(supports if not identical_supports else weights):
        for j, distribution_j in enumerate(supports if not identical_supports else weights):
            if i < j:
                # Setup ground metric for different supports
                if not identical_supports:
                    # Get corresponding precomputed distances
                    if precomputed_distances is not None:
                        start_i = int(np.sum([len(dist) for dist in supports[:i]]))
                        start_j = int(np.sum([len(dist) for dist in supports[:j]]))
                        if precomputed_distances.ndim == 1:
                            precomputed_distances = sp.distance.squareform(precomputed_distances)
                        M = precomputed_distances[
                            start_i : start_i + len(distribution_i),
                            start_j : start_j + len(distribution_j),
                        ]
                    # TODO move everything below into pairwise_mahalanobis_distance
                    # Mahalanobis distance
                    elif isinstance(ground_metric, np.ndarray) or isinstance(ground_metric, torch.Tensor):
                        M = pairwise_mahalanobis_distance(distribution_i, distribution_j, ground_metric, as_numpy=True)
                    # Other metric (see scipy.spatial.distance)
                    elif isinstance(ground_metric, str) or ground_metric is None:
                        M = sp.distance.cdist(distribution_i, distribution_j, metric=ground_metric)
                    else:
                        raise TypeError(
                            f"ground_metric has unknown type {type(ground_metric)}, only np.ndarray, torch.Tensor and str are supported"
                        )

                # Compute OT
                if weights is None:
                    D[i, j] = _emd2([], [], M, numThreads=n_threads)
                else:
                    D[i, j] = _emd2(weights[i], weights[j], M, numThreads=n_threads)
            else:
                D[i, j] = D[j, i]

    # Very small OT distances can become negative in emd2 due to numerical errors
    D[D < 0] = 0

    return D


def _emd2(*args, return_pi=False, **kwargs):
    # Wrapper for emd2 to decide if we use exact solver on CPU or approximative solver on GPU
    if False:  # torch.cuda.is_available():
        # GPU with CUDA
        return ot.bregman.sinkhorn_stabilized(
            *args, reg=1.0, **kwargs
        )  # TODO test this, it seems to be really unstable
    else:
        # CPU
        if return_pi:
            W, log = ot.emd2(*args, log=True, return_matrix=True, **kwargs)
            Pi = log["G"]
            return W, Pi
        else:
            return ot.emd2(*args, **kwargs)
