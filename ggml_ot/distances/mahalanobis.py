import torch
import numpy as np


def pairwise_mahalanobis_distance(X_i, X_j, w, as_numpy=False):
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
    if isinstance(X_i, np.ndarray):
        X_i = torch.from_numpy(X_i).to(dtype=torch.float32)
    if isinstance(X_j, np.ndarray):
        X_j = torch.from_numpy(X_j).to(dtype=torch.float32)
    if isinstance(w, np.ndarray):
        w = torch.from_numpy(w).to(dtype=torch.float32)

    if w == "euclidean":
        w = torch.eye(X_i.shape[-1])

    # Transform poins of X_i,X_j according to W
    if w.dim() == 1:
        # assume cov=0, scale dims by diagonal
        proj_X_i = X_i * w[None, :]
        proj_X_j = X_j * w[None, :]

    else:
        w = torch.transpose(w, 0, 1)
        proj_X_i = torch.matmul(X_i, w)
        proj_X_j = torch.matmul(X_j, w)

    distances = torch.linalg.norm(proj_X_i[:, torch.newaxis, :] - proj_X_j[torch.newaxis, :, :], dim=-1)
    if as_numpy:
        distances = distances.detach().numpy()

    return distances


# TODO Remove (and only support calculation with torch) or merge into combined function (to leverage multithreading from scipy.spatial.distances)
'''
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
'''
