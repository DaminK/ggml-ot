import torch
import numpy as np

from ggml_ot import settings
import scipy.spatial as sp


def pairwise_mahalanobis_distance(X_i, X_j, ground_metric="euclidean", as_numpy=False, squared=False):
    """Compute the Mahalanobis distance between two distributions using w (with torch).

    :param X_i: distribution of shape (num_points n, num_features)
    :type X_i: torch.Tensor
    :param X_j: distribution of shape (num_points m, num_features)
    :type X_j: torch.Tensor
    :param ground_metric: weight tensor defining the mahalanobis distance of shape (rank k, num_features) or a string defining the metric to use, defaults to "euclidean"
    :type ground_metric: torch.Tensor | np.ndarray | str, optional
    :return: Mahalanobis distance between X_i and X_j of shape (num_points n, num_points m)
    :rtype: torch.Tensor
    """
    if isinstance(ground_metric, str) or ground_metric is None:
        # Fixed ground metrics, cdist from scipy only supports numpy arrays
        X_i_np = X_i.detach().cpu().numpy() if isinstance(X_i, torch.Tensor) else X_i
        X_j_np = X_j.detach().cpu().numpy() if isinstance(X_j, torch.Tensor) else X_j

        dists = sp.distance.cdist(X_i_np, X_j_np, metric=ground_metric)
        if squared:
            dists = dists**2  # numpy: no autograd, squaring 0 is safe

        if isinstance(X_i, torch.Tensor):
            # Convert back to tensor if input was tensor
            return torch.from_numpy(dists).to(device=X_i.device, dtype=torch.float32)
        return dists

    elif isinstance(ground_metric, np.ndarray) or isinstance(ground_metric, torch.Tensor):
        w = ground_metric
        # Support numpy inputs
        device = settings.device
        if isinstance(X_i, np.ndarray):
            X_i = torch.from_numpy(X_i).to(dtype=torch.float32, device=device)
        if isinstance(X_j, np.ndarray):
            X_j = torch.from_numpy(X_j).to(dtype=torch.float32, device=device)
        if isinstance(w, np.ndarray):
            w = torch.from_numpy(w).to(dtype=torch.float32, device=device)

        # Ensure w is on same device as X_i (e.g. if dataset is on CPU but w on GPU)
        if isinstance(w, torch.Tensor) and w.device != X_i.device:
            w = w.to(X_i.device)

        if isinstance(w, str) and w == "euclidean":
            w = torch.eye(X_i.shape[-1], device=X_i.device)

        # Transform poins of X_i,X_j according to W
        if w.dim() == 1:
            # assume cov=0, scale dims by diagonal
            proj_X_i = X_i * w[None, :]
            proj_X_j = X_j * w[None, :]

        else:
            w = torch.transpose(w, 0, 1)
            proj_X_i = torch.matmul(X_i, w)
            proj_X_j = torch.matmul(X_j, w)

        diff = proj_X_i[:, None, :] - proj_X_j[None, :, :]
        if squared:
            # Compute ||x-y||^2 directly to avoid NaN gradient from sqrt at d=0.
            distances = (diff**2).sum(-1)
        else:
            distances = torch.linalg.norm(diff, dim=-1)
        if as_numpy:
            return distances.detach().cpu().numpy()

        return distances
    elif callable(ground_metric):
        # Custom ground metric functions, e.g. supports projection function from WDA
        proj_X_i = ground_metric(X_i)
        proj_X_j = ground_metric(X_j)

        dists = np.linalg.norm(proj_X_i[:, None, :] - proj_X_j[None, :, :], axis=-1)
        if squared:
            dists = dists**2
        return dists

    else:
        raise TypeError(
            f"ground_metric has unknown type {type(ground_metric)}, only np.ndarray, torch.Tensor and str are supported"
        )
