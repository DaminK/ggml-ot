"""Linear map projection helpers."""

from __future__ import annotations

import torch


def project_means(means: torch.Tensor, linear_map: torch.Tensor) -> torch.Tensor:
    """Project means with a linear map ``W``: ``mu -> mu W^T``."""
    if linear_map.ndim != 2:
        raise ValueError(f"linear_map must be 2D with shape (k, d). Got {tuple(linear_map.shape)}.")
    if means.shape[-1] != linear_map.shape[-1]:
        raise ValueError(
            "Trailing means dimension must match linear_map feature dimension: "
            f"means.shape[-1]={means.shape[-1]} vs linear_map.shape[-1]={linear_map.shape[-1]}."
        )
    w = linear_map.float()
    m = means.float()
    lead_shape = m.shape[:-1]
    d = int(m.shape[-1])
    k = int(w.shape[0])
    # Flatten leading dims and use dense matmul (faster than einsum for this pattern).
    projected = torch.matmul(m.reshape(-1, d), w.transpose(0, 1))
    return projected.reshape(*lead_shape, k)


def project_covariances(covariances: torch.Tensor, linear_map: torch.Tensor) -> torch.Tensor:
    """Project covariances with a linear map ``W``: ``Sigma -> W Sigma W^T``."""
    if linear_map.ndim != 2:
        raise ValueError(f"linear_map must be 2D with shape (k, d). Got {tuple(linear_map.shape)}.")
    if covariances.ndim < 2:
        raise ValueError(
            f"covariances must have ndim>=2 with trailing square dims (..., d, d). Got {tuple(covariances.shape)}."
        )
    if covariances.shape[-2] != covariances.shape[-1]:
        raise ValueError(f"covariances trailing dims must be square (..., d, d). Got {tuple(covariances.shape)}.")
    if covariances.shape[-1] != linear_map.shape[-1]:
        raise ValueError(
            "Covariance feature dimension must match linear_map feature dimension: "
            f"covariances.shape[-1]={covariances.shape[-1]} vs linear_map.shape[-1]={linear_map.shape[-1]}."
        )
    w = linear_map.float()
    cov = covariances.float()
    batch_shape = cov.shape[:-2]
    d = int(cov.shape[-1])
    k = int(w.shape[0])
    cov_flat = cov.reshape(-1, d, d)
    w_batched = w.unsqueeze(0).expand(cov_flat.shape[0], -1, -1)
    # Compute W @ Sigma @ W^T via batched matmul after flattening leading dims.
    projected_flat = torch.matmul(torch.matmul(w_batched, cov_flat), w_batched.transpose(-1, -2))
    return projected_flat.reshape(*batch_shape, k, k)


def project_gaussians(
    means: torch.Tensor,
    linear_map: torch.Tensor,
    *,
    covariances: torch.Tensor | None = None,
    covariance_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Project Gaussian moments under a linear map with optional covariance regularization."""
    projected_means = project_means(means, linear_map)
    if covariances is None:
        return projected_means, None

    projected_covariances = project_covariances(covariances, linear_map)
    if covariance_eps is not None:
        from ggml_ot._utils._covariance import symmetrise_and_jitter_covariances

        projected_covariances = symmetrise_and_jitter_covariances(projected_covariances, eps=float(covariance_eps))
    return projected_means, projected_covariances


__all__ = [
    "project_gaussians",
    "project_covariances",
    "project_means",
]
