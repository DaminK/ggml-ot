"""Contains regularization loss functions for GGML."""

from __future__ import annotations

import torch

from ggml_ot import settings
from ggml_ot.optimization.params import canonicalize_reg_type
from ggml_ot._utils._array import as_torch as _as_torch
from ggml_ot._utils._covariance import symmetrise_and_jitter_covariances as _symmetrise_and_jitter
from ggml_ot._utils._weights import is_empty_weights as _is_empty_weights
from ggml_ot._utils._weights import align_weights_to_components as _align_weights_to_components


def regularizer_loss(map_A, loss_params):
    """Compute norm-based regularization.

    Supported `reg_type` values are:
    - `1`/`"1"`: elementwise L1 norm
    - `2`/`"2"`/`"fro"`: Frobenius (matrix L2) norm
    - `"nuc"`: nuclear norm
    """
    reg_type = canonicalize_reg_type(loss_params.reg_type)
    # reg_type=1 is elementwise L1, not the induced matrix-1 norm.
    if reg_type == 1:
        norm_val = torch.sum(torch.abs(map_A))
    else:
        norm_val = torch.linalg.norm(map_A, ord=reg_type)
    return loss_params.reg * norm_val


def regularizer_loss_mutual_information(w, mu, sigma, weights, loss_params):
    """Compute norm regularization plus optional mutual-information regularization."""
    norm_loss = regularizer_loss(w, loss_params)

    mi_reg = loss_params.mi_reg
    if isinstance(mi_reg, bool):
        if not mi_reg:
            return norm_loss
    elif float(mi_reg) <= 0:
        return norm_loss

    weighting = loss_params.mi_reg_weighting
    if weighting == "components":
        # Weighted by weights of Gaussian Components
        mi_weights = None if _is_empty_weights(weights) else weights
    elif weighting == "uniform":
        # Uniform weighting across components (equivalent to unweighted MI loss)
        mi_weights = None
    elif weighting == "projection":
        # Projection-norm weighting emphasizes components with mean projected away from center
        mu_t = _as_torch(mu, device=w.device)
        projected_means = torch.matmul(mu_t.float(), w.float().T)
        mi_weights = torch.linalg.norm(projected_means, dim=-1)
    elif weighting == "margin":
        raise NotImplementedError("`mi_reg_weighting='margin'` is not implemented yet.")
    else:
        raise ValueError(f"Unknown mi_reg_weighting option: {weighting}")

    mi_loss = mutual_information_loss(sigma, w, weights=mi_weights, eps=loss_params.eps, sqrt=loss_params.mi_sqrt)

    if isinstance(mi_reg, bool):
        # `True` means multiplicative coupling of norm and MI terms.
        return norm_loss * mi_loss

    mi_reg_t = torch.scalar_tensor(float(mi_reg), device=w.device, dtype=w.dtype)
    return norm_loss + mi_reg_t * mi_loss


def mutual_information(covs):
    """Compute per-component KL to corresponding diagonal Gaussian."""
    eps = 1e-8
    covs_t = covs.float()

    diag_entries = torch.diagonal(covs_t, dim1=-2, dim2=-1).clamp_min(eps)
    logdet_diag = torch.sum(torch.log(diag_entries), dim=-1)

    dim = covs_t.shape[-1]
    eye = torch.eye(dim, device=covs_t.device, dtype=covs_t.dtype)
    sign, logdet_full = torch.linalg.slogdet(covs_t + eps * eye)

    # Numerical guard: for valid covariances sign should be positive.
    invalid = sign <= 0
    if torch.any(invalid):
        logdet_full = torch.where(
            invalid,
            torch.log(torch.clamp(torch.linalg.det(covs_t + eps * eye), min=eps)),
            logdet_full,
        )

    kl = 0.5 * (logdet_diag - logdet_full)
    return torch.clamp(kl, min=0.0)


def mutual_information_loss(sigma, w, weights=None, eps: float = 1e-6, sqrt: bool = False):
    """Compute mutual information loss of projected Gaussians with torch/GPU support.

    If ``sqrt=True``, takes the square root of each per-component MI before
    weighting and summing: :math:`\\sum_k w_k \\sqrt{\\mathrm{MI}_k}`.
    This matches the non-constant factor in the diagonal approximation error
    bound (Theorem diag_mi_bounds).
    """
    device = w.device if isinstance(w, torch.Tensor) else settings.device

    sigma_t = _as_torch(sigma, device=device)
    w_t = _as_torch(w, device=device)

    if sigma_t.dim() == 2:
        if sigma_t.shape[0] != sigma_t.shape[1]:
            raise ValueError(
                f"mutual_information_loss expects square covariance matrices. Got sigma shape={tuple(sigma_t.shape)}."
            )
        sigma_flat = sigma_t.unsqueeze(0)
    elif sigma_t.dim() == 3:
        if sigma_t.shape[-2] != sigma_t.shape[-1]:
            raise ValueError(
                f"mutual_information_loss expects square covariance matrices. Got sigma shape={tuple(sigma_t.shape)}."
            )
        sigma_flat = sigma_t
    elif sigma_t.dim() == 4:
        if sigma_t.shape[-2] != sigma_t.shape[-1]:
            raise ValueError(
                f"mutual_information_loss expects square covariance matrices. Got sigma shape={tuple(sigma_t.shape)}."
            )
        # (B, C, D, D) -> flatten Gaussian components across batch for one KL vector.
        sigma_flat = sigma_t.reshape(-1, sigma_t.shape[-2], sigma_t.shape[-1])
    else:
        raise ValueError(f"Unsupported covariance tensor rank for MI loss: {sigma_t.dim()}")

    # Project covariances with the learned map: W * Sigma * W^T.
    w_f = w_t.float()
    sigma_f = sigma_flat.float()
    w_sigma = torch.matmul(w_f.unsqueeze(0), sigma_f)
    w_sigma = torch.matmul(w_sigma, w_f.T.unsqueeze(0))

    # Symmetrise and add diagonal jitter before the log-det computation.
    # W Σ Wᵀ can be near-singular when W has small singular values during early training,
    # causing NaN in the slogdet backward pass (which involves the matrix inverse).
    w_sigma = _symmetrise_and_jitter(w_sigma, eps=eps)

    kl = mutual_information(w_sigma)

    if sqrt:
        kl = torch.sqrt(kl + 1e-8)

    if weights is not None:
        weights_t = _as_torch(weights, device=device)
        # Align possibly batched/component-level weights to flattened covariance layout.
        weights_flat = _align_weights_to_components(weights_t, tuple(sigma_t.shape), target_n=kl.numel())
        denom = torch.sum(weights_flat)
        if torch.isclose(denom, torch.tensor(0.0, device=device, dtype=weights_flat.dtype)):
            raise ValueError("Weights must not sum to zero.")
        kl = kl * (weights_flat / denom)

    return torch.sum(kl)
