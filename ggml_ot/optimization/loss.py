"""Contains triplet loss functions used for training GGML."""

import torch

from ggml_ot.distances import _sinkhorn2_batched_torch
from ggml_ot.distances.ot import compute_OT
from ggml_ot.distances.bures_wasserstein import bures_covariance_distance
from ggml_ot.optimization.params import WassersteinBuresParams
from ggml_ot._utils._transformation import project_gaussians

from ggml_ot._utils._array import not_none
from ggml_ot.optimization._triplets import (
    get_unique_triplet_pairs,
    map_unique_pair_costs_to_triplets,
)
from ggml_ot._utils._validate import assert_finite_tensor


def batch_triplet_loss_emd2(
    loss,
    distr_points,
    distr_weights,
    distr_covs,
    triplets_idx,
    map_A,
    alpha,
    ot_params=None,
    **kwargs,
):
    """Triplet loss for a batch of triplets (EMD2 path with pair deduplication)."""
    unique_pairs, inverse_indices = get_unique_triplet_pairs(triplets_idx)

    D_unique = compute_OT(
        distr_points,
        distr_covs,
        distr_weights,
        ground_metric=map_A,
        diag_bures_approx=ot_params.diag_bures_approx if isinstance(ot_params, WassersteinBuresParams) else False,
        squared_ground_cost=ot_params.squared_ground_cost,
        identical_supports=ot_params.identical_supports,
        eps=ot_params.eps,
        pairs=unique_pairs,
    )

    D_ij, D_jk = map_unique_pair_costs_to_triplets(
        D_unique,
        inverse_indices,
        n_triplets=int(triplets_idx.shape[0]),
    )

    return loss + torch.nn.functional.relu(D_ij - D_jk + alpha).mean()


def batch_triplet_loss_sinkhorn(
    loss,
    distr_points,
    distr_weights,
    distr_covs,
    triplets_idx,
    map_A,
    alpha,
    ot_params=None,  #: OTParams | None  type hint with lazy loading to avoidcircular import
    **kwargs,
):
    # Compute ground cost under map_A for points/covariances.
    has_covariances = not_none(distr_covs)
    diag_bures_approx = isinstance(ot_params, WassersteinBuresParams) and ot_params.diag_bures_approx
    cov_eps = ot_params.eps

    if has_covariances:
        proj_distr_points, proj_distr_covs = project_gaussians(
            distr_points,
            map_A,
            covariances=distr_covs,
            covariance_eps=cov_eps,
        )
        assert_finite_tensor(proj_distr_covs.detach(), name="proj_distr_covs")
    else:
        proj_distr_points, _ = project_gaussians(distr_points, map_A)
    assert_finite_tensor(proj_distr_points.detach(), name="proj_distr_points")

    # inverse_indices maps concatenated (i,j),(j,k) entries to unique pair rows.
    unique_pairs, inverse_indices = get_unique_triplet_pairs(triplets_idx)

    # Gather data for unique pairs
    idx_u = unique_pairs[:, 0]
    idx_v = unique_pairs[:, 1]

    # Points: (N_unique, N, D)
    P_u = proj_distr_points[idx_u]
    P_v = proj_distr_points[idx_v]

    # Covariances: (N_unique, N, D, D) or None
    if has_covariances:
        Cov_u = proj_distr_covs[idx_u]
        Cov_v = proj_distr_covs[idx_v]

    # Compute Cost Matrices for Unique Pairs
    # M_uv: (N_unique, N, N)
    if ot_params.squared_ground_cost:
        diff = P_u.unsqueeze(2) - P_v.unsqueeze(1)
        M_uv = (diff**2).sum(-1)
    else:
        M_uv = torch.cdist(P_u, P_v, p=2)
    assert_finite_tensor(M_uv.detach(), name="M_uv (mean term)")

    if has_covariances:
        M_cov = bures_covariance_distance(
            Cov_u,
            Cov_v,
            diag_approx=diag_bures_approx,
            eps=None if diag_bures_approx else cov_eps,
            symmetric=False,
        )
        assert_finite_tensor(M_cov.detach(), name="M_cov (bures term)")

        # Bures-Wasserstein: (N_unique, N, N)
        M_uv = M_uv + M_cov
        assert_finite_tensor(M_uv.detach(), name="M_uv (full cost)")

    # Weights: (N_unique, N)
    W_u = distr_weights[idx_u]
    W_v = distr_weights[idx_v]

    # Sinkhorn-specific path for unique pairs.
    # D_unique: (N_unique,)
    D_unique = _sinkhorn2_batched_torch(
        W_u,
        W_v,
        M_uv,
        reg=ot_params.entropic_reg,
        numItermax=ot_params.sinkhorn_max_iter,
        stopThr=ot_params.sinkhorn_stop_thr,
    )
    assert_finite_tensor(D_unique.detach(), name="D_unique (sinkhorn costs)")

    D_ij, D_jk = map_unique_pair_costs_to_triplets(
        D_unique,
        inverse_indices,
        n_triplets=int(triplets_idx.shape[0]),
    )

    # Compute Triplet Loss
    return loss + torch.nn.functional.relu(D_ij - D_jk + alpha).mean()
