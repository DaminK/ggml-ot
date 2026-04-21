"""Pairwise/masking helpers to avoid redundant/unneccessary distance computations."""

from __future__ import annotations

import itertools

import numpy as np
import torch

from ggml_ot import settings
from ggml_ot._utils._array import slice_matrix
from ggml_ot._utils._weights import get_nonzero_weight_mask, has_zero_weight_entries
from ggml_ot.distances.bures_wasserstein import cross_gaussian_distance
from ggml_ot.distances.mahalanobis import pairwise_mahalanobis_distance


def initialize_pair_iteration(*, pairs, n_distributions: int, is_numpy: bool):
    """Create pair iterator and output container for full-matrix or pair-list mode."""
    if pairs is not None:
        pair_iter = pairs
        n_pairs = len(pair_iter)
        if is_numpy:
            return pair_iter, np.zeros(n_pairs)
        return pair_iter, torch.zeros(n_pairs, device=settings.device)

    pair_iter = itertools.combinations(range(n_distributions), 2)
    if is_numpy:
        return pair_iter, np.zeros((n_distributions, n_distributions))
    return pair_iter, torch.zeros((n_distributions, n_distributions), device=settings.device)


def prepare_pair_weight_masks(*, weights, has_weights: bool, i: int, j: int):
    """Return per-pair weights plus non-zero masks when padded components exist."""
    w_i = None if not has_weights else weights[i]
    w_j = None if not has_weights else weights[j]
    use_masking = bool(
        (w_i is not None and has_zero_weight_entries(w_i)) or (w_j is not None and has_zero_weight_entries(w_j))
    )
    mask_i = get_nonzero_weight_mask(w_i) if use_masking else None
    mask_j = get_nonzero_weight_mask(w_j) if use_masking else None
    return w_i, w_j, use_masking, mask_i, mask_j


def build_pair_cost_matrix(
    *,
    supports,
    covariances,
    ground_metric,
    diag_bures_approx: bool,
    eps: float,
    squared_ground_cost: bool = False,
    is_numpy: bool,
    identical_supports: bool,
    has_cov: bool,
    precomputed_full,
    precomputed_identical,
    support_offsets,
    i: int,
    j: int,
    use_masking: bool,
    mask_i,
    mask_j,
):
    """Assemble one pairwise ground-cost matrix `M` with optional masking."""
    if not identical_supports:
        # Non-identical supports: use sliced precompute when available.
        if precomputed_full is not None:
            matrix = precomputed_full[
                support_offsets[i] : support_offsets[i + 1],
                support_offsets[j] : support_offsets[j + 1],
            ]
            if use_masking:
                matrix = slice_matrix(matrix, mask_i, mask_j)
            return matrix

        if not has_cov:
            support_i = supports[i][mask_i] if use_masking else supports[i]
            support_j = supports[j][mask_j] if use_masking else supports[j]
            return pairwise_mahalanobis_distance(
                support_i,
                support_j,
                ground_metric,
                as_numpy=is_numpy,
                squared=squared_ground_cost,
            )

        support_i = supports[i][mask_i] if use_masking else supports[i]
        support_j = supports[j][mask_j] if use_masking else supports[j]
        cov_i = covariances[i][mask_i] if use_masking else covariances[i]
        cov_j = covariances[j][mask_j] if use_masking else covariances[j]
        return cross_gaussian_distance(
            support_i,
            cov_i,
            support_j,
            cov_j,
            ground_metric,
            diag_bures_approx=diag_bures_approx,
            as_numpy=is_numpy,
            eps=eps,
            squared_ground_cost=squared_ground_cost,
        )

    # Identical supports: either shared precomputed matrix or on-the-fly gaussian cost.
    if precomputed_full is not None:
        matrix = precomputed_full
        if use_masking:
            matrix = slice_matrix(matrix, mask_i, mask_j)
        return matrix

    if precomputed_identical is not None:
        return precomputed_identical

    support_i = supports[mask_i] if use_masking else supports
    support_j = supports[mask_j] if use_masking else supports
    cov_i = covariances[mask_i] if use_masking else covariances
    cov_j = covariances[mask_j] if use_masking else covariances
    return cross_gaussian_distance(
        support_i,
        cov_i,
        support_j,
        cov_j,
        ground_metric,
        diag_bures_approx=diag_bures_approx,
        as_numpy=is_numpy,
        eps=eps,
        squared_ground_cost=squared_ground_cost,
    )


def write_pair_distance(output, *, pairs, pair_idx: int, i: int, j: int, dist):
    """Write one solved OT value to either vector (pair list) or symmetric matrix."""
    if pairs is not None:
        output[pair_idx] = dist
        return
    output[i, j] = dist
    output[j, i] = dist
