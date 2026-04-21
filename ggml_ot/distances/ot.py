import numpy as np

from ggml_ot import settings
from ggml_ot._utils._array import clamp_nonnegative, is_numpy_backend, squareform
from ggml_ot._utils._covariance import has_covariances, validate_square_covariances
from ggml_ot._utils._validate import validate_gaussian_ground_metric
from ggml_ot._utils._weights import (
    has_zero_weight_entries,
    is_empty_weights,
    normalize_weights_container,
)
from ggml_ot.distances._cached_distances import Computed_Distances
from ggml_ot.distances._masked_pairwise import (
    build_pair_cost_matrix,
    initialize_pair_iteration,
    prepare_pair_weight_masks,
    write_pair_distance,
)
from ggml_ot.distances.bures_wasserstein import pairwise_gaussian_distance
from ggml_ot.distances.mahalanobis import pairwise_mahalanobis_distance
from ggml_ot.distances._solvers import _emd2

__all__ = ["Computed_Distances", "compute_OT"]


def compute_OT(
    supports,
    covariances=None,
    weights=None,
    identical_supports=False,
    diag_bures_approx=False,
    precomputed_distances=None,
    ground_metric=None,
    entropic_reg: float = 0.0,
    eps: float = 1e-4,
    squared_ground_cost: bool = False,
    pairs=None,
    **kwargs,
):
    """
    Compute optimal transport between distributions.

    :param supports: supports to compute the OT on of shape (num_distributions, num_points, num_features)
    :type supports: array-like
    :param diag_bures_approx: whether to approximate the Bures covariance term diagonally for Gaussian distances
    :type diag_bures_approx: bool
    :param precomputed_distances: precomputed distances to use as ground metric, defaults to None
    :type precomputed_distances: array-like, optional
    :param ground_metric: weight matrix for the mahalanobis distance or string refering to other ground metrics (see scipy.spatial.distance.cdist), defaults to None (euclidean).
       Currently only support matrix or None (Euclidean) for GMMs.
    :type ground_metric: array-like, str, optional
    :param entropic_reg: If ``> 0``, use Sinkhorn-regularized OT with this entropic
       regularization strength, expressed as a fraction of the mean ground-cost
       value. If ``0`` (default), compute exact OT via EMD2.
    :type entropic_reg: float, optional
    :param pairs: If given, only compute OT for these ``(i, j)`` index pairs and return
       a 1-D array/tensor of length *P* instead of the full *N×N* matrix, defaults to None.
    :type pairs: array-like of shape (P, 2), optional
    :return: OT matrix of shape (N, N), or 1-D array of length P when *pairs* is given.
    :rtype: numpy.ndarray or torch.Tensor
    """
    if entropic_reg < 0:
        raise ValueError("`entropic_reg` must be >= 0.")

    sinkhorn_max_iter = kwargs.pop("sinkhorn_max_iter", 100)
    sinkhorn_stop_thr = kwargs.pop("sinkhorn_stop_thr", None)

    # Check if data is numpy array or torch tensor
    is_numpy = is_numpy_backend(supports)
    has_cov = has_covariances(covariances)
    if has_cov:
        validate_square_covariances(
            supports=supports,
            covariances=covariances,
            identical_supports=bool(identical_supports),
        )
    has_weights = not is_empty_weights(weights)
    if has_weights:
        # Normalize once here to avoid repeated per-pair normalization overhead.
        weights = normalize_weights_container(weights)

    precomputed_full = precomputed_distances
    if precomputed_full is not None and getattr(precomputed_full, "ndim", None) == 1:
        precomputed_full = squareform(precomputed_full)
    precomputed_identical = None

    # Setup ground metric for identical supports
    if identical_supports:
        if weights is None:
            raise ValueError(
                "identical_supports == true and weights is None: OT distance will be zero between all distributions for identical supports and weights"
            )

        # For Gaussian supports with potentially padded components, compute
        # per-pair distances after nonzero-weight masking inside the loop.
        if precomputed_full is None:
            if not has_cov:
                precomputed_identical = pairwise_mahalanobis_distance(
                    supports,
                    supports,
                    ground_metric,
                    as_numpy=is_numpy,
                    squared=squared_ground_cost,
                )
            else:
                validate_gaussian_ground_metric(ground_metric)
                # Keep fast precompute path only when no component masking is possible/needed.
                if (not has_weights) or (not has_zero_weight_entries(weights)):
                    precomputed_identical = pairwise_gaussian_distance(
                        supports,
                        covariances,
                        ground_metric,
                        diag_bures_approx=diag_bures_approx,
                        as_numpy=is_numpy,
                        eps=eps,
                        squared_ground_cost=squared_ground_cost,
                    )

    N = len(weights) if identical_supports else len(supports)

    pair_iter, D = initialize_pair_iteration(pairs=pairs, n_distributions=N, is_numpy=is_numpy)

    # Precompute cumulative support offsets for non-identical precomputed slicing.
    support_offsets = None
    if not identical_supports and precomputed_full is not None:
        support_offsets = np.cumsum([0] + [len(s) for s in supports])

    # Validate ground metric once (not per pair).
    if has_cov and not isinstance(ground_metric, str):
        pass  # linear map — no string validation needed
    elif has_cov:
        validate_gaussian_ground_metric(ground_metric)

    # Iterate over distribution pairs
    for p_idx, (i, j) in enumerate(pair_iter):
        w_i, w_j, use_masking, mask_i, mask_j = prepare_pair_weight_masks(
            weights=weights,
            has_weights=has_weights,
            i=i,
            j=j,
        )

        M = build_pair_cost_matrix(
            supports=supports,
            covariances=covariances,
            ground_metric=ground_metric,
            diag_bures_approx=diag_bures_approx,
            eps=eps,
            squared_ground_cost=squared_ground_cost,
            is_numpy=is_numpy,
            identical_supports=identical_supports,
            has_cov=has_cov,
            precomputed_full=precomputed_full,
            precomputed_identical=precomputed_identical,
            support_offsets=support_offsets,
            i=i,
            j=j,
            use_masking=use_masking,
            mask_i=mask_i,
            mask_j=mask_j,
        )

        if not has_weights:
            dist = _emd2(
                None,
                None,
                M=M,
                n_threads=settings.n_threads,
                reg=entropic_reg,
                numItermax=sinkhorn_max_iter,
                stopThr=sinkhorn_stop_thr,
            )
        else:
            if use_masking:
                w_i = w_i[mask_i]
                w_j = w_j[mask_j]

            # Some dataset pipelines can produce per-distribution weight vectors
            # shorter than the padded support/covariance block used for M.
            # In that case, align M to the effective weight lengths.
            m_rows, m_cols = int(M.shape[0]), int(M.shape[1])
            n_rows, n_cols = int(len(w_i)), int(len(w_j))
            if (n_rows != m_rows) or (n_cols != m_cols):
                if n_rows > m_rows or n_cols > m_cols:
                    raise ValueError(
                        "Weight/cost dimension mismatch: "
                        f"len(w_i)={n_rows}, len(w_j)={n_cols}, M.shape={tuple(M.shape)}"
                    )
                M = M[:n_rows, :n_cols]
            dist = _emd2(
                w_i,
                w_j,
                M=M,
                n_threads=settings.n_threads,
                reg=entropic_reg,
                numItermax=sinkhorn_max_iter,
                stopThr=sinkhorn_stop_thr,
            )

        write_pair_distance(D, pairs=pairs, pair_idx=p_idx, i=i, j=j, dist=dist)

    # Very small OT distances can become negative due to numerical errors.
    D = clamp_nonnegative(D)

    return D
