from __future__ import annotations

import warnings

import numpy as np
import torch

from ggml_ot._utils._weights import normalize_weight_rows


def validate_fit_inputs(
    *,
    x: np.ndarray,
    distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
    component_sharing: str,
    k_comps: int | list[int] | tuple[int, ...] | None,
    refit: str,
) -> None:
    """Validate shared fit inputs before GMM orchestration."""
    if x.ndim != 2:
        raise ValueError(f"x must have shape (n_cells, n_features). Got {x.shape}.")
    finite_mask = np.isfinite(x)
    if not bool(np.all(finite_mask)):
        bad = int(np.size(x) - int(np.count_nonzero(finite_mask)))
        raise ValueError(
            f"x contains {bad} non-finite values (NaN/Inf). "
            "Please clean the representation before fit_gmm (e.g., remove or impute non-finite entries)."
        )
    if distribution_index.ndim != 1:
        raise ValueError("distribution_index must have shape (n_cells,).")
    if distribution_index.shape[0] != x.shape[0]:
        raise ValueError("distribution_index length must match number of rows in x.")
    if int(distribution_ids.shape[0]) == 0:
        raise ValueError("distribution_ids must not be empty.")

    if k_comps is None:
        raise ValueError("Pass k_comps as an int or iterable of ints.")
    if refit not in {"full", "none"}:
        raise ValueError("refit must be 'full' or 'none'.")
    if component_sharing not in {"global", "sample_specific"}:
        raise ValueError("component_sharing must be 'global' or 'sample_specific'.")


def validate_gaussian_ground_metric(ground_metric) -> None:
    """Validate supported Gaussian ground-metric selector values."""
    if isinstance(ground_metric, str) and ground_metric not in {"euclidean", "kl_divergence"}:
        raise ValueError(
            "For Gaussian distances, only 'euclidean' and 'kl_divergence' are supported as string ground metrics."
        )


def sanitize_degenerate_sample_specific_gmm_components(
    *,
    supports: np.ndarray,
    distribution_weights: np.ndarray,
    distribution_ids,
    eps: float,
    stage: str,
) -> np.ndarray:
    """Zero and renormalize degenerate sample-specific GMM components.

    Parameters
    ----------
    supports
        Sample-specific component means with shape ``(D, K, d)``.
    distribution_weights
        Per-distribution component weights with shape ``(D, K)``.
    distribution_ids
        Identifiers aligned to the first axis of ``supports`` / ``distribution_weights``.
    eps
        Numerical epsilon used by the EM updates.
    stage
        Short label describing where validation is happening.

    Returns
    -------
    np.ndarray
        Row-normalized per-distribution component weights with degenerate
        floor-weight components set to exactly zero.

    Raises
    ------
    ValueError
        If the inputs are malformed or sanitization would remove all mass from
        a distribution.

    Warns
    -----
    RuntimeWarning
        If any degenerate components are detected and zeroed.
    """
    supports = np.asarray(supports, dtype=np.float64)
    distribution_weights = np.asarray(distribution_weights, dtype=np.float64)
    distribution_ids = [str(x) for x in np.asarray(distribution_ids).tolist()]

    if supports.ndim != 3:
        raise ValueError(f"supports must have shape (D, K, d). Got {supports.shape}.")
    if distribution_weights.ndim != 2:
        raise ValueError(f"distribution_weights must have shape (D, K). Got {distribution_weights.shape}.")
    if supports.shape[:2] != distribution_weights.shape:
        raise ValueError(
            "supports and distribution_weights shapes must align on (D, K). "
            f"Got supports={supports.shape}, distribution_weights={distribution_weights.shape}."
        )
    if supports.shape[0] != len(distribution_ids):
        raise ValueError(
            "distribution_ids length must match first axis of supports. "
            f"Got len(distribution_ids)={len(distribution_ids)}, supports.shape[0]={supports.shape[0]}."
        )

    eps_val = float(eps)
    weight_floor_tol = max(1e-12, 2.0 * eps_val)
    norm_tol = max(1e-12, np.sqrt(float(supports.shape[-1])) * eps_val)
    norms = np.linalg.norm(supports, axis=-1)
    bad = (norms <= norm_tol) & (distribution_weights > 0.0) & (distribution_weights <= weight_floor_tol)
    if not bool(np.any(bad)):
        return normalize_weight_rows(distribution_weights)

    sanitized_weights = np.array(distribution_weights, copy=True, dtype=np.float64)
    sanitized_weights[bad] = 0.0
    row_mass = np.sum(sanitized_weights, axis=1)
    empty_rows = np.flatnonzero(row_mass <= max(1e-12, eps_val))
    if empty_rows.size > 0:
        affected_rows = [distribution_ids[int(idx)] for idx in empty_rows[:5]]
        raise ValueError(
            "Degenerate sample-specific GMM sanitization removed all mixture mass "
            f"during {stage}. Affected distributions: {affected_rows}. "
            "Reduce `k_comps` or use a richer `k_comps` grid."
        )
    sanitized_weights = normalize_weight_rows(sanitized_weights)

    affected = []
    for dist_idx, dist_id in enumerate(distribution_ids):
        comp_idx = np.flatnonzero(bad[dist_idx])
        if comp_idx.size == 0:
            continue
        affected.append(f"{dist_id}: {comp_idx.tolist()}")
        if len(affected) >= 5:
            break

    total_bad = int(np.count_nonzero(bad))
    total_dists = int(np.count_nonzero(np.any(bad, axis=1)))
    warnings.warn(
        f"[{stage}] Zeroed {total_bad} degenerate GMM components across "
        f"{total_dists} distributions (near-zero norm with floor-level weight). "
        f"Examples: {', '.join(affected)}",
        RuntimeWarning,
        stacklevel=2,
    )
    return sanitized_weights


def assert_finite_cost_matrix(cost_matrix) -> None:
    """Fail fast when an OT cost matrix contains NaN/Inf values."""
    if torch.is_tensor(cost_matrix):
        finite = torch.isfinite(cost_matrix)
        if bool(finite.all().item()):
            return
        total = int(cost_matrix.numel())
        finite_count = int(finite.sum().item())
        bad = total - finite_count
        if finite_count > 0:
            valid = cost_matrix[finite]
            min_val = float(valid.min().item())
            max_val = float(valid.max().item())
        else:
            min_val = float("nan")
            max_val = float("nan")
        # FUTURE: consider warning + recovery strategy instead of hard failure.
        raise ValueError(
            "OT cost matrix `M` contains non-finite values "
            f"(bad={bad}/{total}, min_finite={min_val}, max_finite={max_val}). "
            "Failing fast to expose upstream numerical instability."
        )

    matrix = np.asarray(cost_matrix)
    finite = np.isfinite(matrix)
    if bool(np.all(finite)):
        return
    total = int(matrix.size)
    finite_count = int(np.count_nonzero(finite))
    bad = total - finite_count
    if finite_count > 0:
        valid = matrix[finite]
        min_val = float(np.min(valid))
        max_val = float(np.max(valid))
    else:
        min_val = float("nan")
        max_val = float("nan")
    # FUTURE: consider warning + recovery strategy instead of hard failure.
    raise ValueError(
        "OT cost matrix `M` contains non-finite values "
        f"(bad={bad}/{total}, min_finite={min_val}, max_finite={max_val}). "
        "Failing fast to expose upstream numerical instability."
    )


def assert_finite_tensor(
    value: torch.Tensor,
    *,
    name: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
) -> None:
    """Fail fast for non-finite optimization state."""
    if bool(torch.isfinite(value).all().item()):
        return
    total = int(value.numel())
    finite_mask = torch.isfinite(value)
    finite_count = int(finite_mask.sum().item())
    bad = total - finite_count
    if finite_count > 0:
        valid = value[finite_mask]
        min_val = float(valid.min().item())
        max_val = float(valid.max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")
    if epoch is None:
        where = "stage=forward"
    else:
        where = f"epoch={epoch}" if batch_idx is None else f"epoch={epoch}, batch={batch_idx}"
    # FUTURE: consider warning + recovery strategy instead of hard failure.
    raise ValueError(
        f"Non-finite `{name}` detected during GGML training ({where}; "
        f"bad={bad}/{total}, min_finite={min_val}, max_finite={max_val}). "
        "Failing fast to expose upstream numerical instability."
    )


__all__ = [
    "validate_fit_inputs",
    "validate_gaussian_ground_metric",
    "sanitize_degenerate_sample_specific_gmm_components",
    "assert_finite_cost_matrix",
    "assert_finite_tensor",
]
