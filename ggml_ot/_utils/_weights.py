"""Weight / probability helpers.

Centralises weight emptiness checks, 1-D and row-wise normalisation,
and weight-to-covariance alignment so that every consumer uses exactly
the same logic.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Simple 1-D normalisation (numpy only, strict)
# ---------------------------------------------------------------------------


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize a 1-D weight vector to sum to 1.

    Unlike :func:`normalize_weight_vector` this function raises on
    non-positive total mass instead of falling back to a uniform
    distribution.  Use it during data preprocessing where invalid
    weights should be caught early.

    Parameters
    ----------
    weights
        Non-negative array whose sum is positive.

    Returns
    -------
    np.ndarray
        Array of the same shape summing to 1 (``float64``).

    Raises
    ------
    ValueError
        If the total weight is non-positive.
    """
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0:
        raise ValueError("Encountered non-positive mixture weight sum while building dataset.")
    return weights / np.clip(denom, 1e-12, None)


# ---------------------------------------------------------------------------
# Emptiness check
# ---------------------------------------------------------------------------


def is_empty_weights(weights) -> bool:
    """Return ``True`` when *weights* is ``None``, empty, or zero-element."""
    if weights is None:
        return True
    if isinstance(weights, torch.Tensor):
        return weights.numel() == 0
    if isinstance(weights, np.ndarray):
        return weights.size == 0
    return hasattr(weights, "__len__") and len(weights) == 0


# ---------------------------------------------------------------------------
# 1-D normalisation (supports both numpy *and* torch)
# ---------------------------------------------------------------------------


def normalize_weight_vector(weights, *, eps: float = 1e-12):
    """Normalize a 1-D weight vector to sum to 1.

    Handles both :class:`numpy.ndarray` and :class:`torch.Tensor`.
    Non-finite / negative values are cleaned before normalisation; if the
    total mass is near-zero, a uniform distribution is returned.
    """
    if isinstance(weights, torch.Tensor):
        vec = weights.to(dtype=torch.float64)
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D weight vector, got tensor shape={tuple(vec.shape)}.")
        vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        mass = float(vec.sum().item())
        if not np.isfinite(mass) or mass <= eps:
            if vec.numel() == 0:
                return vec
            return torch.full_like(vec, 1.0 / float(vec.numel()))
        return vec / mass

    vec = np.asarray(weights, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"Expected 1D weight vector, got array shape={vec.shape}.")
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = np.clip(vec, a_min=0.0, a_max=None)
    mass = float(np.sum(vec))
    if not np.isfinite(mass) or mass <= eps:
        if vec.size == 0:
            return vec
        return np.full(vec.shape, 1.0 / float(vec.size), dtype=np.float64)
    return vec / mass


# ---------------------------------------------------------------------------
# Container normalisation (multi-row / list / tuple)
# ---------------------------------------------------------------------------


def normalize_weights_container(weights):
    """Normalize weight rows once per ``compute_OT`` call.

    Accepts 1-D vectors, 2-D arrays/tensors, or lists/tuples of 1-D
    vectors and returns the same container type with each row normalised.
    """
    if is_empty_weights(weights):
        return weights

    if isinstance(weights, torch.Tensor):
        if weights.ndim == 1:
            return normalize_weight_vector(weights)
        if weights.ndim == 2:
            return torch.stack(
                [normalize_weight_vector(weights[i]) for i in range(weights.shape[0])],
                dim=0,
            )
        raise ValueError(f"Expected weights tensor with ndim in {{1,2}}, got shape={tuple(weights.shape)}.")

    if isinstance(weights, np.ndarray):
        if weights.ndim == 1:
            return normalize_weight_vector(weights)
        if weights.ndim == 2:
            return np.stack(
                [normalize_weight_vector(weights[i]) for i in range(weights.shape[0])],
                axis=0,
            )
        raise ValueError(f"Expected weights array with ndim in {{1,2}}, got shape={weights.shape}.")

    if isinstance(weights, (list, tuple)):
        normalized = [normalize_weight_vector(w) for w in weights]
        return tuple(normalized) if isinstance(weights, tuple) else normalized

    raise TypeError(
        f"Unsupported weights container type: {type(weights)!r}. "
        "Expected torch.Tensor, np.ndarray, list, tuple, or None."
    )


# ---------------------------------------------------------------------------
# Row-wise normalisation (numpy only, 2-D matrices)
# ---------------------------------------------------------------------------


def normalize_weight_rows(weights: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return row-normalized non-negative mixture weights.

    - Non-finite values are treated as zero.
    - Negative values are clipped to zero.
    - Rows with near-zero mass are replaced by a uniform distribution.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"weights must have shape (n_rows, k). Got {w.shape}.")

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, a_min=0.0, a_max=None)

    k = int(w.shape[1])
    if k < 1:
        raise ValueError("weights must have at least one component column.")

    row_sums = w.sum(axis=1, keepdims=True)
    bad_rows = row_sums[:, 0] <= float(eps)
    good_rows = ~bad_rows

    if np.any(good_rows):
        w[good_rows] /= row_sums[good_rows]
    if np.any(bad_rows):
        w[bad_rows] = 1.0 / float(k)

    # Final renormalization to guard floating-point drift.
    w /= np.clip(w.sum(axis=1, keepdims=True), a_min=float(eps), a_max=None)
    return w


# ---------------------------------------------------------------------------
# Persisted model-weight canonicalization helpers
# ---------------------------------------------------------------------------


def canonicalize_model_weights(model_weights, *, n_components: int) -> np.ndarray | None:
    """Bring stored model weights into canonical shape ``(n_rows, K)``.

    Parameters
    ----------
    model_weights
        Raw weights from persisted GMM payload (commonly stored under ``model['pi']``).
        Can be ``None``, empty, 1-D ``(K,)``, or 2-D ``(D, K)`` / ``(K, 1)``.
    n_components
        Expected number of mixture components *K*.

    Returns
    -------
    np.ndarray | None
        Row-normalized 2-D array ``(n_rows, K)`` or ``None`` when unavailable.
    """
    if model_weights is None:
        return None

    weights = np.asarray(model_weights, dtype=np.float64)
    if weights.size == 0:
        return None

    weights = np.squeeze(weights)
    if weights.ndim == 1:
        if weights.shape[0] != n_components:
            raise ValueError(
                f"Stored model weights have incompatible shape {weights.shape}; expected ({n_components},)."
            )
        return normalize_weights(weights).reshape(1, -1)

    if weights.ndim == 2:
        if weights.shape[1] == n_components:
            return np.asarray(
                [normalize_weights(row) for row in weights],
                dtype=np.float64,
            )
        if weights.shape == (n_components, 1):
            return normalize_weights(weights[:, 0]).reshape(1, -1)

    raise ValueError(
        f"Stored model weights have unsupported shape {np.asarray(model_weights).shape}; expected (K,) or (D, K)."
    )


def get_distribution_weights(
    weight_rows: np.ndarray | None,
    *,
    dist_idx: int,
    n_distributions: int,
    n_components: int,
) -> np.ndarray:
    """Return one distribution weight vector ``(K,)`` from canonical weight rows."""
    if weight_rows is None:
        return np.full(n_components, 1.0 / n_components, dtype=np.float64)
    if weight_rows.shape[0] == 1:
        return weight_rows[0]
    if weight_rows.shape[0] == n_distributions:
        return weight_rows[dist_idx]
    raise ValueError(
        "Stored model weight row count is incompatible with distributions: "
        f"weight rows={weight_rows.shape[0]}, distributions={n_distributions}."
    )


# ---------------------------------------------------------------------------
# Weight ↔ covariance alignment
# ---------------------------------------------------------------------------


def align_weights_to_components(weights: torch.Tensor, sigma_shape: tuple[int, ...], target_n: int) -> torch.Tensor:
    """Align optional component weights with flattened covariance components.

    Used by :func:`mutual_information_loss` to broadcast batch / component
    weight dimensions to a flat vector of per-component KL values.
    """
    # If weights have a trailing component dimension matching target_n,
    # aggregate over preceding axes (e.g. (B, C) -> (C,)).
    if weights.ndim >= 2 and weights.shape[-1] == target_n:
        return weights.reshape(-1, target_n).sum(dim=0)

    w = weights.reshape(-1)

    if w.numel() == target_n:
        return w

    # Common GMM batch case: sigma has shape (B, C, D, D), and weights
    # may be of shape (C,) or (B, C).
    if len(sigma_shape) == 4:
        batch_size, n_components = sigma_shape[0], sigma_shape[1]
        if w.numel() == n_components:
            return w.repeat(batch_size)
        if w.numel() == batch_size:
            return w.repeat_interleave(n_components)

    # Generic fallback: if weights can be grouped into component blocks,
    # sum per component.
    if target_n > 0 and w.numel() % target_n == 0:
        return w.reshape(-1, target_n).sum(dim=0)

    raise ValueError(f"Could not align weights (numel={w.numel()}) with covariance components (numel={target_n}).")


def aggregate_distribution_weights(
    responsibilities: np.ndarray,
    distribution_index: np.ndarray,
    n_distributions: int,
) -> np.ndarray:
    """Aggregate cell-level responsibilities to per-distribution weights."""
    resp = np.asarray(responsibilities, dtype=np.float64)
    dist_idx = np.asarray(distribution_index, dtype=int)

    if resp.ndim != 2:
        raise ValueError("responsibilities must have shape (n_cells, K).")
    if dist_idx.ndim != 1:
        raise ValueError("distribution_index must have shape (n_cells,).")
    if dist_idx.shape[0] != resp.shape[0]:
        raise ValueError("distribution_index length must match number of rows in responsibilities.")

    k = int(resp.shape[1])
    weights = np.zeros((int(n_distributions), k), dtype=np.float64)
    np.add.at(weights, dist_idx, resp)

    counts = np.bincount(dist_idx, minlength=int(n_distributions)).astype(np.float64)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"At least one distribution has no assigned cells: {missing}")

    weights /= counts[:, None]
    return normalize_weight_rows(weights)


# ---------------------------------------------------------------------------
# Nonzero-weight component masks
# ---------------------------------------------------------------------------


def get_nonzero_weight_mask(weights_row):
    """Return boolean mask selecting entries with strictly positive mass."""
    if isinstance(weights_row, torch.Tensor):
        mask = weights_row > 0
        if not bool(mask.any().item()):
            return torch.ones_like(mask, dtype=torch.bool)
        return mask

    row = np.asarray(weights_row)
    mask = row > 0
    if not bool(np.any(mask)):
        return np.ones_like(mask, dtype=bool)
    return mask


def has_zero_weight_entries(weights_row) -> bool:
    """Return ``True`` when at least one entry has zero/negative mass."""
    if isinstance(weights_row, torch.Tensor):
        return bool((weights_row <= 0).any().item())
    row = np.asarray(weights_row)
    return bool(np.any(row <= 0))
