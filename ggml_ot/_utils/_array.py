"""Array backend conversions and dtype casting.

Provides helpers to move data between numpy and torch, cast dtypes,
check for empty / None containers, and unified squareform conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy as sp
import torch


def to_numpy(x: Any) -> np.ndarray:
    """Convert *x* to a numpy ``float64`` array.

    Handles plain arrays, torch tensors (incl. GPU), and generic iterables.
    """
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def snapshot_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a detached/copied numpy snapshot while preserving dtype."""
    if isinstance(x, torch.Tensor):
        return x.detach().clone().cpu().numpy()
    return np.array(x, copy=True)


def to_float32(arrays, backend: str | None = None):
    """Convert input arrays or tensors to float32.

    Parameters
    ----------
    arrays
        A numpy array, PyTorch tensor, or a list/tuple of them.
    backend
        ``"torch"`` → always return a :class:`torch.Tensor`;
        ``"numpy"`` → always return a :class:`numpy.ndarray`;
        ``None``    → keep the original backend.

    Returns
    -------
    Converted data in float32.
    """
    if backend == "torch":
        if arrays is None:
            return torch.tensor([])

        # Fast-path for stacked list/tuple inputs.
        if isinstance(arrays, (list, tuple)):
            if len(arrays) == 0:
                return torch.tensor([])

            if all(isinstance(arr, np.ndarray) for arr in arrays):
                try:
                    return torch.from_numpy(np.asarray(arrays, dtype=np.float32))
                except ValueError:
                    pass

            if all(isinstance(arr, torch.Tensor) for arr in arrays):
                try:
                    return torch.stack([arr.to(torch.float32) for arr in arrays])
                except RuntimeError:
                    pass

        return torch.as_tensor(arrays, dtype=torch.float32)

    elif backend == "numpy":
        return np.asarray(arrays, dtype=np.float32) if arrays is not None else None

    elif backend is None:
        if arrays is None:
            return None
        if isinstance(arrays, (list, tuple)):
            return [to_float32(arr) for arr in arrays]
        if isinstance(arrays, np.ndarray):
            return np.asarray(arrays, dtype=np.float32)
        if isinstance(arrays, torch.Tensor):
            return arrays.to(torch.float32)

    raise TypeError(f"Unknown backend of type {type(arrays)}")


def not_none(arrays) -> bool:
    """Return ``True`` if *arrays* is not ``None`` and not empty."""
    return (arrays is not None) and (len(arrays) > 0) and arrays.shape[-1] != 0


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def is_numpy_backend(data) -> bool:
    """Determine whether *data* lives on the numpy or torch backend.

    Returns ``True`` for numpy arrays, ``False`` for torch tensors.
    Lists/tuples are inspected via their first element.

    Raises
    ------
    TypeError
        If the backend cannot be determined.
    ValueError
        If *data* is an empty sequence.
    """
    if isinstance(data, torch.Tensor):
        return False
    if isinstance(data, np.ndarray):
        return True
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("`supports` must not be empty.")
        first = data[0]
        if isinstance(first, torch.Tensor):
            return False
        if isinstance(first, np.ndarray):
            return True
        raise TypeError(
            f"Unsupported list/tuple element backend of type {type(first)}. Expected np.ndarray or torch.Tensor."
        )
    raise TypeError(f"Unsupported backend of type {type(data)}")


# ---------------------------------------------------------------------------
# Torch conversion
# ---------------------------------------------------------------------------


def as_torch(
    x,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert *x* to a :class:`torch.Tensor` on *device* with *dtype*.

    Accepts numpy arrays, torch tensors, and generic iterables.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def cast_many_to_torch(
    *values: Any,
    device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Cast many inputs to torch tensors on a given device/dtype."""
    return tuple(as_torch(value, device=device, dtype=dtype) for value in values)


def to_output_backend(
    value: torch.Tensor,
    *,
    as_numpy: bool = False,
    clamp_min: float | None = None,
):
    """Apply optional clamping and convert output to requested backend."""
    out = value
    if clamp_min is not None:
        out = out.clamp(min=float(clamp_min))
    if as_numpy:
        return out.detach().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Distribution list helpers
# ---------------------------------------------------------------------------


def as_distribution_list(x):
    """Coerce *x* into a list of 2-D arrays (one per distribution).

    If *x* is already a list/tuple it is returned as-is.  A single 2-D
    array becomes a one-element list; a 3-D+ tensor is split along
    axis 0.
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    arr = to_numpy(x)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim >= 3:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Expected at least 2D array-like data, got shape={arr.shape}.")


# ---------------------------------------------------------------------------
# Squareform (condensed ↔ square distance conversion)
# ---------------------------------------------------------------------------


def squareform(x):
    """Unified :func:`scipy.spatial.distance.squareform` for numpy and torch.

    For numpy arrays the scipy implementation is used directly.
    For torch tensors a custom GPU-friendly implementation is used.

    Parameters
    ----------
    x
        A condensed distance vector or a square distance matrix,
        either as :class:`numpy.ndarray` or :class:`torch.Tensor`.
    """
    if isinstance(x, np.ndarray):
        return sp.spatial.distance.squareform(x)
    if isinstance(x, torch.Tensor):
        return _torch_squareform(x)
    raise TypeError(f"Input type {type(x)} not supported. Expected np.ndarray or torch.Tensor")


def slice_matrix(matrix, row_indexer, col_indexer):
    """Slice a 2-D matrix by row/column indexers for numpy and torch backends."""
    if isinstance(matrix, torch.Tensor):
        return matrix[row_indexer][:, col_indexer]
    arr = np.asarray(matrix)
    return arr[np.ix_(row_indexer, col_indexer)]


def clamp_nonnegative(x):
    """Clamp array/tensor values to be non-negative while preserving backend."""
    if isinstance(x, torch.Tensor):
        return x.clamp(min=0)
    arr = np.asarray(x)
    return np.maximum(arr, 0)


def _torch_squareform(x: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`scipy.spatial.distance.squareform`.

    Converts a condensed distance vector to a square symmetric matrix
    (zero diagonal) or vice-versa.
    """
    if x.ndim == 1:
        n = int(np.ceil(np.sqrt(x.numel() * 2)))
        if n * (n - 1) // 2 != x.numel():
            raise ValueError("Input vector size must be n*(n-1)/2 for some integer n")
        r, c = torch.triu_indices(n, n, offset=1, device=x.device)
        out = torch.zeros((n, n), dtype=x.dtype, device=x.device)
        out[r, c] = x
        out[c, r] = x
        return out
    if x.ndim == 2:
        if x.shape[0] != x.shape[1]:
            raise ValueError("Input matrix must be square")
        n = x.shape[0]
        r, c = torch.triu_indices(n, n, offset=1, device=x.device)
        return x[r, c]
    raise ValueError("Input must be 1D vector or 2D square matrix")
