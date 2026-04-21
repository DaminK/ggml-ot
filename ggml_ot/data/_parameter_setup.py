"""Canonical ``ggml_preprocessed`` payload builder.

Every code path that writes ``adata.uns["ggml_preprocessed"]`` should use
:func:`set_preprocessed` so the schema stays consistent in one place.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def set_preprocessed(
    adata,
    *,
    supports: list[np.ndarray] | np.ndarray,
    covariances: list[np.ndarray] | np.ndarray | None,
    weights: list[np.ndarray] | np.ndarray | None,
    distribution_labels: list[int] | np.ndarray,
    identical_supports: bool,
    use_rep: str | None,
    index_mask: list[bool] | np.ndarray | None = None,
) -> Any:
    """Write the canonical ``ggml_preprocessed`` payload into *adata*.

    Parameters
    ----------
    adata
        AnnData object whose ``.uns`` dict will be updated.
    supports
        Per-distribution support points (centroids or empirical cells).
    covariances
        Per-distribution covariance matrices, or ``None``.
    weights
        Per-distribution mixture weights, or ``None`` (uniform).
    distribution_labels
        Integer class label for each distribution.
    identical_supports
        Whether all distributions share the same support set.
    use_rep
        The ``obsm`` key used, or ``None`` when raw ``.X`` was used.
    index_mask
        Boolean mask over ``adata.obs`` indicating which cells were
        retained.  Defaults to ``[True] * adata.n_obs``.

    Returns
    -------
    adata
        The same object, mutated in place for convenience.
    """
    if index_mask is None:
        index_mask = [True] * adata.n_obs

    adata.uns["ggml_preprocessed"] = {
        "supports": supports,
        "covariances": covariances,
        "weights": weights,
        "distribution_labels": distribution_labels,
        "index_mask": index_mask,
        "identical_supports": identical_supports,
        "use_rep": use_rep,
    }
    return adata
