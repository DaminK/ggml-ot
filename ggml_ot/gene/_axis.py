"""Latent axis ranking and gene-score utilities.

Provides tidy DataFrame outputs for per-axis gene rankings
derived from the GGML loading matrix ``adata.varm["W_ggml"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def rank_latent_axes(
    adata: AnnData,
    *,
    axes: list[int] | None = None,
    gene_symbols: str | None = None,
) -> pd.DataFrame:
    """Rank genes by their contribution to each GGML latent axis.

    Parameters
    ----------
    adata
        Annotated data matrix with ``adata.varm["W_ggml"]``.
    axes
        0-indexed axis indices to rank. ``None`` ranks all axes.
    gene_symbols
        Column in ``adata.var`` for gene names.  Falls back to
        ``adata.var_names`` when ``None``.

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns:
        ``axis``, ``gene``, ``score``, ``abs_score``, ``sign``, ``rank``.
    """
    W = _get_loading_matrix(adata)
    n_genes, n_axes = W.shape

    if axes is None:
        axes = list(range(n_axes))
    else:
        _validate_axes(axes, n_axes)

    gene_names = _resolve_gene_names(adata, gene_symbols)

    rows: list[dict] = []
    for ax in axes:
        scores = W[:, ax].copy()
        order = np.argsort(-np.abs(scores))
        for rank_i, gene_idx in enumerate(order):
            s = float(scores[gene_idx])
            rows.append(
                {
                    "axis": ax,
                    "gene": gene_names[gene_idx],
                    "score": s,
                    "abs_score": abs(s),
                    "sign": int(np.sign(s)),
                    "rank": rank_i + 1,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_loading_matrix(adata: AnnData) -> np.ndarray:
    """Return ``W_ggml`` from ``adata.varm``, raising if absent."""
    if "W_ggml" not in adata.varm:
        raise ValueError(
            "No GGML loading matrix found in adata.varm['W_ggml']. "
            "Train GGML first, or check that the inverse transform is available "
            "when training on a low-dimensional representation (use_rep)."
        )
    return np.asarray(adata.varm["W_ggml"])


def _resolve_gene_names(adata: AnnData, gene_symbols: str | None) -> np.ndarray:
    if gene_symbols is not None:
        return np.asarray(adata.var[gene_symbols])
    return np.asarray(adata.var_names)


def _validate_axes(axes: list[int], n_axes: int) -> None:
    for ax in axes:
        if ax < 0 or ax >= n_axes:
            raise ValueError(f"Axis index {ax} out of range [0, {n_axes}).")
