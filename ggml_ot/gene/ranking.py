from collections.abc import Sequence
from typing import Literal

from anndata import AnnData
import numpy as np
from scanpy.plotting._utils import savefig_or_show
from scanpy.pl import ranking


def _shown_genes(
    scores: np.ndarray,
    components: np.ndarray,
    n_genes: int,
    gene_names: np.ndarray,
    polarity: Literal["positive", "negative", "both"],
) -> np.ndarray:
    top_genes = np.empty((len(components), n_genes), dtype=object)
    for i, component in enumerate(components):
        order = np.argsort(scores[:, component])[::-1]
        if polarity == "both":
            highest = order[: n_genes // 2]
            lowest = order[-(n_genes - n_genes // 2) :]
            order = np.concatenate([highest, lowest])
        else:
            order = order[:n_genes]
        top_genes[i, :] = gene_names[order]
    return top_genes


def _ranking(
    adata: AnnData,
    components: str | Sequence[int] | None = None,
    *,
    n_genes: int | None = None,
    polarity: Literal["positive", "negative", "both"] = "both",
    gene_symbols: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
) -> np.ndarray:
    """Rank and plot gene contributions for each GGML component.

    Parameters
    ----------
    adata
        Annotated data matrix.
    components
        For example, ``'1,2,3'`` means ``[1, 2, 3]``, first, second, third
        principal component.
    n_genes
        Number of genes to plot for each component.
    polarity
        Which loading direction to show: ``'positive'`` for largest positive loadings,
        ``'negative'`` for largest-magnitude negative loadings, or ``'both'`` for
        positive and negative loadings.
        For ``'negative'``, scores are sign-flipped before calling Scanpy's
        ranking plotter, so the plotted y-axis is inverted relative to the
        original GGML loadings.
    gene_symbols
        Key for field in `.var` that stores gene symbols if you do not want to use `.var_names`.
    show
        Show the plot.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_components, n_genes)`` with the shown genes. For
        ``polarity='both'``, the first half are the highest positive loadings
        and the second half are the lowest negative loadings.

    """
    if "W_ggml" not in adata.varm:
        raise ValueError(
            "No GGML components, run GGML first. This error also occurs if you have trained on a low-dimensional representation (e.g. use_rep = PCA) without the inverse transform in the adata (e.g. PCA loadings in varm)."
        )

    if components is None:
        components = np.arange(adata.varm["W_ggml"].shape[-1]) + 1
    elif isinstance(components, str):
        components = [int(x) for x in components.split(",")]
    components = np.array(components) - 1

    if np.any(components < 0):
        msg = "Component indices must be greater than zero."
        raise ValueError(msg)

    if polarity not in {"positive", "negative", "both"}:
        msg = "polarity must be one of {'positive', 'negative', 'both'}."
        raise ValueError(msg)

    if n_genes is None:
        n_genes = min(20, adata.n_vars)
    elif adata.n_vars < n_genes:
        msg = f"Tried to plot {n_genes} variables, but passed anndata only has {adata.n_vars}."
        raise ValueError(msg)

    scores = adata.varm["W_ggml"] / np.max(np.abs(adata.varm["W_ggml"]), axis=0)
    if polarity == "negative":
        scores = -scores

    gene_names = np.asarray(adata.var_names if gene_symbols is None else adata.var[gene_symbols])
    top_genes = _shown_genes(scores, components, n_genes, gene_names, polarity)

    if show is False and save in {None, False}:
        return top_genes

    adata.varm["W_ggmln"] = scores

    try:
        ranking(
            adata,
            "varm",
            "W_ggmln",
            n_points=n_genes,
            indices=components,
            include_lowest=(polarity == "both"),
            labels=gene_symbols if gene_symbols is None else adata.var[gene_symbols],
            show=False,
        )

        savefig_or_show("_ranking", show=show, save=save)
    finally:
        del adata.varm["W_ggmln"]

    return top_genes


def abs_ranking(
    adata: AnnData,
    components: np.ndarray,
    n_genes: int,
    gene_symbols: str | None,
) -> np.ndarray:
    """Return top genes per component by absolute GGML loading."""
    gene_names = np.asarray(adata.var_names if gene_symbols is None else adata.var[gene_symbols])
    top_indices = np.argsort(np.abs(adata.varm["W_ggml"]), axis=0)[::-1].T
    return gene_names[top_indices[:, :n_genes]]
