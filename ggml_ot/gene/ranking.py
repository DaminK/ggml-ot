from collections.abc import Sequence
from anndata import AnnData
import numpy as np
from scanpy.pl import ranking


def ggml_components(
    adata: AnnData,
    components: str | Sequence[int] | None = None,
    *,
    include_lowest: bool = True,
    n_genes: int | None = None,
    gene_symbols: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Ranks and plots genes contributions for each GGML components.

    Parameters
    ----------
    adata
        Annotated data matrix.
    components
        For example, ``'1,2,3'`` means ``[1, 2, 3]``, first, second, third
        principal component.
    include_lowest
        Whether to show the variables with both highest and lowest loadings.
    show
        Show the plot, do not return axis.
    n_genes
        Number of genes to plot for each component.
    gene_symbols
        Key for field in `.var` that stores gene symbols if you do not want to
        use `.var_names`.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.

    Examples
    --------
    TODO
    """
    if components is None:
        components = np.arange(adata.varm["W_ggml"].shape[-1]) + 1
    elif isinstance(components, str):
        components = [int(x) for x in components.split(",")]
    components = np.array(components) - 1

    if np.any(components < 0):
        msg = "Component indices must be greater than zero."
        raise ValueError(msg)

    if n_genes is None:
        n_genes = min(20, adata.n_vars)
    elif adata.n_vars < n_genes:
        msg = f"Tried to plot {n_genes} variables, but passed anndata only has {adata.n_vars}."
        raise ValueError(msg)

    adata.varm["W_ggmln"] = adata.varm["W_ggml"] / np.max(np.abs(adata.varm["W_ggml"]), axis=0)

    ranking(
        adata,
        "varm",
        "W_ggmln",
        n_points=n_genes,
        indices=components,
        include_lowest=include_lowest,
        labels=gene_symbols if gene_symbols is None else adata.var[gene_symbols],
    )

    del adata.varm["W_ggmln"]

    # sc.pl.savefig_or_show("ggml_components", show=show, save=save)
