import numpy as np
import gprofiler

import matplotlib.pyplot as plt
import seaborn as sns

from ggml_ot.plot._utils import savefig_or_show


def top_ranked(
    adata, components=None, n_genes=None, gene_symbols: str | None = None, up_regulated=False, down_regulated=False
):
    """Return genes with the highest contributions to each GGML component.

    Parameters
    ----------
    adata
        Annotated data matrix.
    components
        For example, ``'1,2,3'`` means ``[1, 2, 3]``, first, second, third
        GGML component. Defaults to all GGML components.
    n_genes
        Number of genes to rank for each component.
    gene_symbols
        Key for field in `.var` that stores gene symbols if you do not want to
        use `.var_names`.
    up_regulated
        If True, rank by positive loadings only.
    down_regulated
        If True, rank by negative loadings only. When both flags are equal,
        absolute loadings are used.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_components, n_genes)`` with gene names.
    """
    if components is None:
        components = np.arange(adata.varm["W_ggml"].shape[-1]) + 1
    elif isinstance(components, str):
        components = [int(x) for x in components.split(",")]
    components = np.array(components) - 1

    gene_name = np.asarray(adata.var.index if gene_symbols is None else adata.var[gene_symbols])

    top_genes = np.zeros((len(components), n_genes), dtype=object)
    for c in components:
        gene_regulation = adata.varm["W_ggml"][:, c]
        if down_regulated and not up_regulated:
            gene_regulation = -gene_regulation
        elif up_regulated == down_regulated:
            gene_regulation = np.abs(gene_regulation)

        top_genes[c, :] = gene_name[np.argsort(gene_regulation)[-n_genes:]]

    if down_regulated and not up_regulated:
        top_genes = np.flip(top_genes, axis=1)

    return top_genes


def enrichment(
    adata,
    n_genes=50,
    components=None,
    gene_symbols: str | None = None,
    ordered=False,
    alpha=0.05,
    organism="hsapiens",
    orient="v",
    save: str | bool | None = None,
    show: bool | None = None,
    **kwargs,
):
    """Performs enrichment analysis on top-ranked genes and visualizes the enriched biological terms.

    Parameters
    ----------
    adata
        Anndata object
    n_genes
        Number of considered top-ranked genes
    components
        For example, ``'1,2,3'`` means ``[1, 2, 3]``, first, second, third GGML component. Defaults to all GGML components.
    gene_symbols
        Key for field in `.var` that stores gene symbols if you do not want to use `.var_names`.
    ordered
        Whether the gene lists are ordered by importance
    alpha
        Threshold for significance in enrichment
    organism
        Organism ID for g:Profiler
    orient
        Whether to layout the plots horizontally (``"h"``) or vertically (``"v"``).
    show
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves under the default name into
        ``settings.figdir``.  A *str* is used as the filename.
    kwargs
        Passed to `sns.barplot`

    """
    if components is None:
        components = np.arange(adata.varm["W_ggml"].shape[-1]) + 1
    elif isinstance(components, str):
        components = [int(x) for x in components.split(",")]
    components = np.array(components) - 1

    top_genes = top_ranked(adata, components, n_genes, gene_symbols)

    # Gene Enrichment
    if orient == "h":
        layout = (1, len(top_genes))
        figsize = (5 * len(top_genes), 6)
    elif orient == "v":
        layout = (len(top_genes), 1)
        figsize = (7, 5 * len(top_genes))
    fig, axs = plt.subplots(*layout, figsize=figsize)

    for c, c_top_genes in enumerate(top_genes):
        gp = gprofiler.GProfiler(return_dataframe=True)
        enrich = gp.profile(
            query=list(c_top_genes),
            ordered=ordered,
            user_threshold=alpha,
            organism=organism,
        )
        enrich["NES"] = -np.log10(enrich["p_value"])

        if orient == "h":
            sns.barplot(x="name", y="p_value", data=enrich, ax=axs[c], **kwargs)
            axs[c].tick_params(axis="x", rotation=90)
            if c == 0:
                axs[c].set_ylim(0, alpha * 1.05)
            else:
                axs[c].get_yaxis().set_visible(False)
        elif orient == "v":
            sns.barplot(x="p_value", y="name", data=enrich, ax=axs[c], **kwargs)
            if c == len(top_genes) - 1:
                axs[c].set_xlim(0, alpha * 1.05)
            else:
                axs[c].get_xaxis().set_visible(False)

            axs[c].vlines(
                x=alpha,
                ymin=axs[c].get_ylim()[0],
                ymax=axs[c].get_ylim()[1],
                color="black",
                label="axvline - full height",
                linestyles="dashed",
            )
            if c == 0:
                axs[c].text(
                    x=alpha,
                    y=axs[c].get_ylim()[1],
                    s="alpha",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                )

        axs[c].set_title(f"W_GGML {c + 1}")

    fig.suptitle("Enriched Processes")
    fig.tight_layout()

    savefig_or_show(fig, default_name=f"enriched_processes_top{n_genes}", show=show, save=save)
