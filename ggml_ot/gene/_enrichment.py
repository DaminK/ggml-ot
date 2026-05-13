import textwrap

import numpy as np
import gprofiler

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns

from ggml_ot.plot._utils import savefig_or_show


def _normalize_component_ids(adata, components):
    n_components = adata.varm["W_ggml"].shape[-1]
    if components is None:
        component_ids = np.arange(1, n_components + 1)
    elif isinstance(components, str):
        component_ids = [int(x) for x in components.split(",")]
    else:
        component_ids = components

    component_ids = np.asarray(component_ids, dtype=int)
    if component_ids.ndim != 1:
        raise ValueError("components must be a one-dimensional sequence of component IDs.")
    if np.any((component_ids < 1) | (component_ids > n_components)):
        raise IndexError(f"components must be between 1 and {n_components}.")
    return component_ids


def _format_neg_log10_pvalue_tick(value, _position):
    if value < 0 or not np.isclose(value, round(value)):
        return ""
    exponent = int(round(value))
    return rf"$10^{{-{exponent}}}$"


def top_ranked(
    adata, components=None, n_genes=None, gene_symbols: str | None = None, up_regulated=False, down_regulated=False
):
    """Return genes with the highest contributions to each GGML component.

    Parameters
    ----------
    adata
        Annotated data matrix.
    components
        One-based component IDs. For example, ``'1,2,3'`` means the first,
        second, and third GGML component. Defaults to all GGML components.
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
    component_indices = _normalize_component_ids(adata, components) - 1  # convert to zero-based indices

    gene_name = np.asarray(adata.var.index if gene_symbols is None else adata.var[gene_symbols])

    top_genes = np.zeros((len(component_indices), n_genes), dtype=object)
    for row, component_idx in enumerate(component_indices):
        gene_regulation = adata.varm["W_ggml"][:, component_idx]
        if down_regulated and not up_regulated:
            gene_regulation = -gene_regulation
        elif up_regulated == down_regulated:
            gene_regulation = np.abs(gene_regulation)

        top_genes[row, :] = gene_name[np.argsort(gene_regulation)[-n_genes:]]

    if down_regulated and not up_regulated:
        top_genes = np.flip(top_genes, axis=1)

    return top_genes


def enrichment(
    adata,
    n_genes=50,
    components=None,
    gene_symbols: str | None = None,
    up_regulated=False,
    down_regulated=False,
    ordered=False,
    alpha=0.05,
    organism="hsapiens",
    max_terms: int | None = None,
    log_axis=False,
    orient="v",
    wrap_width: int | None = 35,
    figsize: tuple[float, float] | None = None,
    font_size: float | None = None,
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
        One-based component IDs. For example, ``'1,2,3'`` means the first,
        second, and third GGML component. Defaults to all GGML components.
    gene_symbols
        Key for field in `.var` that stores gene symbols if you do not want to use `.var_names`.
    up_regulated
        If True, run enrichment on genes with the largest positive loadings.
    down_regulated
        If True, run enrichment on genes with the largest negative loadings. When both flags are equal,
        absolute loadings are used.
    ordered
        Whether the gene lists are ordered by importance
    alpha
        Threshold for significance in enrichment
    organism
        Organism ID for g:Profiler
    max_terms
        Maximum number of enriched terms shown per component. ``None``
        shows all enriched terms.
    log_axis
        Whether to plot ``-log10(p_value)`` instead of raw p-values.
    orient
        Whether to layout the plots horizontally (``"h"``) or vertically (``"v"``).
    wrap_width
        Maximum number of characters per line for term names. Long names are
        wrapped at word boundaries to fit within this width. ``None`` disables
        wrapping.
    figsize
        Figure size (width, height) in inches. ``None`` uses a default that
        scales with the number of components.
    font_size
        Base font size in points applied to ticks, labels, titles, and the
        alpha annotation. ``None`` keeps matplotlib defaults. Use this together
        with ``figsize`` to tune the text-to-plot ratio.
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
    component_ids = _normalize_component_ids(adata, components)

    top_genes = top_ranked(
        adata,
        component_ids,
        n_genes,
        gene_symbols,
        up_regulated=up_regulated,
        down_regulated=down_regulated,
    )

    # Gene Enrichment
    if orient == "h":
        layout = (1, len(top_genes))
        default_figsize = (5 * len(top_genes), 6)
        share_axis = {"sharey": True}
    elif orient == "v":
        layout = (len(top_genes), 1)
        default_figsize = (7, 5 * len(top_genes))
        share_axis = {"sharex": True}
    rc_params = {"font.size": font_size} if font_size is not None else {}
    with plt.rc_context(rc_params):
        fig, axs = plt.subplots(*layout, figsize=figsize or default_figsize, **share_axis)
        axs = np.atleast_1d(axs)

        enrichments = []
        plot_col = "neg_log10_p_value" if log_axis else "p_value"
        plot_label = "p-value"
        alpha_value = -np.log10(alpha) if log_axis else alpha
        axis_max = alpha_value if log_axis else alpha
        log_tick_formatter = FuncFormatter(_format_neg_log10_pvalue_tick)
        for c, (component_id, c_top_genes) in enumerate(zip(component_ids, top_genes)):
            gp = gprofiler.GProfiler(return_dataframe=True)
            enrich = gp.profile(
                query=list(c_top_genes),
                ordered=ordered,
                user_threshold=alpha,
                organism=organism,
            )
            enrich["neg_log10_p_value"] = -np.log10(enrich["p_value"])
            if max_terms is not None:
                enrich = enrich.head(max_terms)
            if wrap_width is not None:
                enrich = enrich.assign(
                    name=enrich["name"].astype(str).map(lambda s: textwrap.fill(s, width=wrap_width))
                )
            enrichments.append(enrich)
            if log_axis:
                axis_max = max(axis_max, enrich[plot_col].max())
        axis_max *= 1.05

        for c, (component_id, enrich) in enumerate(zip(component_ids, enrichments)):
            if orient == "h":
                sns.barplot(x="name", y=plot_col, data=enrich, ax=axs[c], **kwargs)
                axs[c].tick_params(axis="x", rotation=90)
                axs[c].set_ylim(0, axis_max)
                axs[c].set_ylabel(plot_label)
                if log_axis:
                    axs[c].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                    axs[c].yaxis.set_major_formatter(log_tick_formatter)
                if c != 0:
                    axs[c].get_yaxis().set_visible(False)
            elif orient == "v":
                sns.barplot(x=plot_col, y="name", data=enrich, ax=axs[c], **kwargs)
                axs[c].set_xlim(0, axis_max)
                axs[c].set_xlabel(plot_label)
                if log_axis:
                    axs[c].xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                    axs[c].xaxis.set_major_formatter(log_tick_formatter)
                if c != len(top_genes) - 1:
                    axs[c].get_xaxis().set_visible(False)

                axs[c].vlines(
                    x=alpha_value,
                    ymin=axs[c].get_ylim()[0],
                    ymax=axs[c].get_ylim()[1],
                    color="black",
                    label="axvline - full height",
                    linestyles="dashed",
                )
                if c == 0:
                    axs[c].text(
                        x=alpha_value,
                        y=axs[c].get_ylim()[1],
                        s=rf"$\alpha = {alpha}$",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                    )
            if len(component_ids) > 1:
                axs[c].set_title(f"W_GGML {component_id}")

        fig.suptitle("Enriched Processes")
        fig.tight_layout()

        savefig_or_show(fig, default_name=f"enriched_processes_top{n_genes}", show=show, save=save)
