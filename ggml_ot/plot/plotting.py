"""High-level composite plotting functions for ggml_ot.

This module contains the main user-facing wrappers (e.g.
``clustermap_embedding``, ``distribution``) and delegates to
the specialised sub-modules for the heavy lifting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import umap
from sklearn.decomposition import PCA

from ggml_ot.plot.clustermap import clustermap
from ggml_ot.plot.embedding import emb
from ggml_ot.plot._utils import savefig_or_show


# -------------------------------------------------------------------
# Clustermap + Embedding composite
# -------------------------------------------------------------------


def clustermap_embedding(
    distances,
    labels,
    *,
    plot="clustermap_embedding",
    title=None,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs,
):
    """Plot a clustermap, an embedding, or both side-by-side.

    Wraps :func:`ggml_ot.pl.clustermap` and
    :func:`ggml_ot.pl.embedding`.

    Parameters
    ----------
    distances : numpy.ndarray
        Precomputed pairwise distance matrix.
    labels : list
        Class label for each sample.
    plot : str, default ``"clustermap_embedding"``
        Which plot(s) to draw.
        One of ``"clustermap"``, ``"embedding"``,
        ``"clustermap_embedding"``.
    title : str or None, default None
        Overall figure title.
    show : bool or None, default None
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save : str, bool, or None, default None
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves to ``settings.figdir/clustermap_embedding.<figformat>``;
        a *str* overrides the filename.
    **kwargs
        Forwarded to :func:`clustermap` and :func:`emb`.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure.
    """
    if plot not in {"clustermap", "embedding", "clustermap_embedding"}:
        raise ValueError("`plot` must be one of {'clustermap', 'embedding', 'clustermap_embedding'}.")

    ax2 = None
    fig = None

    if plot in {"clustermap", "clustermap_embedding"}:
        g = clustermap(distances, labels, show=False, save=False)

        if plot == "clustermap_embedding":
            g.gs.update(left=0.05, right=0.55)
            gs2 = mpl.gridspec.GridSpec(1, 1, left=0.57, top=0.83)
            ax2 = g.figure.add_subplot(gs2[0])
            g.figure.set_size_inches(10, 5)
        fig = g.figure

    if plot in {"embedding", "clustermap_embedding"}:
        emb(distances, labels, ax=ax2, show=False, save=False, **kwargs)
        if fig is None:
            fig = plt.gcf()

    if title is not None:
        plt.suptitle(title)

    savefig_or_show(fig, default_name="clustermap_embedding", show=show, save=save)

    return fig


# -------------------------------------------------------------------
# Distribution scatter plot
# -------------------------------------------------------------------


def distribution(
    dataset=None,
    distributions=None,
    labels=None,
    *,
    projection=lambda x: x,
    title="Distributions",
    legend=True,
    dim_red="umap",
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Visualise high-dimensional distributions in 2-D.

    Provide either *dataset* or both *distributions* and *labels*.
    Samples are projected via PCA or UMAP and displayed as a scatter
    plot, coloured by class.

    Parameters
    ----------
    dataset : TripletDataset or AnnData_TripletDataset, optional
        Dataset containing supports and labels.
    distributions : array-like or None, default None
        Distributions of shape ``(n_distributions, n_points, n_features)``.
    labels : array-like or None, default None
        Class label per distribution (length ``n_distributions``).
    projection : callable, default identity
        Pre-processing transform applied before dimensionality reduction.
    title : str, default ``"Distributions"``
        Plot title.
    legend : bool, default True
        Whether to show the legend.
    dim_red : str, default ``"umap"``
        Reduction method — ``"umap"`` or ``"pca"``.
    show : bool or None, default None
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save : str, bool, or None, default None
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves to ``settings.figdir/distribution.<figformat>``;
        a *str* overrides the filename.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the scatter plot.
    """
    distributions, labels = _extract_data(dataset, distributions, labels)

    offset = distributions.shape[0] / len(np.unique(labels)) if distributions.shape[0] > 5 else distributions.shape[0]

    reducer = _get_reducer(distributions, dim_red, projection)
    df_projected = _build_df(distributions, labels, projection, reducer, offset)
    ax = _plot_distributions(df_projected, title, legend)

    savefig_or_show(ax, default_name="distribution", show=show, save=save)
    return ax


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _extract_data(dataset, distributions, labels):
    if dataset is not None:
        return np.array(dataset.supports), np.array(dataset.distribution_labels)
    if distributions is None or labels is None:
        raise ValueError("Either dataset or (distributions, labels) must be provided.")
    if type(distributions) is not np.ndarray:
        distributions = np.array(distributions)
    return distributions, labels


def _get_reducer(distributions, dim_red, projection):
    dim = distributions.shape[-1]
    if dim <= 2:
        return None
    flat_data = projection(distributions.reshape(-1, dim))
    if dim_red == "umap":
        reducer = umap.UMAP()
    elif dim_red == "pca":
        reducer = PCA(n_components=2, svd_solver="full")
    else:
        raise ValueError(f"Unsupported dim_red: {dim_red}")
    reducer.fit_transform(flat_data)
    return reducer


def _build_df(distributions, labels, projection, reducer, offset):
    dfs = []
    for i, (dist, label) in enumerate(zip(distributions, labels)):
        projected = projection(dist)
        if reducer is not None:
            projected = reducer.transform(projected)
        dfs.append(
            pd.DataFrame(
                {
                    "x": projected[:, 0],
                    "y": projected[:, 1],
                    "class": str(label),
                    "dist": i % offset,
                }
            )
        )
    return pd.concat(dfs, axis=0)


def _plot_distributions(df, title, legend):
    """Create a scatter plot and return the Axes (does not call show)."""
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(df, x="x", y="y", hue="class", style="dist", alpha=0.5)
    if legend:
        sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))
    else:
        ax.get_legend().remove()
    ax.set_title(title)
    return ax
