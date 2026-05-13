"""Hierarchically-clustered heatmap with sample annotations."""

from __future__ import annotations

import copy

import numpy as np
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc

from matplotlib.colors import LogNorm
import seaborn as sns
import torch

from ggml_ot.plot._utils import savefig_or_show


def clustermap(
    distances,
    labels,
    *,
    hier_clustering=True,
    linkage="complete",
    title="Clustermap",
    dist_name="OT Distance",
    log=False,
    cmap="Set2",
    hue_order=None,
    annotation=False,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs,
):
    """Plot a hierarchically-clustered heatmap with sample annotations.

    Parameters
    ----------
    distances : array-like
        Distance matrix of shape ``(n_samples, n_samples)``.
        If a tensor is provided, it will be converted to a numpy array.
    labels : array-like
        Class label for each sample (used for row/column colour bar).
    hier_clustering : bool, default True
        Whether to perform hierarchical clustering.
    linkage : str, default ``"complete"``
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`.
    title : str or None, default ``"Clustermap"``
        Title displayed above the heatmap.
    dist_name : str, default ``"OT Distance"``
        Label for the colour-bar.
    log : bool, default False
        Apply logarithmic colour scaling.
    cmap : str or dict, default ``"Set2"``
        Colour palette for sample annotations.
    hue_order : array-like or None, default None
        Custom ordering of class labels for colour mapping.
    annotation : bool, default False
        Display sample labels on the x-axis.
    show : bool or None, default None
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save : str, bool, or None, default None
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves to ``settings.figdir/clustermap.<figformat>``;
        a *str* overrides the filename.
    **kwargs
        Forwarded to :func:`seaborn.clustermap`.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        The cluster-grid object (always returned).
    """
    colors = _get_color_mapping(copy.deepcopy(labels), cmap, hue_order)

    # Ensure distances is a numpy array (handle Tensors)
    if isinstance(distances, torch.Tensor):
        distances = distances.detach().cpu().numpy()
    elif not isinstance(distances, np.ndarray):
        distances = np.asarray(distances)

    distances_copy = copy.deepcopy(distances)

    # Compute hierarchical clustering
    if hier_clustering:
        distances_copy[np.eye(len(distances_copy), dtype=bool)] = 0
        Z = hc.linkage(
            sp.distance.squareform(distances_copy),
            method=linkage,
            optimal_ordering=True,
        )
    else:
        Z = None

    # Log-scale normalisation
    norm = None
    if log:
        norm = LogNorm()
        distances_copy[distances_copy <= 0] = np.min(distances_copy[distances_copy > 0])

    # Create clustermap
    grid = _plot_clustermap(
        distances_copy,
        colors,
        Z,
        norm,
        linkage,
        title,
        dist_name,
        annotation,
        **kwargs,
    )

    savefig_or_show(grid, default_name="clustermap", show=show, save=save)

    return grid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_color_mapping(labels, cmap, hue_order):
    unique_inds = np.unique(labels, return_index=True)[1]
    unique_labels = np.asarray([labels[i] for i in sorted(unique_inds)]).tolist() if hue_order is None else hue_order

    if isinstance(cmap, str):
        palette = sns.color_palette(palette=cmap, n_colors=len(unique_labels))
        colors = [palette[unique_labels.index(lbl)] for lbl in labels]
    else:
        colors = [cmap[lbl] for lbl in labels]
    return colors


def _plot_clustermap(
    distances,
    colors,
    Z,
    log_norm,
    linkage,
    title,
    dist_name,
    annotation,
    **kwargs,
):
    grid = sns.clustermap(
        distances,
        figsize=(5, 5),
        row_cluster=Z is not None,
        col_cluster=Z is not None,
        row_linkage=Z,
        col_linkage=Z,
        dendrogram_ratio=0.15,
        row_colors=colors,
        col_colors=colors,
        method=linkage,
        cmap=sns.cm.rocket_r,
        cbar_pos=(0.05, 0.1, 0.1, 0.02),
        cbar_kws={"orientation": "horizontal"},
        yticklabels=False,
        xticklabels=annotation,
        norm=log_norm,
        **kwargs,
    )

    grid.ax_heatmap.tick_params(right=False, bottom=bool(annotation))
    grid.ax_col_dendrogram.set_visible(False)
    grid.ax_cbar.set_title(dist_name, size="small")

    if title is not None:
        grid.figure.suptitle(title)

    return grid
