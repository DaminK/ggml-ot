### Plot clustermap ###
import numpy as np

import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
import copy

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import seaborn as sns


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
    distances, colors, linkage, log_norm, method, title, dist_name, annotation, return_figure, **kwargs
):
    fig = sns.clustermap(
        distances,
        figsize=(5, 5),
        row_cluster=linkage is not None,
        col_cluster=linkage is not None,
        row_linkage=linkage,
        col_linkage=linkage,
        dendrogram_ratio=0.15,
        row_colors=colors,
        col_colors=colors,
        method=method,
        cmap=sns.cm.rocket_r,
        cbar_pos=(0.05, 0.1, 0.1, 0.02),
        cbar_kws={"orientation": "horizontal"},
        yticklabels=False,
        xticklabels=annotation,
        norm=log_norm,
        **kwargs,
    )

    fig.ax_heatmap.tick_params(right=False, bottom=bool(annotation))
    fig.ax_col_dendrogram.set_visible(False)
    fig.ax_cbar.set_title(dist_name, size="small")

    if title is not None:
        if return_figure:
            fig.ax_heatmap.set_title(title, pad=17)
        else:
            fig.figure.suptitle(title)

    return fig


def clustermap(
    distances,
    labels,
    hier_clustering=True,
    method="average",
    title="Clustermap",
    dist_name="OT Distance",
    log=False,
    save_path=None,
    cmap="Set2",
    hue_order=None,
    annotation=False,
    return_figure=False,
    **kwargs,
):
    """Plots hierarchically-clustered heatmap with patient annotations.

    :param dists: distance matrix of shape (n_samples, n_samples)
    :type dists: array-like
    :param labels: list of labels of each sample
    :type labels: array-like
    :param hier_clustering: whether to perform hierarchical clustering or not, defaults to True
    :type hier_clustering: bool, optional
    :param method: linkage method to use for hierarchical clustering, defaults to "average"
    :type method: str, optional
    :param title: title of the plot, defaults to None
    :type title: str, optional
    :param dist_name: name of the distance measure for the title of the colorbar, defaults to ""
    :type dist_name: str, optional
    :param log: whether to apply a logarithmic scaling to the distance matrix, defaults to False
    :type log: bool, optional
    :param save_path: file path to save generated plot (not saved if None), defaults to None
    :type save_path: str, optional
    :param cmap: color palette for clustermap, defaults to "tab20"
    :type cmap: str, optional
    :param hue_order: custom ordering of class labels for color mapping, defaults to None
    :type hue_order: array-like, optional
    :param annotation: whether to display sample labels on x-axis, defaults to False
    :type annotation: bool, optional
    :return: linkage matrix from hierarchical or None if clustering = True
    :rtype: numpy.ndarray or None
    """
    # creating list of colors for conds
    colors = _get_color_mapping(copy.deepcopy(labels), cmap, hue_order)

    distances_copy = copy.deepcopy(distances)
    # TODO figure out why the original distance and/or label are changed as a side effect in this function

    # compute hierarchical clustering if cluster == True
    if hier_clustering:
        distances_copy[np.eye(len(distances_copy), dtype=bool)] = 0
        linkage = hc.linkage(sp.distance.squareform(distances_copy), method=method, optimal_ordering=True)
    else:
        linkage = None

    # apply log scaling if log = True (useful when data spans a broad range of values)
    norm = None
    if log:
        norm = LogNorm()
        distances_copy[distances_copy <= 0] = np.min(distances_copy[distances_copy > 0])

    # create clustermap
    fig = _plot_clustermap(distances_copy, colors, linkage, norm, method, title, dist_name, annotation, return_figure)

    # save plot if desired
    if save_path is not None:
        fig.figure.savefig(save_path)

    if not return_figure:
        plt.show()
        return linkage
    else:
        return fig
