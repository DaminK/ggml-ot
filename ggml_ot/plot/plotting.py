import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import umap
import pandas as pd

from sklearn.decomposition import PCA

from ggml_ot.plot.clustermap import clustermap
from ggml_ot.plot.embedding import emb


def clustermap_embedding(distances, labels, plot="clustermap_embedding", title=None, **kwargs):
    """Plots clustermap and embedding

    Wraps :meth:`ggml_ot.pl.clustermap` and :meth:`ggml_ot.pl.embedding`.

    Parameters
    ----------
    distances : np.ndarray
        precomputed pairwise distance matrix, see :meth:`ggml_ot.data.TripletDataset.compute_OT` or :meth:`ggml_ot.distances.compute_OT`.
    labels : list
        List of distribution labels
    plot : str, optional
        Type of plot to create, by default "clustermap_embedding"
        Options are "clustermap", "embedding", "clustermap_embedding"
    title : str, optional
        Title of the figure, by default None
    **kwargs :
        Additional arguments passed to :meth:`ggml_ot.pl.clustermap` and :meth:`ggml_ot.pl.embedding`.
    """

    if "clustermap" in plot:
        g = clustermap(distances, labels, return_figure=True)  # TODO Make **kwargs consistent with plot_emb

        # As Clustermap is a figure-level plotting function, we cannot plot it into an axis. We therefore add an axis to the clustermap figure.
        if plot == "clustermap_embedding":
            g.gs.update(left=0.05, right=0.55)  # TODO Fix, somehow this reverts any changes to the ax_cbar axis
            gs2 = mpl.gridspec.GridSpec(1, 1, left=0.57, top=0.83)
            ax2 = g.figure.add_subplot(gs2[0])
            g.figure.set_size_inches(10, 5)
    else:
        ax2 = None

    if "embedding" in plot:
        # g.figure.ax_cbar #See TODO above, this has to be manually moved again
        emb(distances, labels, ax=ax2, **kwargs)

    if title is not None:
        plt.suptitle(title)

    plt.show()


### Plot distributions ###


def _extract_data(dataset, distributions, labels):
    if dataset is not None:
        return np.array(dataset.supports), np.array(dataset.distribution_labels)
    if distributions is None or labels is None:
        raise ValueError("Either dataset or (distributions, labels) must be provided.")
    # convert input distributions to numpy array
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
        )  # TODO: correct offset variable
    return pd.concat(dfs, axis=0)


def _plot_distributions(df, title, legend):
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(df, x="x", y="y", hue="class", style="dist", alpha=0.5)
    if legend:
        sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))
    else:
        ax.get_legend().remove()
    ax.set_title(title)
    plt.show()


def distribution(
    dataset=None,
    distributions=None,
    labels=None,
    projection=lambda x: x,
    title="Distributions",
    legend=True,
    dim_red="umap",
):
    # TODO this plotting function is still a mess
    """Visualizes high-dimensional distributions in 2D using optional PCS projection. Either provide a dataset or distributions and labels.
    The distributions are plotted as a scatter plot where the classes are distinguishable by color
    and the distributions by shape.

    :param dataset: TripletDataset object containing the data, defaults to None
    :type dataset: :class:`ggml_ot.data.TripletDataset` or :class:`ggml_ot.data.AnnData_TripletDataset`, optional
    :param distributions: distributions to plot of shape (num_distributions, num_points, num_features), defaults to None
    :type distributions: array-like
    :param labels: labels corresponding to distributions of shape (num_distributions), defaults to None
    :type labels: array-like
    :param projection: transformation to apply to distributions before plotting, defaults to lambdax:x
    :type projection: callable, optional
    :param title: title of the plot, defaults to "Distributions"
    :type title: str, optional
    :param legend: whether to show legend, defaults to True
    :type legend: bool, optional
    """

    distributions, labels = _extract_data(dataset, distributions, labels)

    offset = distributions.shape[0] / len(np.unique(labels)) if distributions.shape[0] > 5 else distributions.shape[0]

    reducer = _get_reducer(distributions, dim_red, projection)
    # apply projection and PCA to projected distributions
    # create x,y coordinates and class labels for each data point
    df_projected = _build_df(distributions, labels, projection, reducer, offset)
    # visualize in scatter plot
    _plot_distributions(df_projected, title, legend)


### Plot ellipses ###


def ellipses(covariances, ax=None, title="Ellipses"):
    """Visualizes ellipses representing the covariance matrix.

    :param covariances: list of 2D covariance matrices or a single 2D covariance matrix
    :type covariances: array-like
    :param ax: axes object on which to plot the ellipses, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    :param title: title of the plot, defaults to "Ellipses"
    :type title: str, optional
    :return: axes containing the plotted ellipses
    :rtype: matplotlib.axes.Axes
    """

    # if no axes is provided, create one
    if ax is None:
        print("Create fig")
        _, ax = plt.subplots(ncols=len(covariances), figsize=(3 * len(covariances), 3))

    max = 0
    covariances = np.asarray(covariances)
    # make a list if only one covariance matrix is given
    if covariances.ndim == 2:
        covariances = [covariances]

    colors = sns.color_palette("Set2", len(covariances))

    # compute and plot ellipses
    for i, covariance in enumerate(covariances):
        # normalize matrix
        covariance = covariance / np.max(covariance)

        # compute eigenvalues and eigenvectors and angle of ellipse
        v, w = np.linalg.eigh(covariance)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)

        # create ellipse
        ell = mpl.patches.Ellipse((0, 0), v[0], v[1], angle=180 + angle, color=colors[i])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal")
        max = np.max([max, v[0], v[1]])

    ax.set_xlim([-max, max])
    ax.set_ylim([-max, max])

    ax.set_xticks(np.arange(-max, max, dtype=int))
    ax.set_yticks(np.arange(-max, max, dtype=int))

    ax.set_title(title)

    return ax


### Plot heatmap ###


def heatmap(
    results,
    labels="auto",
    xlabels=None,
    ylabels=None,
    ax=None,
    title="Pairwise distances",
):
    """Visualizes a 2D matrix as a heatmap.
    It represents the values of an input matrix by colors.

    :param results: data to be represented as a heatmap
    :type results: array-like
    :param labels: labels of the data, defaults to 'auto'
    :type labels: “auto”, bool, array-like, or int, optional
    :param xlabels: labels of the x-axis, defaults to None
    :type xlabels: “auto”, bool, array-like, or int, optional
    :param ylabels: labels of the y-axis, defaults to None
    :type ylabels: “auto”, bool, array-like, or int, optional
    :param ax: axes on which to draw the plot, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    :param title: title of the plot, defaults to None
    :type title: str, optional
    """
    # assign the labels
    xlabels = labels if xlabels is None else xlabels
    ylabels = labels if ylabels is None else ylabels
    # plot seaborn's heatmap with given parameters
    ax = sns.heatmap(
        results,
        xticklabels=xlabels,
        yticklabels=ylabels,
        ax=ax,
        square=results.shape[0] == results.shape[1],
    )
    ax.set_title(title)


### transport plan ###


def transport_plan(plan):
    plt.figure()
    sns.heatmap(plan)
    plt.show()
