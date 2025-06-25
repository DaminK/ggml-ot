import numpy as np

import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib as mpl

import umap
import pandas as pd
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.decomposition import PCA
from pydiffmap import diffusion_map
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
import warnings
from ggml_ot.distances import compute_OT


def plot_distribution_adata(
    adata, n_cells=1000, projection=lambda x: x, title="Distributions", legend=True
):
    """Visualizes high-dimensional distributions extracted from an Anndata object as an input in
    2D using optional PCS projection. The distributions are plotted as a scatter plot where the
    classes are distinguishable by color and the distributions by shape.

    :param adata: Anndata object or path to the object containing the data from which to extract and visualize
    the distributions
    :type adata: Anndata
    :param n_cells: number of cells to subsample per patient, set to 0 to keep all cells, defaults to 1000
    :type n_cells: int, optional
    :param projection: transformation to apply to distributions before plotting, defaults to lambdax:x
    :type projection: callable, optional
    :param title: title of the plot, defaults to "Distributions"
    :type title: str, optional
    :param legend: whether to show legend, defaults to True
    :type legend: bool, optional
    """
    from ggml_ot.data import scRNA_Dataset

    training_data = scRNA_Dataset(adata, n_cells=n_cells, filter_genes=False)
    a = training_data.distributions
    b = training_data.distributions_labels

    plot_distribution(
        distributions=a, labels=b, projection=projection, title=title, legend=legend
    )


def plot_distribution(
    distributions,
    labels,
    projection=lambda x: x,
    title="Distributions",
    legend=True,
    dim_red="umap",
):
    # TODO this plotting function is still a mess
    """Visualizes high-dimensional distributions in 2D using optional PCS projection.
    The distributions are plotted as a scatter plot where the classes are distinguishable by color
    and the distributions by shape.

    :param distributions: distributions to plot of shape (num_distributions, num_points, num_features)
    :type distributions: array-like
    :param labels: labels corresponding to distributions of shape (num_distributions)
    :type labels: array-like
    :param projection: transformation to apply to distributions before plotting, defaults to lambdax:x
    :type projection: callable, optional
    :param title: title of the plot, defaults to "Distributions"
    :type title: str, optional
    :param legend: whether to show legend, defaults to True
    :type legend: bool, optional
    """

    # convert input distributions to numpy array
    if type(distributions) is not np.ndarray:
        distributions = np.array(distributions)

    offset = (
        distributions.shape[0] / len(np.unique(labels))
        if distributions.shape[0] > 5
        else distributions.shape[0]
    )

    reducer = None
    dim = distributions.shape[-1]
    if dim > 2:
        if dim_red == "umap":
            flat_dists = projection(distributions.reshape(-1, dim))
            reducer = umap.UMAP()  # ,n_neighbors=10
            reducer.fit_transform(flat_dists)  # SJ
        elif dim_red == "pca":
            flat_dists = projection(distributions.reshape(-1, dim))
            reducer = PCA(n_components=2, svd_solver="full")
            reducer.fit_transform(flat_dists)

    # apply projection and PCA to projected distributions
    # create x,y coordinates and class labels for each data point
    i = 0
    dfList_projected = []
    for dist, label in zip(distributions, labels):
        stacked_projected = projection(dist)
        if reducer is not None:
            stacked_projected = reducer.transform(stacked_projected)
        dfList_projected.append(
            pd.DataFrame(
                {
                    "x": stacked_projected[:, 0],
                    "y": stacked_projected[:, 1],
                    "class": str(label),  # l.item()
                    "dist": i % offset,
                }
            )
        )  # TODO: correct offset variable
        i += 1

    # visualize in scatter plot
    df_projected = pd.concat(dfList_projected, axis=0)
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(
        df_projected, x="x", y="y", hue="class", style="dist", alpha=0.5
    )
    if legend:
        sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))
    else:
        ax.get_legend().remove()
    ax.set_title(title)
    plt.show()


def plot_ellipses(covariances, ax=None, title="Ellipses"):
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
        ell = mpl.patches.Ellipse(
            (0, 0), v[0], v[1], angle=180 + angle, color=colors[i]
        )
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


def plot_heatmap(
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


def plot_transport_plan(plan):
    plt.figure()
    sns.heatmap(plan)
    plt.show()


def plot_emb_adata(
    adata,
    precomputed_distances=None,
    ground_metric=None,
    n_cells=1000,
    method="umap",
    precomputed_emb=None,
    colors=None,
    symbols=None,
    ax=None,
    cluster_ID=None,
    title="Embedding",
    cmap="tab20",
    save_path=None,
    verbose=True,
    legend="Top",
    s=15,
    hue_order=None,
    annotation=None,
    linewidth=0.02,
    annotation_image_path=None,
):
    """Visualize the embedding of a distance matrix extracted from an Anndata object as an input
    using various reduction methods in form of a scatter plot.

    :param adata: Anndata object or path to the object containing the data from which to extract and visualize the distributions
    :type adata: Anndata
    :param precomputed_distances: precomputed distances to use as ground metric, defaults to None
    :type precomputed_distances: array-like, optional
    :param ground_metric: ground metric to use, defaults to None
    :type ground_metric: "euclidean", "cosine", "cityblock", optional
    :param n_cells: number of cells to subsample per patient, set to 0 to keep all cells, defaults to 1000
    :type n_cells: int, optional
    :param method: dimensionality reduction method, defaults to 'umap'
    :type method: "umap", "tsne", "diffusion", "fast_diffusion", "mds", "phate", optional
    :param precomputed_emb: precomputed embeddings to plot of shape (n_samples, 2), defaults to None
    :type precomputed_emb: array-like, optional
    :param colors: list of class labels to use for coloring the points, defaults to None
    :type colors: array-like, optional
    :param symbols: list of labels to use for marker styles, defaults to None
    :type symbols: array-like, optional
    :param ax: axes on which to draw the embedding, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    :param cluster_ID: boolean array indicating whether a point is a centroid/ medoid/ representative point of a cluster or not, defaults to None
    :type cluster_ID: array-like of bool, optional
    :param title: title of the plot, defaults to None
    :type title: str, optional
    :param cmap: colormap used for coloring the points, defaults to "tab20"
    :type cmap: str, array-like, dict or matplotlib.colors.Colormap, optional
    :param save_path: file path to save generated plot (not saved if None), defaults to None
    :type save_path: str, optional
    :param verbose: display title if True, defaults to True
    :type verbose: bool, optional
    :param legend: defines where to place the legend, defaults to 'Top'
    :type legend: "Top", "Side", optional
    :param s: marker size used in the plot, defaults to 15
    :type s: int, optional
    :param hue_order: order in which class labels are presented in legend, defaults to None
    :type hue_order: array-like of str, optional
    :param annotation: text to display on each point, defaults to None
    :type annotation: array-like of str, optional
    :param linewidth: linewidth of marker edges, defaults to 0.02
    :type linewidth: float, optional
    :param annotation_image_path: list of image paths to overlay on plot, defaults to None
    :type annotation_image_path: array-like of str, optional
    :return: 2D embedding for plotting
    :rtype: numpy.ndarray
    """
    from ggml_ot.data import scRNA_Dataset

    training_data = scRNA_Dataset(adata, n_cells=n_cells, filter_genes=False)
    symbols = [i % 10 for i in range(len(training_data.distributions))]
    colors = training_data.disease_labels

    if precomputed_distances is None and ground_metric is None:
        w = adata.uns["W_ggml"]
    else:
        w = None

    if precomputed_emb is None:
        dists = compute_OT(
            training_data.distributions,
            precomputed_distances=precomputed_distances,
            ground_metric=ground_metric,
            w=w,
        )
    else:
        dists = None

    plot_emb(
        dists=dists,
        method=method,
        precomputed_emb=precomputed_emb,
        colors=colors,
        symbols=symbols,
        ax=ax,
        cluster_ID=cluster_ID,
        title=title,
        cmap=cmap,
        save_path=save_path,
        verbose=verbose,
        legend=legend,
        s=s,
        hue_order=hue_order,
        annotation=annotation,
        linewidth=linewidth,
        annotation_image_path=annotation_image_path,
    )


def plot_emb(
    dists,
    method="umap",
    precomputed_emb=None,
    colors=None,
    symbols=None,
    ax=None,
    cluster_ID=None,
    title="Embedding",
    cmap="tab20",
    save_path=None,
    verbose=True,
    legend="Top",
    s=15,
    hue_order=None,
    annotation=None,
    linewidth=0.02,
    annotation_image_path=None,
):
    """Visualizes the embedding of a distance matrix using various reduction methods in form of a scatter plot.

    :param dists: distance matrix to plot the embeddings from of shape (n_samples, n_samples)
    :type dists: array-like
    :param method: dimensionality reduction method, defaults to 'umap'
    :type method: "umap", "tsne", "diffusion", "fast_diffusion", "mds", "phate", optional
    :param precomputed_emb: precomputed embeddings to plot of shape (n_samples, 2), defaults to None
    :type precomputed_emb: array-like, optional
    :param colors: list of class labels to use for coloring the points, defaults to None
    :type colors: array-like, optional
    :param symbols: list of labels to use for marker styles, defaults to None
    :type symbols: array-like, optional
    :param ax: axes on which to draw the embedding, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    :param cluster_ID: boolean array indicating whether a point is a centroid/ medoid/ representative point of a cluster or not, defaults to None
    :type cluster_ID: array-like of bool, optional
    :param title: title of the plot, defaults to None
    :type title: str, optional
    :param cmap: colormap used for coloring the points, defaults to "tab20"
    :type cmap: str, array-like, dict or matplotlib.colors.Colormap, optional
    :param save_path: file path to save generated plot (not saved if None), defaults to None
    :type save_path: str, optional
    :param verbose: display title if True, defaults to True
    :type verbose: bool, optional
    :param legend: defines where to place the legend, defaults to 'Top'
    :type legend: "Top", "Side", optional
    :param s: marker size used in the plot, defaults to 15
    :type s: int, optional
    :param hue_order: order in which class labels are presented in legend, defaults to None
    :type hue_order: array-like of str, optional
    :param annotation: text to display on each point, defaults to None
    :type annotation: array-like of str, optional
    :param linewidth: linewidth of marker edges, defaults to 0.02
    :type linewidth: float, optional
    :param annotation_image_path: list of image paths to overlay on plot, defaults to None
    :type annotation_image_path: array-like of str, optional
    :return: 2D embedding for plotting
    :rtype: numpy.ndarray
    """

    # compute the embedding with the provided method if no precomputed embedding is given
    if precomputed_emb is None:
        # use UMAP
        if method == "umap":
            with warnings.catch_warnings():
                # unfortunatly UMAP throws a warning that transformations of data points is not possible using precomputed metrics.
                # We do not think that this is a relveant information to users and the warning can not be masked through parameters,
                # so we catch it manually here.
                warnings.simplefilter("ignore", category=UserWarning)
                reducer = umap.UMAP(metric="precomputed")  # ,n_neighbors=10
                emb = reducer.fit_transform(dists)

        # use t-SNE
        elif method == "tsne":
            emb = TSNE(
                n_components=2,
                metric="precomputed",
                learning_rate="auto",
                init="random",
                perplexity=3,
            ).fit_transform(dists)

        # use diffusion map
        elif method == "diffusion":
            mydmap = diffusion_map.DiffusionMap.from_sklearn(
                n_evecs=2, epsilon=0.1, alpha=0.5, k=64
            )
            emb = mydmap.fit_transform(dists / dists.max())
            emb = emb[:, [0, 1]]

        # use fast diffusion map
        elif method == "fast_diffusion":
            maxim = np.max(dists)
            epsilon = maxim * 0.7
            print(epsilon)
            scaled_matrix = dists**2 / epsilon
            # take negative exponent of scaled matrix to create Isotropic kernel
            kernel = np.exp(-scaled_matrix)
            D_inv = np.diag(1 / kernel.sum(1))
            diff = np.dot(D_inv, kernel)
            eigenvals, eigenvectors = np.linalg.eig(diff)
            sort_idx = np.argsort(eigenvals)[::-1]
            eigenvectors = eigenvectors[sort_idx]
            emb = np.transpose(eigenvectors[[0, 1], :])

        # use multidimensional scaling
        elif method == "mds":
            mds = manifold.MDS(
                n_components=2, dissimilarity="precomputed", normalized_stress="auto"
            )
            emb = mds.fit_transform(dists)

        # # use phate
        # elif method == "phate":
        #     phate_op = phate.PHATE(knn_dist="precomputed_distance",verbose=0)
        #     emb = phate_op.fit_transform(dists)

    # use given embedding if provided
    else:
        emb = precomputed_emb

    # create dataframe with data points and metadata
    df_embed = pd.DataFrame(emb, columns=["x", "y"])
    df_embed["Classes"] = colors
    df_embed["Condition"] = symbols
    df_embed["annotation"] = annotation
    df_embed["Type"] = (
        None
        if cluster_ID is None
        else ["Cluster" if is_cluster else "Trial" for is_cluster in cluster_ID]
    )
    type_to_size = {
        "Cluster": 50,
        "Trial": 7,
        None: 3 if annotation_image_path is None else 200,
    }

    if ax is None:
        _, ax = (
            plt.subplots(figsize=(5, 5))
            if annotation_image_path is None
            else plt.subplots(figsize=(30, 7))
        )

    # create scatter plot
    ax = sns.scatterplot(
        df_embed,
        x="x",
        y="y",
        edgecolor="white",
        alpha=1.0,
        s=s,
        linewidth=linewidth,
        hue="Classes" if colors is not None else None,
        style="Condition" if symbols is not None else None,
        size="Type" if cluster_ID is not None else None,
        sizes=type_to_size if cluster_ID is not None else None,
        ax=ax,
        palette=cmap,
        legend=False if not legend else "auto",
        hue_order=hue_order,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.subplots_adjust(top=10 / 12)

    # display title if desired
    if verbose:
        ax.set_title(title)  # plt.gca().set_aspect('equal', 'datalim')

    # place legend where/ if desired
    if legend == "Top":
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.0, 1.075, 0.9, 0.10)
            if len(np.unique(colors)) > 5
            else (0.0, 1.05, 0.9, 0.075),
            prop=dict(weight="bold"),
            handletextpad=0.1,
            frameon=False,
            shadow=False,
            ncol=4 if len(np.unique(colors)) > 5 else 5,
            mode="expand",
        )
    elif legend == "Side":
        plt.legend(frameon=False)
        sns.move_legend(ax, "right", bbox_to_anchor=(1.5, 0.5))

    # load, crop and resize images
    def crop(im, w, h):
        width, height = im.size  # Get dimensions
        left = (1 - w) / 2 * width
        top = (1 - h) / 2 * height
        right = (w + 1) / 2 * width
        bottom = (h + 1) / 2 * height
        return im.crop((left, top, right, bottom))

    def getImage(path, zoom=0.5, w=0.6, h=0.72):  # 0.04
        return OffsetImage(np.asarray(crop(Image.open(path), w, h)), zoom=zoom)

    if annotation_image_path is not None:
        if "histo" in annotation_image_path[0]:
            scaling = 0.025
            width = 0.8
            height = 0.8
        elif "niche" in annotation_image_path[0]:
            scaling = 0.45
            width = 0.6
            height = 0.75
        else:
            # Probably cell distr
            scaling = 0.4
            width = 0.8
            height = 0.8

        for p in range(0, df_embed.shape[0]):
            ab = AnnotationBbox(
                getImage(annotation_image_path[p], zoom=scaling, w=width, h=height),
                (df_embed.x[p], df_embed.y[p]),
                xycoords="data",
                boxcoords="offset points",
                frameon=False,
                box_alignment=(0, 0),
                pad=0.1,
            )
            ax.add_artist(ab)

    # add text labels
    # add annotations one by one with a loop, credit https://python-graph-gallery.com/46-add-text-annotation-on-scatterplot/
    if annotation is not None:
        for p in range(0, df_embed.shape[0]):
            ax.text(
                df_embed.x[p],
                df_embed.y[p],
                df_embed.annotation[p],
                horizontalalignment="left",
                size="x-small",
                color="black",
            )

    # save plot if desired
    if save_path is not None:
        print(save_path)
        ax.figure.savefig(save_path)

    return emb


def plot_clustermap_adata(
    adata,
    n_cells=1000,
    hier_clustering=True,
    method="average",
    title=None,
    dist_name="",
    log=False,
    save_path=None,
    cmap="tab20",
    hue_order=None,
    annotation=False,
):
    """Visualize a given distance matrix as a hierarchically-clustered heatmap (clustermap) extracted from an
    Anndata object as an input.

    :param adata: Anndata object or path to the object containing the data from which to extract and visualize the distributions
    :type adata: Anndata
    :param n_cells: number of cells to subsample per patient, set to 0 to keep all cells, defaults to 1000
    :type n_cells: int, optional
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
    from ggml_ot.data import scRNA_Dataset

    training_data = scRNA_Dataset(adata, n_cells=n_cells, filter_genes=False)
    w = adata.uns["w_theta"]
    dists = compute_OT(training_data.distributions, w=w)

    plot_clustermap(
        dists=dists,
        labels=training_data.distributions_labels,
        hier_clustering=hier_clustering,
        method=method,
        title=title,
        dist_name=dist_name,
        log=log,
        save_path=save_path,
        cmap=cmap,
        hue_order=hue_order,
        annotation=annotation,
    )


def plot_clustermap(
    dists,
    labels,
    hier_clustering=True,
    method="average",
    title=None,
    dist_name="",
    log=False,
    save_path=None,
    cmap="tab20",
    hue_order=None,
    annotation=False,
):
    """Visualize a given distance matrix as a hierarchically-clustered heatmap (clustermap).

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
    unique_inds = np.unique(labels, return_index=True)[1]
    unique_labels = (
        np.asarray([labels[index] for index in sorted(unique_inds)]).tolist()
        if hue_order is None
        else hue_order
    )

    # define color palette
    if isinstance(cmap, str):
        cmap = sns.color_palette(palette=cmap, n_colors=len(unique_labels))
        colors = [cmap[unique_labels.index(label)] for label in labels]
    else:
        colors = [cmap[label] for label in labels]

    dist = np.copy(dists)

    # compute hierarchical clustering if cluster == True
    if hier_clustering:
        dist[np.eye(len(dist), dtype=bool)] = 0
        linkage = hc.linkage(
            sp.distance.squareform(dist), method=method, optimal_ordering=True
        )
    else:
        linkage = None

    # apply log scaling if log = True (useful when data spans a broad range of values)
    norm = None
    if log:
        norm = LogNorm()
        dist[dist <= 0] = np.min(dist[dist > 0])

    # create clustermap
    fig = sns.clustermap(
        dist,
        figsize=(5, 5),
        row_cluster=hier_clustering,
        col_cluster=hier_clustering,
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
        norm=norm,
    )

    fig.ax_heatmap.tick_params(
        right=False, bottom=False if annotation is False else True
    )  # sounds really stupid but annotation might be non-boolean
    fig.ax_col_dendrogram.set_visible(False)
    fig.ax_cbar.set_title(dist_name)

    if title is not None:
        fig.fig.suptitle(title)
    plt.show()

    # save plot if desired
    if save_path is not None:
        fig.figure.savefig(save_path)

    return linkage
