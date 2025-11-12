### Plot embedding ###
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

import umap
from sklearn.manifold import TSNE
from sklearn import manifold
from pydiffmap import diffusion_map


def emb(
    distances,
    labels,
    method="umap",
    precomputed_emb=None,
    symbols=None,
    ax=None,
    cluster_ID=None,
    title="Embedding",
    cmap="Set2",
    save_path=None,
    legend="auto",
    s=15,
    hue_order=None,
    annotation=None,
    linewidth=0.02,
    annotation_image_path=None,
    **kwargs,
):
    """Plots embedding of a distance matrix using various reduction methods.

    :param dists: distance matrix to plot the embeddings from of shape (n_samples, n_samples)
    :type dists: array-like
    :param labels: list of class labels to use for coloring the points, defaults to None
    :type labels: array-like
    :param method: dimensionality reduction method, defaults to 'umap'
    :type method: "umap", "tsne", "diffusion", "fast_diffusion", "mds", "phate", optional
    :param precomputed_emb: precomputed embeddings to plot of shape (n_samples, 2), defaults to None
    :type precomputed_emb: array-like, optional

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

    emb = precomputed_emb if precomputed_emb is not None else _compute_embedding(distances, method)
    df, type_to_size = _create_dataframe(emb, labels, symbols, annotation, cluster_ID, annotation_image_path)

    # Determine legend handling
    position_legend = legend
    if position_legend in ["Side", "Top", "Bottom", "Inside"]:
        legend = "auto"

    # Scatter plot
    # create axis if not provided
    if ax is None:
        figsize = (30, 7) if annotation_image_path is not None else (5, 5)
        _, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        df,
        x="x",
        y="y",
        edgecolor="white",
        alpha=1.0,
        s=s,
        linewidth=linewidth,
        hue="Classes" if labels is not None else None,
        style="Condition" if symbols is not None else None,
        size="Type" if cluster_ID is not None else None,
        sizes=type_to_size if cluster_ID is not None else None,
        ax=ax,
        palette=cmap,
        legend=legend,
        hue_order=hue_order,
    )

    # move legend after plotting
    if position_legend in ["Side", "Top", "Bottom", "Inside"]:
        _setup_legend(ax, position_legend)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # plt.subplots_adjust(top=10 / 12)

    if position_legend == "Side" or position_legend == "right":
        legend = True
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    elif position_legend == "left":
        legend = True
        sns.move_legend(ax, "upper right", bbox_to_anchor=(-0.02, 1), frameon=False)
    elif position_legend == "Top":
        legend = True
        sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True)
    elif position_legend == "Bottom":
        legend = True
        sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)
    elif position_legend == "Inside":
        legend = True
        sns.move_legend(ax, "best", frameon=True)

    # display title if desired
    if title is not None:
        ax.set_title(title)

    # add images or text annotations if desired
    _add_image_overlays(ax, df, annotation_image_path)
    if annotation is not None:
        _add_text_annotations(ax, df)

    # save if desired
    if save_path is not None:
        print(save_path)
        ax.figure.savefig(save_path)

    return emb


def _compute_embedding(distances, method):
    if method == "umap":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            n_neighbors = np.max([int(len(distances) / 4), 15])  # increase default n_neighbors=15 for large datasets
            reducer = umap.UMAP(metric="precomputed", n_neighbors=n_neighbors)
            return reducer.fit_transform(
                distances,
            )

    elif method == "tsne":
        return TSNE(
            n_components=2,
            metric="precomputed",
            learning_rate="auto",
            init="random",
            perplexity=3,
        ).fit_transform(distances)

    elif method == "diffusion":
        mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs=2, epsilon=0.1, alpha=0.5, k=64)
        emb = mydmap.fit_transform(distances / distances.max())
        return emb[:, [0, 1]]

    elif method == "fast_diffusion":
        maxim = np.max(distances)
        epsilon = maxim * 0.7
        scaled = distances**2 / epsilon
        kernel = np.exp(-scaled)
        D_inv = np.diag(1 / kernel.sum(1))
        diff = np.dot(D_inv, kernel)
        eigenvals, eigenvectors = np.linalg.eig(diff)
        sort_idx = np.argsort(eigenvals)[::-1]
        eigenvectors = eigenvectors[sort_idx]
        return np.transpose(eigenvectors[[0, 1], :])

    elif method == "mds":
        mds = manifold.MDS(n_components=2, dissimilarity="precomputed", normalized_stress="auto")
        return mds.fit_transform(distances)

    else:
        raise ValueError(f"Unknown embedding method: {method}")


def _create_dataframe(emb, colors, symbols, annotation, cluster_ID, annotation_image_path):
    df = pd.DataFrame(emb, columns=["x", "y"])
    df["Classes"] = colors
    df["Condition"] = symbols
    df["annotation"] = annotation
    df["Type"] = None if cluster_ID is None else ["Cluster" if is_cluster else "Trial" for is_cluster in cluster_ID]
    type_to_size = {
        "Cluster": 50,
        "Trial": 7,
        None: 3 if annotation_image_path is None else 200,
    }
    return df, type_to_size


def _setup_legend(ax, position_legend):
    # handle legend position
    if position_legend == "Side":
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), frameon=True)
    elif position_legend == "Top":
        sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True)
    elif position_legend == "Bottom":
        sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)
    elif position_legend == "Inside":
        sns.move_legend(ax, "best", frameon=True)


def _add_image_overlays(ax, df, paths):
    if paths is None:
        return

    # adjust image parameters based on context
    if "histo" in paths[0]:
        scaling, width, height = 0.025, 0.8, 0.8
    elif "niche" in paths[0]:
        scaling, width, height = 0.45, 0.6, 0.75
    else:
        scaling, width, height = 0.4, 0.8, 0.8

    def crop_image(im, w, h):
        width_px, height_px = im.size
        left = (1 - w) / 2 * width_px
        top = (1 - h) / 2 * height_px
        right = (w + 1) / 2 * width_px
        bottom = (h + 1) / 2 * height_px
        return im.crop((left, top, right, bottom))

    for i, row in df.iterrows():
        im = Image.open(paths[i])
        cropped = crop_image(im, width, height)
        img = OffsetImage(np.asarray(cropped), zoom=scaling)
        ab = AnnotationBbox(
            img,
            (row.x, row.y),
            xycoords="data",
            boxcoords="offset points",
            frameon=False,
            box_alignment=(0, 0),
            pad=0.1,
        )
        ax.add_artist(ab)


def _add_text_annotations(ax, df):
    for _, row in df.iterrows():
        if row["annotation"] is not None:
            ax.text(
                row.x,
                row.y,
                row.annotation,
                horizontalalignment="left",
                size="x-small",
                color="black",
            )
