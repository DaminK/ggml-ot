"""2-D embedding of a precomputed distance matrix."""

from __future__ import annotations

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

from ggml_ot.plot._utils import savefig_or_show


def emb(
    distances,
    labels,
    *,
    method="umap",
    precomputed_emb=None,
    symbols=None,
    ax=None,
    cluster_ID=None,
    title="Embedding",
    cmap="Set2",
    legend="auto",
    s=200,
    hue_order=None,
    annotation=None,
    linewidth=0.02,
    annotation_image_path=None,
    return_embedding: bool = False,
    show: bool | None = None,
    save: str | bool | None = None,
    **kwargs,
):
    """Plot a 2-D embedding of a precomputed distance matrix.

    Parameters
    ----------
    distances : array-like
        Distance matrix of shape ``(n_samples, n_samples)``.
    labels : array-like
        Class labels used for colouring the points.
    method : str, default ``"umap"``
        Dimensionality-reduction method.
        One of ``"umap"``, ``"tsne"``, ``"diffusion"``,
        ``"fast_diffusion"``, ``"mds"``.
    precomputed_emb : array-like or None, default None
        Precomputed 2-D coordinates of shape ``(n_samples, 2)``.  When
        provided, *method* is ignored.
    symbols : array-like or None, default None
        Labels used for marker styles.
    ax : matplotlib.axes.Axes or None, default None
        Axes to draw on.  A new figure is created when ``None``.
    cluster_ID : array-like of bool or None, default None
        Flags indicating centroid / medoid / representative points.
    title : str or None, default ``"Embedding"``
        Plot title.
    cmap : str, dict, or colormap, default ``"Set2"``
        Colour palette for the scatter plot.
    legend : str, default ``"auto"``
        Legend placement.  One of ``"auto"``, ``"Side"``/``"right"``,
        ``"left"``, ``"Top"``, ``"Bottom"``, ``"Inside"``, or ``False``.
    s : int, default 200
        Marker size.
    hue_order : array-like or None, default None
        Custom legend ordering of class labels.
    annotation : array-like of str or None, default None
        Text annotations displayed on each point.
    linewidth : float, default 0.02
        Edge width of scatter markers.
    annotation_image_path : array-like of str or None, default None
        Image paths to overlay on the plot.
    return_embedding : bool, default False
        If ``True``, return ``(ax, embedding)`` instead of just ``ax``.
    show : bool or None, default None
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save : str, bool, or None, default None
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves to ``settings.figdir/embedding.<figformat>``;
        a *str* overrides the filename.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the embedding plot (when
        ``return_embedding=False``).
    tuple[matplotlib.axes.Axes, numpy.ndarray]
        ``(ax, embedding)`` when ``return_embedding=True``.

    Notes
    -----
    Computes a 2-D layout from a precomputed distance matrix using the
    chosen dimensionality-reduction method and renders a scatter plot
    coloured by ``labels``.  This is the main function behind
    ``ggml_ot.pl.embedding`` and is used by the benchmark evaluation
    helpers to visualise OT distances between patients.

    Supported methods: ``"umap"`` (default, adaptive ``n_neighbors``),
    ``"tsne"``, ``"mds"``, ``"diffusion"``, ``"fast_diffusion"``.
    """
    embedding = precomputed_emb if precomputed_emb is not None else _compute_embedding(distances, method, labels=labels)
    df, type_to_size = _create_dataframe(embedding, labels, symbols, annotation, cluster_ID, annotation_image_path)

    # Determine legend handling
    position_legend = legend
    if position_legend in ["Side", "Top", "Bottom", "Inside"]:
        legend = "auto"

    # Create axes if not provided
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

    # Legend placement
    if position_legend in ["Side", "Top", "Bottom", "Inside"]:
        _setup_legend(ax, position_legend)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if position_legend == "Side" or position_legend == "right":
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    elif position_legend == "left":
        sns.move_legend(ax, "upper right", bbox_to_anchor=(-0.02, 1), frameon=False)
    elif position_legend == "Top":
        sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True)
    elif position_legend == "Bottom":
        sns.move_legend(ax, "upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)
    elif position_legend == "Inside":
        sns.move_legend(ax, "best", frameon=True)

    if title is not None:
        ax.set_title(title)

    # Image / text overlays
    _add_image_overlays(ax, df, annotation_image_path)
    if annotation is not None:
        _add_text_annotations(ax, df)

    savefig_or_show(ax, default_name="embedding", show=show, save=save)

    if return_embedding:
        return ax, embedding
    return ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_embedding(distances, method, labels=None):
    if method == "umap":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            n_neighbors = _estimate_n_neighbors(distances, labels)
            reducer = umap.UMAP(metric="precomputed", n_neighbors=n_neighbors, min_dist=0.5)
            return reducer.fit_transform(distances)

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


def _estimate_n_neighbors(distances, labels=None):
    """Pick a reasonable n_neighbors for UMAP on a precomputed distance matrix.

    Strategy (floor + class cap):
      1. Start with ~1/3 of the samples as the base.
      2. If class labels are provided, cap at the smallest class size
         so that tiny classes are not absorbed into larger ones.
      3. Always keep at least 5 to guarantee a connected UMAP graph
         (prevents isolated-island artefacts).
    """
    n_samples = distances.shape[0]
    # Base: roughly one-third of samples
    n_neighbors = max(5, n_samples // 3)

    if labels is not None:
        _, counts = np.unique(labels, return_counts=True)
        min_class_size = int(counts.min()) if counts.size else n_samples
        # Cap at smallest class size to protect small classes,
        # but never drop below 5 (graph connectivity floor)
        n_neighbors = max(5, min(n_neighbors, min_class_size))

    # Cannot exceed n-1 (UMAP hard constraint)
    return min(n_neighbors, n_samples - 1)


def _setup_legend(ax, position_legend):
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
