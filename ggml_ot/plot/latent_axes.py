"""Latent axis gene-ranking plots.

Visualises the top-ranked genes per GGML latent axis, either from
a pre-computed ranking DataFrame or directly from an AnnData-backed
dataset.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from ggml_ot.plot._utils import savefig_or_show


def latent_axes(
    ranking_df: pd.DataFrame,
    *,
    n_genes: int = 20,
    axes: list[int] | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Plot top-ranked genes per latent axis.

    Parameters
    ----------
    ranking_df
        Long-format DataFrame from :func:`ggml_ot.gene.rank_latent_axes`
        with columns ``axis``, ``gene``, ``score``, ``rank``.
    n_genes
        Number of top genes to display per axis.
    axes
        Subset of axis indices to plot.  ``None`` plots all.
    show
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves under the default name into
        ``settings.figdir``.  A *str* is used as the filename.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    if axes is not None:
        ranking_df = ranking_df[ranking_df["axis"].isin(axes)]

    axis_ids = sorted(ranking_df["axis"].unique())
    n_axes = len(axis_ids)

    fig, axarr = plt.subplots(1, n_axes, figsize=(4 * n_axes, 0.35 * n_genes + 1), squeeze=False)
    axarr = axarr.ravel()

    for i, ax_id in enumerate(axis_ids):
        ax_df = ranking_df[ranking_df["axis"] == ax_id].head(n_genes).copy()
        ax_df = ax_df.sort_values("abs_score", ascending=True)

        colors = ["#4C72B0" if s > 0 else "#C44E52" for s in ax_df["score"]]
        axarr[i].barh(ax_df["gene"], ax_df["abs_score"], color=colors)
        axarr[i].set_title(f"Axis {ax_id}")
        axarr[i].set_xlabel("|loading|")
        if i > 0:
            axarr[i].set_yticklabels([])

    fig.suptitle("Top gene loadings per GGML axis", y=1.02)
    fig.tight_layout()

    shown = savefig_or_show(fig, default_name="latent_axes", show=show, save=save)
    return None if shown else fig
