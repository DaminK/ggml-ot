"""Enrichment result plots.

Dot-plots and bar-plots for pathway enrichment results from both
axis-level and GMM-component-level analyses.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from ggml_ot.plot._utils import savefig_or_show


def latent_enrichment(
    enrichment_df: pd.DataFrame,
    *,
    group_col: str = "axis",
    top_n: int = 10,
    pvalue_threshold: float = 0.05,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Dot-plot of enrichment results per axis or component.

    Parameters
    ----------
    enrichment_df
        Long-format DataFrame with columns ``<group_col>``, ``pathway``,
        ``score``, ``norm_score``, ``pvalue`` and optionally
        ``pathway_label`` for prettier plot labels.
    group_col
        Column identifying the group (``"axis"`` or ``"component"``).
    top_n
        Number of top pathways to show per group (by absolute score).
    pvalue_threshold
        Significance threshold; non-significant hits are greyed out.
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
    groups = sorted(enrichment_df[group_col].unique())
    n_groups = len(groups)
    pathway_col = "pathway_label" if "pathway_label" in enrichment_df.columns else "pathway"

    fig, axarr = plt.subplots(1, n_groups, figsize=(5 * n_groups, 0.4 * top_n + 1.5), squeeze=False)
    axarr = axarr.ravel()

    for i, grp in enumerate(groups):
        grp_df = enrichment_df[enrichment_df[group_col] == grp].copy()
        grp_df["abs_score"] = grp_df["score"].abs()
        grp_df = grp_df.nlargest(top_n, "abs_score")
        grp_df = grp_df.sort_values("abs_score", ascending=True)

        colors = []
        for _, row in grp_df.iterrows():
            if row["pvalue"] > pvalue_threshold:
                colors.append("#CCCCCC")
            elif row["score"] > 0:
                colors.append("#4C72B0")
            else:
                colors.append("#C44E52")

        axarr[i].barh(grp_df[pathway_col], grp_df["abs_score"], color=colors)
        axarr[i].set_title(f"{group_col.capitalize()} {grp}")
        axarr[i].set_xlabel("|enrichment score|")

        # Significance line
        if i == 0:
            axarr[i].axvline(0, color="grey", linewidth=0.5)

    fig.suptitle("Pathway enrichment", y=1.02)
    fig.tight_layout()

    default_name = f"{group_col}_enrichment"
    shown = savefig_or_show(fig, default_name=default_name, show=show, save=save)
    return None if shown else fig


def gmm_component_enrichment(
    enrichment_df: pd.DataFrame,
    *,
    top_n: int = 10,
    pvalue_threshold: float = 0.05,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Shorthand for :func:`latent_enrichment` with ``group_col="component"``."""
    return latent_enrichment(
        enrichment_df,
        group_col="component",
        top_n=top_n,
        pvalue_threshold=pvalue_threshold,
        show=show,
        save=save,
    )
