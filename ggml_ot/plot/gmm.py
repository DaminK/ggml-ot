"""GMM component summary plots.

Visualises cell-type composition, patient-level component weights,
and component purity for globally-shared GMMs.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ggml_ot.plot._utils import savefig_or_show


def gmm_components(
    summary: dict[str, pd.DataFrame],
    *,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Multi-panel summary of GMM component biology.

    Parameters
    ----------
    summary
        Dict returned by :meth:`dataset.summarize_gmm_components` containing
        ``"celltype_table"``, ``"patient_weights"``, ``"purity"``.
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
    ct = summary["celltype_table"]
    pw = summary["patient_weights"]
    purity = summary["purity"]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, 0.4 * len(ct))))

    # --- Panel 1: cell-type composition heatmap ---
    sns.heatmap(
        ct.astype(float),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"shrink": 0.6},
    )
    axes[0].set_title("Cell-type composition")
    axes[0].set_ylabel("Component")
    axes[0].set_xlabel("")

    # --- Panel 2: patient weight heatmap ---
    sns.heatmap(
        pw.astype(float),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[1],
        cbar_kws={"shrink": 0.6},
    )
    axes[1].set_title("Patient weights")
    axes[1].set_ylabel("Patient")
    axes[1].set_xlabel("Component")

    # --- Panel 3: purity bar chart ---
    colors = sns.color_palette("Set2", n_colors=len(purity["dominant_type"].unique()))
    type_colors = {t: c for t, c in zip(sorted(purity["dominant_type"].unique()), colors)}
    bar_colors = [type_colors[t] for t in purity["dominant_type"]]

    axes[2].barh(purity.index, purity["purity"], color=bar_colors)
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Component purity")
    axes[2].set_xlabel("Max cell-type proportion")

    # Legend for dominant types
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=type_colors[t], label=t) for t in sorted(type_colors)]
    axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.suptitle("GMM component summary", y=1.02)
    fig.tight_layout()

    shown = savefig_or_show(fig, default_name="gmm_components", show=show, save=save)
    return None if shown else fig
