"""Shared plotting utilities for the ggml_ot.plot subpackage.

Centralises save / show logic so that every public plot function
behaves consistently.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def savefig_or_show(
    fig_or_ax,
    *,
    default_name: str,
    show: bool | None = None,
    save: str | bool | None = None,
) -> bool:
    """Handle save-to-disk and show logic for plot functions.

    Parameters
    ----------
    fig_or_ax
        The matplotlib ``Figure``, ``Axes``, or seaborn ``ClusterGrid``
        that was drawn on.  A ``Figure`` is extracted automatically.
    default_name
        Default filename stem when ``save=True`` (e.g. ``"clustermap"``).
    show
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves as
        ``settings.figdir/<default_name>.<settings.figformat>``.
        A *str* value is treated as the full filename (or suffix),
        saved into ``settings.figdir/``.

    Returns
    -------
    bool
        ``True`` if ``plt.show()`` was called (callers that return a
        ``Figure`` should return ``None`` in this case to prevent
        Jupyter from auto-displaying the figure a second time).
    """
    from ggml_ot.settings import settings  # deferred to avoid circular import

    # Resolve figure from whatever was passed in.
    fig = _resolve_figure(fig_or_ax)

    # --- Save ---
    if save:
        figdir = Path(settings.figdir) if settings.figdir is not None else None
        if figdir is None:
            raise ValueError(
                "Cannot save plot: `ggml_ot.settings.figdir` is not set. "
                "Set it first, e.g. `ggml_ot.settings.figdir = './figures'`."
            )
        figdir.mkdir(parents=True, exist_ok=True)

        if isinstance(save, str):
            filename = save
        else:
            filename = f"{default_name}.{settings.figformat}"

        filepath = figdir / filename
        fig.savefig(filepath, dpi=settings.figdpi, bbox_inches="tight")
        print(f"Saved plot to {filepath}")

    # --- Show ---
    if show is None:
        show = matplotlib.is_interactive()
    if show:
        plt.show()
        return True
    return False


def _resolve_figure(fig_or_ax):
    """Return a ``matplotlib.Figure`` from various plot objects."""
    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        return fig_or_ax
    if hasattr(fig_or_ax, "figure"):
        # Axes, ClusterGrid, etc.
        fig = fig_or_ax.figure
        if isinstance(fig, matplotlib.figure.Figure):
            return fig
        return fig  # some wrappers return the figure directly
    raise TypeError(
        f"Cannot resolve a matplotlib Figure from {type(fig_or_ax)!r}. Pass a Figure, Axes, or ClusterGrid."
    )


def _normalise_plot_types(plot) -> list[str]:
    """Normalize plot selection input to a list of concrete plot names."""
    if plot in (False, None):
        return []
    if plot is True:
        return ["clustermap_embedding"]
    if isinstance(plot, str):
        return [plot]
    if isinstance(plot, (list, tuple)):
        plot_types = []
        for item in plot:
            if item is True:
                plot_types.append("clustermap_embedding")
            elif item not in (False, None):
                plot_types.append(str(item))
        return plot_types
    raise TypeError(f"Unsupported plot type specification: {type(plot)!r}")


def _split_dirname(split_idx: int, n_splits: int) -> str:
    """Return the canonical directory name for one train/test split."""
    width = max(2, len(str(max(n_splits - 1, 0))))
    return f"split_{split_idx:0{width}d}"


def _resolve_plot_title(plot_title: str | None, ground_metric, split_idx: int, n_splits: int) -> str | None:
    """Resolve the plot title for one train/test split."""
    title = (
        plot_title
        if plot_title is not None
        else (f"{ground_metric} ground metric" if isinstance(ground_metric, str) else "GGML")
    )
    if title is None:
        return None
    if n_splits > 1:
        return f"{title} ({_split_dirname(split_idx, n_splits).replace('_', ' ')})"
    return title


def _save_train_test_split_plots(
    distances,
    labels,
    symbols,
    plot,
    *,
    ground_metric=None,
    split_idx: int,
    n_splits: int,
    plot_split_dir: str | Path | None,
    plot_title: str | None,
) -> None:
    """Render and optionally save benchmark split plots.

    Parameters
    ----------
    distances
        Pairwise distance matrix for the split.
    labels
        Distribution labels for the split.
    symbols
        Symbol annotations for the split, typically train/test.
    plot
        Plot selection, matching the public ``train_test(plot_type=...)`` API.
    split_idx
        Zero-based split index.
    n_splits
        Total number of train/test splits.
    plot_split_dir
        Root directory for split outputs. When ``None``, plots are only shown.
    plot_title
        Base title for the split plot(s).
    """
    plot_types = _normalise_plot_types(plot)
    if not plot_types:
        return

    from ggml_ot import settings
    from ggml_ot.plot.clustermap import clustermap
    from ggml_ot.plot.embedding import emb as embedding
    from ggml_ot.plot.plotting import clustermap_embedding

    split_dir = None if plot_split_dir is None else Path(plot_split_dir) / _split_dirname(split_idx, n_splits)
    resolved_title = _resolve_plot_title(plot_title, ground_metric, split_idx, n_splits)

    previous_figdir = settings.figdir
    if split_dir is not None:
        settings.figdir = split_dir

    try:
        for plot_type in plot_types:
            save_name = f"{plot_type}.{settings.figformat}" if split_dir is not None else False
            show = False if split_dir is not None else None

            if plot_type == "clustermap":
                grid = clustermap(distances, labels, title=resolved_title, show=show, save=save_name)
                if split_dir is not None:
                    plt.close(grid.figure)
            elif plot_type == "embedding":
                ax = embedding(
                    distances,
                    labels,
                    symbols=symbols,
                    title=resolved_title,
                    legend="Side",
                    show=show,
                    save=save_name,
                )
                if split_dir is not None:
                    plt.close(ax.figure)
            elif plot_type == "clustermap_embedding":
                fig = clustermap_embedding(
                    distances,
                    labels,
                    symbols=symbols,
                    title=resolved_title,
                    show=show,
                    save=save_name,
                )
                if split_dir is not None:
                    plt.close(fig)
            else:
                raise ValueError(f"Unknown plot type: {plot_type!r}")
    finally:
        settings.figdir = previous_figdir
