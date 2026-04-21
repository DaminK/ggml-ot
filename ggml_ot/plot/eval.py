"""Evaluation and result-display helpers for ggml_ot.

Contains functions for rendering confusion matrices, styled metric
tables, and related formatting utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.io.formats.style as style

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from ggml_ot.plot._utils import savefig_or_show


def confusion_matrix(
    predicted,
    true,
    *,
    title=None,
    ax=None,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Plot a confusion-matrix heatmap.

    Parameters
    ----------
    predicted : array-like
        Predicted labels.
    true : array-like
        Ground-truth labels.
    title : str or None, default None
        Plot title.
    ax : matplotlib.axes.Axes or None, default None
        Axes to draw on.  A new figure is created when ``None``.
    show : bool or None, default None
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save : str, bool, or None, default None
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves to ``settings.figdir/confusion_matrix.<figformat>``;
        a *str* overrides the filename.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.
    """
    annot_labels_ind = np.unique(true, return_index=True)[1]
    annot_labels = true[annot_labels_ind]

    cf_matrix = sk_confusion_matrix(true, predicted, labels=annot_labels)
    if ax is None:
        plt.figure()
    ax = sns.heatmap(
        cf_matrix,
        annot=True,
        cmap="Blues",
        xticklabels=annot_labels,
        yticklabels=annot_labels,
        ax=ax,
        fmt="g",
    )
    ax.set(xlabel="Predicted Label", ylabel="True Label")
    ax.set_title(title)

    savefig_or_show(ax, default_name="confusion_matrix", show=show, save=save)

    return ax


def table(
    df,
    *,
    style_performance=False,
    print_latex=False,
    title="",
    save: str | bool | None = None,
):
    """Display a DataFrame of evaluation metrics as a formatted table.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to display.
    style_performance : bool, default False
        Highlight the best-performing row in the table.
    print_latex : bool, default False
        Print a LaTeX representation of the table.
    title : str, default ``""``
        Caption for the styled table.
    save : str, bool, or None, default None
        ``True`` renders the table as a PDF to
        ``settings.figdir/table.<figformat>``; a *str* overrides the
        filename.  ``None``/``False`` skips saving.
    """
    if style_performance:
        data_df = df.data if isinstance(df, style.Styler) else df

        # Highlight best row
        best_index = data_df[("knn", "mean")].idxmax()
        idx = pd.IndexSlice
        slice_ = idx[idx[best_index], :]

        def _styler_map_or_applymap(s, func, subset=None):
            if hasattr(s, "map"):
                return s.map(func, subset=subset)
            else:
                return s.applymap(func, subset=subset)

        def _styler_map_index_or_apply_index(s, func, axis=0):
            if hasattr(s, "map_index"):
                return s.map_index(func)
            else:
                return s.apply_index(func)

        cm_green = sns.color_palette("light:green", as_cmap=True)
        cm_red = sns.color_palette("light:tomato", as_cmap=True)

        df = (
            data_df.style.set_caption(title)
            .background_gradient(cmap=cm_green, subset=pd.IndexSlice[:, ("knn", "mean")])
            .pipe(
                lambda s: s.background_gradient(
                    cmap=cm_red, subset=pd.IndexSlice[:, ("epoch_time(s)", "mean")]
                ).applymap(lambda x: "color: transparent; background-color: transparent" if pd.isnull(x) else "")
                if ("epoch_time(s)", "mean") in data_df.columns
                else s
            )
            .pipe(lambda s: _styler_map_or_applymap(s, lambda _: "font-weight:bold", subset=slice_))
            .pipe(
                lambda s: _styler_map_index_or_apply_index(s, lambda i: "font-weight:bold" if i == best_index else None)
            )
            .format(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
            .format_index(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        )

    display(df)

    # --- Save table as PDF via matplotlib ---
    if save:
        _save_table_as_figure(df, title=title, save=save)

    if print_latex:
        data_df = df.data if isinstance(df, style.Styler) else df
        best_index = data_df[("knn", "mean")].idxmax()

        sep = "±"
        merged_df = combine_mean_sd(data_df, sep=sep, fmt="{:.2f}", mean_label="mean", sd_label="SD", zero_tol=1e-8)

        if len(merged_df) > 1:
            idx = pd.IndexSlice
            slice_ = idx[idx[best_index], :]
            merged_df = (
                merged_df.style.set_caption(title)
                .pipe(lambda s: _styler_map_or_applymap(s, lambda _: "font-weight:bold", subset=slice_))
                .pipe(
                    lambda s: _styler_map_index_or_apply_index(
                        s, lambda i: "font-weight:bold" if i == best_index else None
                    )
                )
                .format(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
                .format_index(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
            )

        print(merged_df.data.to_latex(index=True, float_format="{:.2f}".format))


def _save_table_as_figure(df, *, title, save):
    """Render a DataFrame as a matplotlib table and save to disk."""

    data_df = df.data if isinstance(df, style.Styler) else df

    fig, ax = plt.subplots(figsize=(max(8, len(data_df.columns) * 1.2), max(2, len(data_df) * 0.5 + 1)))
    ax.axis("off")

    tbl = ax.table(
        cellText=data_df.values,
        colLabels=[" ".join(c) if isinstance(c, tuple) else str(c) for c in data_df.columns],
        rowLabels=[str(i) for i in data_df.index],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.4)

    if title:
        ax.set_title(title, fontsize=10, pad=20)

    savefig_or_show(fig, default_name="table", show=False, save=save)
    plt.close(fig)


def combine_mean_sd(df, sep="±", fmt="{:.2f}", mean_label="mean", sd_label="SD", zero_tol=1e-8):
    """Combine ``(metric, mean)`` and ``(metric, SD)`` columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 2-level MultiIndex columns ``(metric, stat)``.
    sep : str, default ``"±"``
        Separator between mean and standard deviation.
    fmt : str, default ``"{:.2f}"``
        Format string for numeric values.
    mean_label : str, default ``"mean"``
        Label identifying the mean statistic.
    sd_label : str, default ``"SD"``
        Label identifying the standard-deviation statistic.
    zero_tol : float, default ``1e-8``
        Threshold below which SD is treated as zero.

    Returns
    -------
    pandas.DataFrame
        DataFrame with combined ``mean±SD`` columns.
    """
    top_order = list(df.columns.get_level_values(0).unique())
    new_cols = []
    new_frames = []

    for top in top_order:
        mean_col = (top, mean_label)
        sd_col = (top, sd_label)
        mean_s = df[mean_col]
        sd_s = df[sd_col]

        non_na = sd_s.dropna()
        is_zero_sd = non_na.abs().le(zero_tol).all() if len(non_na) > 0 else True

        if is_zero_sd:
            col_name = (top, mean_label)
            formatted = mean_s.map(lambda x: fmt.format(x) if pd.notnull(x) else "nan").rename(col_name)
        else:
            fmt_mean = mean_s.map(lambda x: fmt.format(x) if pd.notnull(x) else "nan")
            fmt_sd = sd_s.map(lambda x: fmt.format(x) if pd.notnull(x) else "nan")
            combined = (fmt_mean + sep + fmt_sd).rename((top, f"{mean_label}{sep}{sd_label}"))
            col_name = (top, f"{mean_label}{sep}{sd_label}")
            formatted = combined

        new_cols.append(col_name)
        new_frames.append(formatted)

    out = pd.concat(new_frames, axis=1)

    out.columns = pd.MultiIndex.from_tuples(out.columns)
    if df.columns.names:
        out.columns.names = df.columns.names

    return out


def gmm_fit_validation_boxplot(
    results,
    *,
    train_col: str = "train_nll",
    validation_col: str = "validation_nll",
    train_label: str = "Train",
    validation_label: str = "Validation",
    ylabel: str = "Per-cell negative log-likelihood",
    title: str | None = None,
    ax=None,
    show_points: bool = True,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Plot train-vs-validation GMM fit NLL distributions as a box plot.

    Parameters
    ----------
    results : pandas.DataFrame
        Patient-wise GMM fit validation dataframe.
    train_col
        Column containing train NLL values.
    validation_col
        Column containing validation NLL values.
    train_label
        Display label for the train split.
    validation_label
        Display label for the validation split.
    ylabel
        Y-axis label.
    title
        Optional plot title.
    ax : matplotlib.axes.Axes or None, default None
        Axes to draw on. A new figure is created when ``None``.
    show_points
        Overlay individual patient points.
    show : bool or None, default None
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save : str, bool, or None, default None
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves under the default name into
        ``settings.figdir``.  A *str* is used as the filename.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the box plot.
    """
    data = pd.DataFrame(results).copy()
    required_cols = {train_col, validation_col}
    missing = required_cols.difference(data.columns)
    if missing:
        raise KeyError(f"Missing GMM fit validation columns: {sorted(missing)}")

    train_values = pd.to_numeric(data[train_col], errors="coerce").dropna().to_numpy()
    validation_values = pd.to_numeric(data[validation_col], errors="coerce").dropna().to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.0))

    positions = [1, 2]
    colors = ["#4C72B0", "#DD8452"]
    box = ax.boxplot(
        [train_values, validation_values],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        whis=1.5,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.2)
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.4)
    for whisker in box["whiskers"]:
        whisker.set_color("#555555")
        whisker.set_linewidth(1.0)
    for cap in box["caps"]:
        cap.set_color("#555555")
        cap.set_linewidth(1.0)
    for flier in box["fliers"]:
        flier.set_markeredgecolor("#555555")
        flier.set_alpha(0.5)

    if show_points:
        rng = np.random.default_rng(0)
        for pos, values in zip(positions, [train_values, validation_values]):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), pos, dtype=float) + jitter,
                values,
                color="#2B2B2B",
                alpha=0.35,
                s=14,
                linewidths=0,
                zorder=3,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([train_label, validation_label])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)

    savefig_or_show(ax, default_name="gmm_fit_validation_boxplot", show=show, save=save)
    return ax
