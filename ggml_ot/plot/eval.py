import pandas.io.formats.style as style
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display


def confusion_matrix(predicted, true, title=None, ax=None):
    """Plots a heatmap of the confusion matrix of given predicted and true labels.

    :param predicted: predicted labels
    :type predicted: array-like
    :param true: labels of ground truth
    :type true: array-like
    :param title: title of the plot, defaults to None
    :type title: str, optional
    :param ax: axis to plot on, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    """
    annot_labels_ind = np.unique(true, return_index=True)[1]
    annot_labels = true[annot_labels_ind]

    cf_matrix = sk_confusion_matrix(true, predicted, labels=annot_labels)
    if ax is None:
        plt.figure()
    ax = sns.heatmap(
        cf_matrix,
        annot=True,  # fmt='.0',
        cmap="Blues",
        xticklabels=annot_labels,
        yticklabels=annot_labels,
        ax=ax,
        fmt="g",
    )
    ax.set(xlabel="Predicted Label", ylabel="True Label")
    ax.set_title(title)


def table(df, style_performance=False, print_latex=False, title=""):
    """
    Displays a DataFrame of evaluation metrics as a formatted (LaTeX) table.

    Can be used to show the results of :meth:`ggml_ot.test`, :meth:`ggml_ot.train_test` and :meth:`ggml_ot.tune`. You can also concatenate different results DataFrames and pass them to this function.

    Parameters
    ----------
    df : DataFrame
        Data to display
    style_performance : bool, optional
        Whether to highlight the best performing row (e.g. parameter combination) in the table, by default False
    print_latex: bool, optional
        Whether to print the table as LaTeX, defaults to False
    """

    if style_performance:
        data_df = df.data if isinstance(df, style.Styler) else df
        best_index = data_df[("knn_acc", "Mean")].idxmax()

        # Merge Mean±SD
        sep = "±"
        merged_df = combine_mean_sd(data_df, sep=sep, fmt="{:.2f}", mean_label="Mean", sd_label="SD", zero_tol=1e-8)

        # lots of manual styling due to str columns with mean±sd
        if len(merged_df) > 1:
            idx = pd.IndexSlice
            slice_ = idx[idx[best_index], :]
            cm_green = sns.color_palette("light:green", as_cmap=True)
            cm_red = sns.color_palette("light:tomato", as_cmap=True)
            lvl2_col = merged_df.columns.levels[1][0]
            df = (
                merged_df.style.set_caption(title)
                .apply(lambda c: mean_sd_to_hex(c, cm_green, sep=sep), subset=[("knn_acc", lvl2_col)])
                .apply(lambda c: mean_sd_to_hex(c, cm_red, sep=sep), subset=[("epoch_time", lvl2_col)])
                # .background_gradient(cmap=cm_green, subset=[("knn_acc", "Mean")]) #only works for float
                # .background_gradient(cmap=cm_red, subset=[("epoch_time", "Mean")]) #only works for float
                .map(lambda _: "font-weight:bold", subset=slice_)
                .map_index(lambda i: "font-weight:bold" if i == best_index else None)
                .format(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
                .format_index(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
            )
        else:
            df = merged_df.style.hide(axis="index")

    format_df = (
        df if isinstance(df, style.Styler) else df.style.format(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    )

    if print_latex:
        print(format_df.data.to_latex(index=True, float_format="{:.2f}".format))

    display(format_df)


def mean_sd_to_hex(mean_sd_str, cmap, sep="±"):
    mean_float = mean_sd_str.map(lambda s: float(s.split(sep)[0])).to_numpy()
    mean_float_norm = (mean_float - mean_float.min()) / (mean_float.max() - mean_float.min())
    # TODO integrate font color
    # dark = relative_luminance(rgba) < text_color_threshold
    # text_color = "#f1f1f1" if dark else "#000000"
    # color: {text_color};"
    return [f"background-color:{mpl.colors.rgb2hex(cmap(v))}" for v in mean_float_norm]


# From Pandas
def relative_luminance(rgba) -> float:
    """
    Calculate relative luminance of a color.

    The calculation adheres to the W3C standards
    (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

    Parameters
    ----------
    color : rgb or rgba tuple

    Returns
    -------
    float
        The relative luminance as a value from 0 to 1
    """
    r, g, b = (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgba[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def combine_mean_sd(df, sep="±", fmt="{:.2f}", mean_label="Mean", sd_label="SD", zero_tol=1e-8):
    """
    Given a DataFrame with 2-level MultiIndex columns (Metric, Stat),
    combine (Metric, Mean) and (Metric, SD) into (Metric, 'Mean±SD') with formatted strings.
    Omit SD if it only contains zeros, i.e., only one split was tested.
    """
    top_order = list(df.columns.get_level_values(0).unique())
    new_cols = []
    new_frames = []

    for top in top_order:
        mean_col = (top, mean_label)
        sd_col = (top, sd_label)
        mean_s = df[mean_col]
        sd_s = df[sd_col]

        # decide whether sd is (effectively) zero
        non_na = sd_s.dropna()
        is_zero_sd = non_na.abs().le(zero_tol).all() if len(non_na) > 0 else True

        if is_zero_sd:
            # only keep mean (formatted)
            col_name = (top, mean_label)
            formatted = mean_s.map(lambda x: fmt.format(x) if pd.notnull(x) else "").rename(col_name)
        else:
            # combine mean±sd
            fmt_mean = mean_s.map(lambda x: fmt.format(x) if pd.notnull(x) else "")
            fmt_sd = sd_s.map(lambda x: fmt.format(x) if pd.notnull(x) else "")
            combined = (fmt_mean + sep + fmt_sd).rename((top, f"{mean_label}{sep}{sd_label}"))
            col_name = (top, f"{mean_label}{sep}{sd_label}")
            formatted = combined

        new_cols.append(col_name)
        new_frames.append(formatted)

    out = pd.concat(new_frames, axis=1)

    # ensure MultiIndex and preserve level names if present
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    if df.columns.names:
        out.columns.names = df.columns.names

    return out
