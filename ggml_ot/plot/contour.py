import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Literal


def dict_to_slice(d: dict) -> dict:
    """Map a dict of parameter values to a multiindex on a result DataFrame."""
    fixed_param_order = ["n_comps", "reg_type", "alpha", "reg"]
    index = [d[p] if (p in d.keys() and d[p] is not None) else slice(None) for p in fixed_param_order]
    return pd.IndexSlice[tuple(index)]


def dict_to_grid(df, x, y, fixed_params, col_val) -> pd.DataFrame:
    """Create a pivot table grid for contour plotting from a results DataFrame."""
    fixed_params = fixed_params or {}
    index_slice = dict_to_slice({x: None, y: None, **fixed_params})
    subset_df = df.loc[index_slice, col_val].reset_index()

    # flatten MultiIndex columns as pivot_table does not support them
    subset_df.columns = [" ".join(col).strip() for col in subset_df.columns.values]
    flat_col_val = " ".join(col_val).strip()

    return pd.pivot_table(subset_df, index=y, columns=x, values=flat_col_val)


def contour_hyperparams(
    results_df,
    x: str = "alpha",
    y: str = "reg",
    fixed_params: dict | None = None,
    value_col: str | tuple | None = ("knn_acc", "Mean"),
    log_axis: bool | Literal["x", "y"] = True,
    levels: int = 20,
    cmap: str = "RdBu_r",
    pad=True,
    return_figure: bool = False,
):
    """
    Contour plot visualizing grid search over two hyperparameters (x and y),
    while fixing or averaging across remaining parameters.

    Parameters
    ----------
    results_df :
        DataFrame from :meth:`ggml_ot.tune`
    x : "alpha" | "reg" | "n_comps" | "reg_type"
        Name of the hyperparameter to use as horizontal axis.
    y : "alpha" | "reg" | "n_comps" | "reg_type"
        Name of the hyperparameter to use as vertical axis
    fixed_params : dict, optional
        Mapping of the remaining two level names -> value to fix for the plot. e.g.
        {"comps_n": 2, "reg_type": 1}. If the value of a parameter is set to None, averages across all it's values. If None (default), averages across all values of the remaining parameters.
    value_col : str or tuple, optional
        Column selector when results_df is a DataFrame with multiple columns. Can be a tuple
        when columns are a MultiIndex (default ("knn_acc","Mean")).
    log_axis : bool, default True
        If True, axes values are log10-transformed for plotting (tick labels show original values).
    levels : int, default 20
        Number of contour levels.
    cmap : str, default "RdBu_r"
        Matplotlib colormap name.
    pad: bool, default True
        If True, pads the contour grid by repeating edge values to avoid cutting off contours at the edges.
    return_figure : bool, default False
        If True, returns the matplotlib figure and axis containing the contour plot.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axis containing the contour plot. Only returned if return_figure=True.

    """
    assert x != "reg_type" and y != "reg_type ", (
        "reg_type is a categorical parameter and cannot be used as axis in contour plots."
    )

    grid = dict_to_grid(results_df, x=x, y=y, fixed_params=fixed_params, col_val=value_col)

    X = np.asarray(grid.columns, dtype=float)
    Y = np.asarray(grid.index, dtype=float)
    Z = grid.to_numpy()

    if pad:
        X = np.pad(X, (1, 1), "constant", constant_values=(0.9 * X[0], X[-1] * 1.1))
        Y = np.pad(Y, (1, 1), "constant", constant_values=(0.9 * Y[0], Y[-1] * 1.1))
        Z = np.pad(Z, ((1, 1), (1, 1)), "edge")

    if log_axis == "x" or log_axis is True:
        X = np.log10(X)
    if log_axis == "y" or log_axis is True:
        Y = np.log10(Y)

    fig, ax = plt.subplots()
    cntr = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, corner_mask=True)

    if pad:
        X = X[1:-1]
        Y = Y[1:-1]

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xticks(X)
    ax.set_yticks(Y)

    fmt = ticker.FuncFormatter(lambda v, _: r"$10^{{{:g}}}$".format(v))
    if log_axis == "x" or log_axis is True:
        ax.xaxis.set_major_formatter(fmt)
    if log_axis == "y" or log_axis is True:
        ax.yaxis.set_major_formatter(fmt)

    fig.colorbar(cntr, ax=ax, label=" ".join(value_col).strip() if value_col is not None else "value")

    if fixed_params is not None:
        title = ",".join(list(reversed([f" {p}: {v}" for p, v in fixed_params.items() if v is not None])))
        plt.suptitle(title)

    plt.tight_layout()

    if return_figure:
        return fig, ax
    else:
        plt.show()
