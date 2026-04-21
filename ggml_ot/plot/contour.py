"""Contour-plot visualisation of hyperparameter grid-search results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Literal

from ggml_ot.plot._utils import savefig_or_show


_LABEL_STYLES: dict[str, dict[str, str]] = {
    "package": {},  # use raw parameter names (alpha, reg, mi_reg, n_comps)
    "paper": {
        "alpha": r"$\alpha$",
        "reg": r"$\lambda_F$",
        "mi_reg": r"$\lambda_{\mathrm{MI}}$",
        "n_comps": r"$d$",
    },
}


def contour_hyperparams(
    results_df,
    *,
    x: str = "alpha",
    y: str = "reg",
    fixed_params: dict | None = None,
    value_col: str | tuple | None = ("knn", "mean"),
    log_axis: bool | Literal["x", "y"] = True,
    label_style: Literal["package", "paper"] = "package",
    levels: int = 20,
    cmap: str = "RdBu_r",
    pad: bool = True,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Contour plot of grid-search results over two hyperparameters.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame produced by :func:`ggml_ot.tune`.
    x : str, default ``"alpha"``
        Hyperparameter for the horizontal axis.
    y : str, default ``"reg"``
        Hyperparameter for the vertical axis.
    fixed_params : dict or None, default None
        Fix remaining hyperparameters to specific values.  Keys with
        ``None`` values are averaged across.
    value_col : str or tuple, default ``("knn", "mean")``
        Column selector for the metric to display.
    log_axis : bool or ``"x"`` or ``"y"``, default True
        Log10-transform axes (tick labels show original values).
    label_style : ``"package"`` or ``"paper"``, default ``"package"``
        ``"package"`` uses raw parameter names (alpha, reg, …).
        ``"paper"`` uses LaTeX notation (α, λ_F, λ_MI, d).
    levels : int, default 20
        Number of contour levels.
    cmap : str, default ``"RdBu_r"``
        Matplotlib colourmap name.
    pad : bool, default True
        Pad the contour grid to avoid clipping at the edges.
    show : bool or None, default None
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save : str, bool, or None, default None
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves to ``settings.figdir/contour_hyperparams.<figformat>``;
        a *str* overrides the filename.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes containing the contour plot.
    """
    if x == "reg_type" or y == "reg_type":
        raise ValueError("reg_type is a categorical parameter and cannot be used as axis in contour plots.")

    grid = _dict_to_grid(results_df, x=x, y=y, fixed_params=fixed_params, col_val=value_col)

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

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    else:
        ax.set_aspect("equal", adjustable="box")

    cntr = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, corner_mask=True)

    if pad:
        X = X[1:-1]
        Y = Y[1:-1]

    labels = _LABEL_STYLES.get(label_style, {})
    ax.set_xlabel(labels.get(x, x))
    ax.set_ylabel(labels.get(y, y))
    ax.set_xticks(X)
    ax.set_yticks(Y)

    fmt = ticker.FuncFormatter(lambda v, _: r"$10^{{{:g}}}$".format(v))
    if log_axis == "x" or log_axis is True:
        ax.xaxis.set_major_formatter(fmt)
    if log_axis == "y" or log_axis is True:
        ax.yaxis.set_major_formatter(fmt)

    fig.colorbar(cntr, ax=ax, label=" ".join(value_col).strip() if value_col is not None else "value")

    if fixed_params is not None:
        parts = [f" {labels.get(p, p)}: {v}" for p, v in fixed_params.items() if v is not None]
        suptitle = ",".join(reversed(parts))
        plt.suptitle(suptitle)

    plt.tight_layout()

    savefig_or_show(fig, default_name="contour_hyperparams", show=show, save=save)

    return fig, ax


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _dict_to_slice(d: dict) -> dict:
    """Map a dict of parameter values to a MultiIndex slice."""
    fixed_param_order = ["n_comps", "reg_type", "alpha", "reg", "mi_reg"]
    index = [d[p] if (p in d.keys() and d[p] is not None) else slice(None) for p in fixed_param_order]
    return pd.IndexSlice[tuple(index)]


def _dict_to_grid(df, x, y, fixed_params, col_val) -> pd.DataFrame:
    """Create a pivot table grid for contour plotting."""
    fixed_params = fixed_params or {}
    index_slice = _dict_to_slice({x: None, y: None, **fixed_params})
    subset_df = df.loc[index_slice, col_val].reset_index()

    subset_df.columns = [" ".join(col).strip() for col in subset_df.columns.values]
    flat_col_val = " ".join(col_val).strip()

    return pd.pivot_table(subset_df, index=y, columns=x, values=flat_col_val)
