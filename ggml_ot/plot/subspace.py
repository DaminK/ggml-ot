"""Subspace and GMM visualisation for GGML.

Provides functions to visualise points and Gaussian ellipses in
2-D subspaces — ground-truth latent space, rotated (mixed) space,
or a learned projection — colored by arbitrary metadata columns.

All public functions follow the ``show`` / ``save`` convention from
:mod:`ggml_ot.plot._utils`.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Sequence, Literal

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from anndata import AnnData
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from ggml_ot._utils._covariance import canonicalize_covariances
from ggml_ot.plot._utils import savefig_or_show
from ggml_ot._utils._array import to_numpy as _to_np, not_none


# =========================================================================
# 1. Low-level helpers
# =========================================================================


def _resolve_palette(
    values: np.ndarray,
    palette: dict | str | None,
) -> dict:
    """Return a ``{value: color}`` mapping.

    Parameters
    ----------
    values
        Array of categorical values (e.g. class labels).
    palette
        - ``dict``: passed through.
        - ``str``: name of a matplotlib/seaborn colormap.
        - ``None``: auto-generated from *tab10*.
    """
    unique = np.unique(values)
    if isinstance(palette, dict):
        return palette
    cmap_name = palette if isinstance(palette, str) else "tab10"
    cmap = mpl.colormaps[cmap_name]
    return {v: cmap(i % cmap.N) for i, v in enumerate(unique)}


@dataclass(frozen=True)
class _SynthColorScheme:
    class_palette: dict
    patient_palette: dict
    neutral_color: str = "dimgray"


def _shift_color_lightness(color, delta: float):
    """Return a lightened/darkened variant of *color*."""
    r, g, b = mcolors.to_rgb(color)
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    lightness = float(np.clip(lightness + delta, 0.0, 1.0))
    return colorsys.hls_to_rgb(h, lightness, s)


def _patient_lightness_deltas(n_patients: int) -> list[float]:
    """Return stable per-class lightness offsets for patient colours."""
    if n_patients <= 0:
        return []
    deltas = [0.0]
    step = 0.10
    while len(deltas) < n_patients:
        deltas.append(-step)
        if len(deltas) < n_patients:
            deltas.append(step)
        step += 0.08
    return deltas[:n_patients]


def _build_patient_palette(*, patient_to_label: dict, class_palette: dict) -> dict:
    """Derive patient colours from the class palette."""
    patient_palette = {}
    unique_labels = sorted(set(patient_to_label.values()), key=str)
    for label in unique_labels:
        patient_ids = sorted([pid for pid, lbl in patient_to_label.items() if lbl == label], key=str)
        base_color = class_palette[label]
        deltas = _patient_lightness_deltas(len(patient_ids))
        for patient_id, delta in zip(patient_ids, deltas):
            patient_palette[patient_id] = base_color if delta == 0.0 else _shift_color_lightness(base_color, delta)
    return patient_palette


def _build_synth_color_scheme(
    *,
    sample_labels: np.ndarray,
    sample_patient_ids: np.ndarray,
    palette: dict | str | None,
) -> _SynthColorScheme:
    """Build a single colour scheme shared across synthetic-dataset panels."""
    class_palette = _resolve_palette(np.asarray(sample_labels), palette)
    patient_to_label = {}
    for patient_id, label in zip(np.asarray(sample_patient_ids).tolist(), np.asarray(sample_labels).tolist()):
        if patient_id not in patient_to_label:
            patient_to_label[patient_id] = label
    patient_palette = _build_patient_palette(patient_to_label=patient_to_label, class_palette=class_palette)
    return _SynthColorScheme(class_palette=class_palette, patient_palette=patient_palette)


def _resolve_synth_point_palette(
    *,
    color_key: str,
    color_values: np.ndarray,
    color_scheme: _SynthColorScheme,
    palette: dict | str | None,
) -> dict:
    """Resolve the point palette from the shared synthetic colour scheme."""
    if color_key == "labels":
        return color_scheme.class_palette
    if color_key == "patient_ids":
        return color_scheme.patient_palette
    return _resolve_palette(color_values, palette)


def _resolve_projection(
    X: np.ndarray,
    projection: np.ndarray | None,
    components: tuple[int, int],
) -> np.ndarray:
    """Project *X* to 2-D.

    Parameters
    ----------
    X
        Input array of shape ``(N, D)``.
    projection
        Optional ``(D, K)`` or ``(K, D)`` projection matrix.
        If ``None``, columns *components* of *X* are used directly.
    components
        Pair of component indices to select after projection.

    Returns
    -------
    np.ndarray of shape ``(N, 2)``
    """
    if projection is not None:
        P = np.asarray(projection)
        # Accept both (D, K) and (K, D); heuristic: rows < cols → (K, D)
        if P.shape[0] < P.shape[1]:
            P = P.T  # now (D, K)
        X_proj = X @ P
    else:
        X_proj = X

    c0, c1 = components
    return X_proj[:, [c0, c1]]


def _get_tight_bounds(
    means: np.ndarray,
    covs: np.ndarray,
    padding_sigma: float = 3.5,
) -> tuple[float, float, float, float]:
    """Compute axis bounds that encompass all ellipses."""
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for i in range(len(means)):
        sig_x = np.sqrt(covs[i][0, 0])
        sig_y = np.sqrt(covs[i][1, 1])
        x_min = min(x_min, means[i][0] - padding_sigma * sig_x)
        x_max = max(x_max, means[i][0] + padding_sigma * sig_x)
        y_min = min(y_min, means[i][1] - padding_sigma * sig_y)
        y_max = max(y_max, means[i][1] + padding_sigma * sig_y)
    return x_min, x_max, y_min, y_max


def _resolve_fitted_gmm_provenance(
    *,
    dataset,
    synth_data: dict,
    supports: np.ndarray,
    covariances: np.ndarray,
    dist_patient_ids: np.ndarray,
) -> Literal["synthetic_ground_truth", "fit_gmm", "unknown"]:
    """Resolve how the current fitted GMM parameters were obtained.

    Synthetic datasets created with ``representation="gmm"`` can mark their
    GMM parameters explicitly as ground truth. Refitted datasets mark the
    output as ``"fit_gmm"``. If the provenance flag is absent, component
    identity is treated as unknown.
    """
    provenance = synth_data.get("fitted_gmm_provenance", None)
    if provenance in {"synthetic_ground_truth", "fit_gmm"}:
        return provenance
    return "unknown"


def _synth_gmm_view_label(
    *,
    provenance: Literal["synthetic_ground_truth", "fit_gmm", "unknown"],
    fitted_gmm_view: Literal["selected_patients", "class_average", "all_patients"],
) -> str:
    """Human-readable label for the GMM overlays used in synthetic panels."""
    if provenance == "synthetic_ground_truth":
        if fitted_gmm_view == "class_average":
            return "Ground-truth Class-Average GMM"
        return "Ground-truth Patient GMMs"
    if provenance == "fit_gmm":
        if fitted_gmm_view == "class_average":
            return "Class-Averaged Refitted GMM"
        return "Refitted Patient GMMs"
    if fitted_gmm_view == "class_average":
        return "Class-Average GMM"
    return "Patient GMMs"


def _synth_clean_unmix_projection(synth_data: dict) -> np.ndarray:
    """Return the map from mixed coordinates back to the clean synthetic basis."""
    Q = np.asarray(synth_data["Q_mixing"], dtype=np.float64)
    R = np.asarray(synth_data["R_rotation"], dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("`synth_data['Q_mixing']` must be a square matrix.")
    if R.shape != (2, 2):
        raise ValueError("`synth_data['R_rotation']` must have shape (2, 2).")

    projection = np.eye(Q.shape[0], dtype=np.float64)
    projection[:2, :2] = R
    return Q @ projection


def _scatter_subspace_with_patient_focus(
    X: np.ndarray,
    *,
    color_by: np.ndarray,
    patient_ids: np.ndarray | None,
    selected_patient_ids: Sequence[int] | None,
    point_view: Literal["all", "highlight_selected", "selected_only"],
    projection: np.ndarray | None = None,
    components: tuple[int, int] = (0, 1),
    palette: dict | str | None = None,
    ax: mpl.axes.Axes,
    s: float = 15,
    alpha: float = 0.3,
    background_alpha: float = 0.05,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> mpl.axes.Axes:
    """Scatter helper that can grey-out or subset points by patient id."""
    X_2d = _resolve_projection(X, projection, components)
    color_by = np.asarray(color_by)
    cmap = _resolve_palette(color_by, palette)
    colors = np.array([cmap[v] for v in color_by], dtype=object)

    use_focus = (
        patient_ids is not None
        and selected_patient_ids is not None
        and len(selected_patient_ids) > 0
        and point_view in {"highlight_selected", "selected_only"}
    )

    if use_focus:
        patient_ids = np.asarray(patient_ids)
        selected_arr = np.asarray(selected_patient_ids)
        selected_mask = np.isin(patient_ids, selected_arr)
        selected_alpha = min(1.0, max(0.7, alpha * 4.0))
        selected_size = max(28.0, s * 2.2)

        if point_view == "highlight_selected":
            bg_mask = ~selected_mask
            if np.any(bg_mask):
                ax.scatter(
                    X_2d[bg_mask, 0],
                    X_2d[bg_mask, 1],
                    c="lightgray",
                    s=s,
                    alpha=background_alpha,
                    edgecolors="none",
                    zorder=1,
                )
            if np.any(selected_mask):
                sel_idx = np.where(selected_mask)[0]
                ax.scatter(
                    X_2d[selected_mask, 0],
                    X_2d[selected_mask, 1],
                    c=[colors[i] for i in sel_idx],
                    s=selected_size,
                    alpha=selected_alpha,
                    edgecolors="black",
                    linewidths=0.35,
                    zorder=3,
                )
        else:  # selected_only
            if np.any(selected_mask):
                sel_idx = np.where(selected_mask)[0]
                ax.scatter(
                    X_2d[selected_mask, 0],
                    X_2d[selected_mask, 1],
                    c=[colors[i] for i in sel_idx],
                    s=selected_size,
                    alpha=selected_alpha,
                    edgecolors="none",
                    zorder=2,
                )
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=s, alpha=alpha, edgecolors="none", zorder=1)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title)
    return ax


# =========================================================================
# 2. Ellipse drawing
# =========================================================================


def ellipse_overlay(
    means: np.ndarray,
    covs: np.ndarray,
    ax: mpl.axes.Axes,
    *,
    labels: np.ndarray | None = None,
    palette: dict | str | None = None,
    color_mode: Literal["signal", "all", "none"] = "signal",
    mode_labels: Sequence[str] | None = None,
    n_signal: int = 0,
    diagonal_approx: bool = True,
    lw_signal: float = 3.0,
    lw_noise: float = 2.5,
    alpha_signal: float = 1.0,
    alpha_noise: float = 0.5,
    fill_alpha_signal: float = 0.2,
    fill_alpha_noise: float = 0.05,
    annotate: bool = True,
    annotation_fontsize: float = 16,
) -> mpl.axes.Axes:
    """Draw GMM covariance ellipses onto an existing axes.

    Parameters
    ----------
    means
        ``(K, 2)`` array of component means.
    covs
        ``(K, 2, 2)`` array of component covariances.
    ax
        Matplotlib axes to draw on.
    labels
        Per-component labels used for colouring components.
    palette
        Colour mapping — see :func:`_resolve_palette`.
    color_mode
        ``"signal"`` colours only signal modes, ``"all"`` colours every
        component, and ``"none"`` keeps all ellipses grey.
    mode_labels
        Optional per-component text annotations (e.g. math names).
    n_signal
        Number of leading components that are *signal* (the rest are noise).
    diagonal_approx
        If ``True``, overlay a dashed diagonal-approximation ellipse on
        signal modes.
    lw_signal, lw_noise
        Line widths for signal / noise ellipses.
    alpha_signal, alpha_noise
        Edge-colour alpha for signal / noise.
    fill_alpha_signal, fill_alpha_noise
        Face-colour alpha for signal / noise.
    annotate
        Whether to annotate signal modes with ``mode_labels``.
    annotation_fontsize
        Font size for mode labels.

    Returns
    -------
    matplotlib.axes.Axes
    """
    n_total = len(means)
    if color_mode not in {"signal", "all", "none"}:
        raise ValueError("`color_mode` must be one of {'signal', 'all', 'none'}.")

    use_label_colors = labels is not None and color_mode != "none"
    if use_label_colors:
        labels = np.asarray(labels)
        color_map = _resolve_palette(labels, palette)
    else:
        color_map = {}
    neutral_color = "dimgray"

    for k in range(n_total):
        is_signal = k < n_signal
        use_component_color = use_label_colors and (color_mode == "all" or (color_mode == "signal" and is_signal))
        if use_component_color:
            color = color_map.get(labels[k], neutral_color)
        else:
            color = neutral_color
        lw = lw_signal if is_signal else lw_noise
        alpha = alpha_signal if is_signal else alpha_noise
        fill_a = fill_alpha_signal if is_signal else fill_alpha_noise

        _draw_ellipse(
            means[k], covs[k], ax, color=color, linestyle="-", alpha=alpha, lw=lw, fill_alpha=fill_a, is_diag=False
        )

        if is_signal and diagonal_approx:
            _draw_ellipse(
                means[k],
                covs[k],
                ax,
                color=color,
                linestyle="--",
                alpha=alpha_noise,
                lw=lw_noise,
                fill_alpha=0.0,
                is_diag=True,
            )

        if is_signal and annotate and mode_labels is not None and k < len(mode_labels):
            txt = ax.text(
                means[k][0],
                means[k][1],
                mode_labels[k],
                color=color,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=annotation_fontsize,
                zorder=30,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3.0, foreground="white")])

    return ax


def _draw_ellipse(
    mean,
    cov,
    ax,
    *,
    color,
    linestyle="-",
    alpha=1.0,
    lw=1.5,
    fill_alpha=0.2,
    is_diag=False,
):
    """Draw a single 2-D Gaussian ellipse (±2σ)."""
    cov_draw = np.diag(np.diag(cov)) if is_diag else cov
    zorder = 20 if is_diag else 10

    eigvals, eigvecs = np.linalg.eig(cov_draw)
    eigvals = np.sqrt(np.real(eigvals))
    eigvecs = np.real(eigvecs)
    angle = np.rad2deg(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    ell = Ellipse(
        xy=mean,
        width=eigvals[0] * 4,
        height=eigvals[1] * 4,
        angle=angle,
        linestyle=linestyle,
        linewidth=lw,
        zorder=zorder,
    )
    ell.set_edgecolor(mcolors.to_rgba(color, alpha))
    if fill_alpha > 0 and not is_diag:
        ell.set_facecolor(mcolors.to_rgba(color, fill_alpha))
    else:
        ell.set_facecolor("none")
    ax.add_patch(ell)


# =========================================================================
# 3. Scatter plots
# =========================================================================


def scatter_subspace(
    X: np.ndarray,
    *,
    color_by: np.ndarray | None = None,
    style_by: np.ndarray | None = None,
    projection: np.ndarray | None = None,
    components: tuple[int, int] = (0, 1),
    palette: dict | str | None = None,
    ax: mpl.axes.Axes | None = None,
    s: float = 15,
    alpha: float = 0.3,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = True,
    show: bool | None = None,
    save: str | bool | None = None,
) -> mpl.axes.Axes:
    """2-D scatter plot of points in a subspace.

    Works with any point cloud: raw observations, AnnData-backed data,
    or projected samples.

    Parameters
    ----------
    X
        ``(N, D)`` point array.
    color_by
        Length-*N* array for hue (class label, mode name, …).
    style_by
        Length-*N* array for marker style (patient id, …).
    projection
        Optional ``(D, K)`` or ``(K, D)`` linear-projection matrix.
        When ``None``, ``components`` select columns from *X* directly.
    components
        Pair of component indices to plot (default ``(0, 1)``).
    palette
        Colour palette — ``dict``, colormap name, or ``None``.
    ax
        Existing axes; a new figure is created when ``None``.
    s
        Marker size.
    alpha
        Point opacity.
    xlabel, ylabel
        Axis labels.
    title
        Plot title.
    legend
        Whether to show a legend.
    show, save
        Passed to :func:`ggml_ot.plot._utils.savefig_or_show`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    X_2d = _resolve_projection(X, projection, components)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))

    if color_by is not None:
        color_by = np.asarray(color_by)
        cmap = _resolve_palette(color_by, palette)
        colors = [cmap[v] for v in color_by]
    else:
        colors = None

    if style_by is not None:
        style_by = np.asarray(style_by)
        unique_styles = np.unique(style_by)
        markers = ["o", "s", "D", "^", "v", "P", "<", ">", "X", "*"]
        style_to_marker = {s: markers[i % len(markers)] for i, s in enumerate(unique_styles)}
        for sval in unique_styles:
            mask = style_by == sval
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[colors[j] for j in np.where(mask)[0]] if colors else None,
                marker=style_to_marker[sval],
                s=s,
                alpha=alpha,
                edgecolors="none",
                label=str(sval),
            )
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=s, alpha=alpha, edgecolors="none")

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title)
    if legend and color_by is not None:
        cmap_dict = _resolve_palette(color_by, palette)
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap_dict[v], label=str(v), markersize=8)
            for v in sorted(cmap_dict.keys(), key=str)
        ]
        ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.95)

    savefig_or_show(ax, default_name="scatter_subspace", show=show, save=save)
    return ax


def scatter_3d(
    X: np.ndarray,
    *,
    color_by: np.ndarray | None = None,
    style_by: np.ndarray | None = None,
    components: tuple[int, int, int] = (0, 1, 2),
    palette: dict | str | None = None,
    ax: mpl.axes.Axes | None = None,
    s: float = 15,
    alpha: float = 0.4,
    elev: float = 25,
    azim: float = 130,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    zlabel: str = "Dim 3",
    title: str | None = None,
    legend: bool = True,
    show: bool | None = None,
    save: str | bool | None = None,
) -> mpl.axes.Axes:
    """3-D scatter plot.

    Parameters
    ----------
    X
        ``(N, D)`` point array (columns *components* are shown).
    color_by, style_by
        Arrays for hue and marker style.
    components
        Three column indices to plot.
    palette
        Colour mapping.
    ax
        Existing 3-D axes; creates new figure if ``None``.
    s, alpha
        Marker size and opacity.
    elev, azim
        Camera elevation and azimuth.
    xlabel, ylabel, zlabel
        Axis labels.
    title
        Plot title.
    legend
        Whether to show a legend.
    show, save
        Passed to ``savefig_or_show``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    c0, c1, c2 = components
    pts = X[:, [c0, c1, c2]]

    if ax is None:
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, projection="3d")

    color_by_arr = np.asarray(color_by) if color_by is not None else None
    cmap = _resolve_palette(color_by_arr, palette) if color_by_arr is not None else None

    if style_by is not None:
        style_by = np.asarray(style_by)
        unique_styles = np.unique(style_by)
        markers = ["o", "s", "D", "^", "v", "P", "<", ">", "X", "*"]
        style_to_marker = {sv: markers[i % len(markers)] for i, sv in enumerate(unique_styles)}
        for sv in unique_styles:
            mask = style_by == sv
            c = [cmap[v] for v in color_by_arr[mask]] if cmap else None
            ax.scatter(
                pts[mask, 0],
                pts[mask, 1],
                pts[mask, 2],
                c=c,
                marker=style_to_marker[sv],
                s=s,
                alpha=alpha,
                edgecolors="none",
            )
    else:
        c = [cmap[v] for v in color_by_arr] if cmap else None
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=s, alpha=alpha, edgecolors="none")

    ax.set_xlabel(xlabel, labelpad=-13, fontsize=12)
    ax.set_ylabel(ylabel, labelpad=-13, fontsize=12)
    ax.set_zlabel(zlabel, labelpad=-13, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title)

    if legend and cmap:
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[v], label=str(v), markersize=8)
            for v in sorted(cmap.keys(), key=str)
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2)

    savefig_or_show(ax, default_name="scatter_3d", show=show, save=save)
    return ax


# =========================================================================
# 4. Composite panel figure
# =========================================================================


def panel_subspaces(
    data: dict | None = None,
    *,
    adata=None,
    projection: np.ndarray | None = None,
    components: tuple[int, int] = (0, 1),
    palette: dict | str | None = None,
    color_key: str = "labels",
    mode_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (18, 5.5),
    point_alpha_3d: float = 0.4,
    point_alpha_2d: float = 0.15,
    elev: float = 25,
    azim: float = 130,
    show: bool | None = None,
    save: str | bool | None = None,
) -> mpl.figure.Figure:
    """Three-panel figure: mixed 3-D | rotated 2-D | clean 2-D (or learned).

    Accepts either the raw ``data`` dict from :func:`synth_gmm` or an
    ``AnnData`` produced by :func:`synth_gmm_anndata`.  When
    *projection* is supplied, the third panel shows the learned subspace
    instead of the ground-truth clean space.

    Parameters
    ----------
    data
        Dict returned by :func:`ggml_ot.data.synth_gmm`.
    adata
        AnnData from :func:`ggml_ot.data.synth_gmm_anndata` (alternative
        to ``data``).
    projection
        Learned ``(D, K)`` projection matrix.  When given, the third
        panel shows data projected through it.
    components
        Component indices for 2-D panels (default ``(0, 1)``).
    palette
        Colour mapping for class labels.
    color_key
        Which metadata to colour by (``"labels"``, ``"mode_names"``,
        ``"signal_noise"``, ``"patient_ids"``).
    mode_labels
        Math-mode labels for signal components (default auto).
    figsize
        Figure size.
    point_alpha_3d, point_alpha_2d
        Point opacities.
    elev, azim
        Camera angles for the 3-D panel.
    show, save
        Passed to ``savefig_or_show``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ------------------------------------------------------------------
    # Extract arrays from data dict or AnnData
    # ------------------------------------------------------------------
    if adata is not None:
        X_high = np.asarray(adata.X)
        X_rot = np.asarray(adata.obsm["X_rotated"])
        X_clean = np.asarray(adata.obsm["X_clean"])
        color_arr = np.asarray(
            adata.obs.get(
                {
                    "labels": "label",
                    "mode_names": "mode_name",
                    "signal_noise": "signal_noise",
                    "patient_ids": "sample",
                }.get(color_key, color_key)
            )
        )
        style_arr = np.asarray(adata.obs["sample"])
        means_rot = adata.uns.get("agg_means_rot")
        covs_rot = adata.uns.get("agg_covs_rot")
        means_2d = adata.uns.get("agg_means_2d")
        covs_2d = adata.uns.get("agg_covs_2d")
        n_noise = adata.uns.get("n_noise", 0)
    elif data is not None:
        samples = data["samples"]
        X_high = data["X_high_all"]
        X_rot = np.vstack(samples["rot"])
        X_clean = np.vstack(samples["clean"])
        color_arr = np.asarray(samples.get(color_key, samples["labels"]))
        style_arr = np.asarray(samples["patient_ids"])
        means_rot = data.get("agg_means_rot")
        covs_rot = data.get("agg_covs_rot")
        means_2d = data.get("agg_means_2d")
        covs_2d = data.get("agg_covs_2d")
        n_noise = data.get("n_noise", 0)
    else:
        raise ValueError("Provide either `data` (synth_gmm dict) or `adata`.")

    n_signal = len(means_2d) - n_noise if means_2d is not None else 0

    if palette is None and color_key == "labels":
        palette = _default_class_palette(color_arr)

    if mode_labels is None:
        mode_labels = [
            r"$\mathcal{X}_i$",
            r"$\mathcal{Y}_i$",
            r"$\mathcal{X}_j$",
            r"$\mathcal{Y}_j$",
        ]

    # Ellipse-level labels (class per component)
    if means_2d is not None:
        ell_labels = np.array([0 if k in [0, 2] else 1 for k in range(n_signal)] + [0] * n_noise)
    else:
        ell_labels = None

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)

    # Panel A – mixed 3-D
    ax_a = fig.add_subplot(131, projection="3d")
    scatter_3d(
        X_high,
        color_by=color_arr,
        style_by=style_arr,
        palette=palette,
        ax=ax_a,
        alpha=point_alpha_3d,
        elev=elev,
        azim=azim,
        show=False,
        save=False,
    )

    # Panel B – rotated 2-D
    ax_b = fig.add_subplot(132)
    scatter_subspace(
        X_rot,
        color_by=color_arr,
        components=components,
        palette=palette,
        ax=ax_b,
        alpha=point_alpha_2d,
        xlabel="Latent Dim 1",
        ylabel="Latent Dim 2",
        legend=False,
        show=False,
        save=False,
    )
    if means_rot is not None and covs_rot is not None:
        ellipse_overlay(
            means_rot,
            covs_rot,
            ax_b,
            labels=ell_labels,
            palette=palette,
            mode_labels=mode_labels,
            n_signal=n_signal,
        )
        x1, x2, y1, y2 = _get_tight_bounds(means_rot, covs_rot)
        ax_b.set_xlim(x1, x2)
        ax_b.set_ylim(y1, y2)
    ax_b.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax_b.axvline(0, color="k", linestyle=":", alpha=0.3)
    ax_b.set_xticks([])
    ax_b.set_yticks([])

    # Panel C – clean 2-D (or learned subspace)
    ax_c = fig.add_subplot(133)
    if projection is not None:
        scatter_subspace(
            X_high,
            color_by=color_arr,
            projection=projection,
            components=components,
            palette=palette,
            ax=ax_c,
            alpha=point_alpha_2d,
            xlabel="Learned Dim 1",
            ylabel="Learned Dim 2",
            legend=False,
            show=False,
            save=False,
        )
    else:
        scatter_subspace(
            X_clean,
            color_by=color_arr,
            components=components,
            palette=palette,
            ax=ax_c,
            alpha=point_alpha_2d,
            xlabel="Latent Dim 1",
            ylabel="Latent Dim 2",
            legend=False,
            show=False,
            save=False,
        )
        if means_2d is not None and covs_2d is not None:
            ellipse_overlay(
                means_2d,
                covs_2d,
                ax_c,
                labels=ell_labels,
                palette=palette,
                mode_labels=mode_labels,
                n_signal=n_signal,
            )
            x1, x2, y1, y2 = _get_tight_bounds(means_2d, covs_2d)
            ax_c.set_xlim(x1, x2)
            ax_c.set_ylim(y1, y2)
    ax_c.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax_c.axvline(0, color="k", linestyle=":", alpha=0.3)
    ax_c.set_xticks([])
    ax_c.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    shown = savefig_or_show(fig, default_name="panel_subspaces", show=show, save=save)
    return None if shown else fig


def _default_class_palette(labels: np.ndarray) -> dict:
    """Generate default red/blue palette for binary labels."""
    unique = sorted(np.unique(labels), key=str)
    default_colors = ["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42"]
    return {v: default_colors[i % len(default_colors)] for i, v in enumerate(unique)}


# =========================================================================
# 5. GMM projection helper
# =========================================================================


def _project_gmm(
    means: np.ndarray,
    covs: np.ndarray,
    projection: np.ndarray,
    components: tuple[int, int] = (0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """Project GMM parameters to 2-D via a linear map.

    Uses the row-vector convention ``x_proj = x @ P`` (same as
    :func:`_resolve_projection`):

    - Mean:  ``mu_2d = (mu @ P)[[c0, c1]]``
    - Cov:   ``Sig_2d = (P.T @ Sigma @ P)[[c0,c1],:][:,[c0,c1]]``

    Parameters
    ----------
    means
        ``(K, D)`` component means.
    covs
        ``(K, D, D)`` component covariances.
    projection
        ``(D, K_out)`` or ``(K_out, D)`` projection matrix.
        Auto-transposed when ``rows < cols`` (same heuristic as
        :func:`_resolve_projection`).  Pass a square orthogonal mixing
        matrix ``Q`` to invert the mixing transform.
    components
        Pair of projected-space column indices to return.

    Returns
    -------
    means_2d : np.ndarray, shape ``(K, 2)``
    covs_2d  : np.ndarray, shape ``(K, 2, 2)``
    """
    P = np.asarray(projection, dtype=np.float64)
    if P.shape[0] < P.shape[1]:
        P = P.T  # normalise to (D, K_out)

    c0, c1 = components
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)

    means_proj = means @ P  # (K, K_out)
    means_2d = means_proj[:, [c0, c1]]  # (K, 2)

    K = len(covs)
    covs_2d = np.empty((K, 2, 2), dtype=np.float64)
    for k in range(K):
        full = P.T @ covs[k] @ P  # (K_out, K_out)
        covs_2d[k] = full[np.ix_([c0, c1], [c0, c1])]

    return means_2d, covs_2d


# =========================================================================
# 6. Combined scatter + GMM panel
# =========================================================================


def plot_gmm_panel(
    X: np.ndarray,
    *,
    means: np.ndarray | None = None,
    covs: np.ndarray | None = None,
    projection: np.ndarray | None = None,
    components: tuple[int, int] = (0, 1),
    color_by: np.ndarray | None = None,
    palette: dict | str | None = None,
    ell_labels: np.ndarray | None = None,
    mode_labels: Sequence[str] | None = None,
    n_signal: int = 0,
    ax: mpl.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 5.5),
    point_alpha: float = 0.12,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
) -> mpl.axes.Axes:
    """2-D scatter with optional GMM ellipse overlay.

    Combines :func:`scatter_subspace`, :func:`_project_gmm`, and
    :func:`ellipse_overlay` into a single call.

    Parameters
    ----------
    X
        ``(N, D)`` point array.
    means
        ``(K, D)`` GMM component means in the same space as *X*.
        When ``None`` only the scatter is drawn.
    covs
        ``(K, D, D)`` GMM component covariances.
    projection
        Optional linear projection applied to both *X* and *means* / *covs*.
        Uses the same ``(D, K_out)`` / ``(K_out, D)`` auto-transpose logic
        as :func:`scatter_subspace`.  When ``None``, *components* select
        columns of *X* / *means* / *covs* directly (assumes they are already
        2-D or the first two dimensions carry the signal).
    components
        Pair of column indices to plot after projection.
    color_by
        Length-*N* array for hue.
    palette
        Colour mapping for points and (when *ell_labels* is given) ellipses.
    ell_labels
        ``(K,)`` class labels for ellipse colouring.
    mode_labels
        Annotation strings for the leading *n_signal* components.
    n_signal
        Number of leading components treated as signal (drawn more
        prominently by :func:`ellipse_overlay`).
    ax
        Existing axes; creates a new figure when ``None``.
    figsize
        Figure size when *ax* is ``None``.
    point_alpha
        Point opacity.
    title, xlabel, ylabel
        Axis decoration.
    show, save
        Passed to :func:`~ggml_ot.plot._utils.savefig_or_show`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    scatter_subspace(
        X,
        color_by=color_by,
        projection=projection,
        components=components,
        palette=palette,
        ax=ax,
        alpha=point_alpha,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        legend=(color_by is not None),
        show=False,
        save=False,
    )

    if means is not None and covs is not None:
        means_np = _to_np(means)
        covs_np = _to_np(covs)

        if projection is not None:
            means_2d, covs_2d = _project_gmm(means_np, covs_np, projection, components)
        else:
            c0, c1 = components
            if means_np.ndim == 2 and means_np.shape[1] == 2:
                means_2d, covs_2d = means_np, covs_np
            else:
                means_2d = means_np[:, [c0, c1]]
                covs_2d = covs_np[:, [c0, c1], :][:, :, [c0, c1]]

        ellipse_overlay(
            means_2d,
            covs_2d,
            ax,
            labels=ell_labels,
            palette=palette,
            mode_labels=mode_labels,
            n_signal=n_signal,
        )

        x1, x2, y1, y2 = _get_tight_bounds(means_2d, covs_2d)
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)

    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax.axvline(0, color="k", linestyle=":", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    savefig_or_show(ax, default_name="plot_gmm_panel", show=show, save=save)
    return ax


def latent_gmm(
    data,
    *,
    dims: tuple[int, int] = (0, 1),
    component_view: Literal["raw", "grouped"] = "raw",
    gmm_key: str | None = None,
    grouping_key: str | None = None,
    color_by: str | np.ndarray | None = None,
    palette: dict | str | None = None,
    grouping_method: Literal["mean", "bures-wasserstein"] = "mean",
    n_groups: int | None = None,
    group_representative: Literal["mean", "gaussian"] = "gaussian",
    barycenter_weighting: Literal["uniform", "component_weights"] = "component_weights",
    selected_groups: Sequence[int] | None = None,
    group_display: Literal["all", "highlight", "only"] = "all",
    selected_patients: Sequence[str] | None = None,
    point_view: Literal["all", "highlight_selected", "selected_only"] = "all",
    title: str | None = None,
    ax: mpl.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 5.5),
    show: bool | None = None,
    save: str | bool | None = None,
) -> mpl.axes.Axes:
    """Plot fitted GMM components in the learned GGML latent space."""
    if component_view not in {"raw", "grouped"}:
        raise ValueError("component_view must be 'raw' or 'grouped'.")
    if group_display not in {"all", "highlight", "only"}:
        raise ValueError("group_display must be 'all', 'highlight', or 'only'.")
    if point_view not in {"all", "highlight_selected", "selected_only"}:
        raise ValueError("point_view must be 'all', 'highlight_selected', or 'selected_only'.")
    if component_view != "grouped" and selected_groups is not None:
        raise ValueError("selected_groups is only supported when component_view='grouped'.")
    if component_view != "grouped" and group_display != "all":
        raise ValueError("group_display is only supported when component_view='grouped'.")
    if component_view == "grouped" and group_representative != "gaussian":
        raise ValueError(
            "Grouped latent GMM plots currently require group_representative='gaussian' "
            "so grouped ellipses have covariances."
        )

    adata, resolved_gmm_key = _resolve_latent_gmm_input(data, gmm_key)
    if "X_ggml" not in adata.obsm:
        raise KeyError("adata.obsm['X_ggml'] is required for latent_gmm plotting.")
    X_latent = np.asarray(adata.obsm["X_ggml"], dtype=np.float64)
    _validate_dims(dims=dims, latent_dim=int(X_latent.shape[1]))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    color_values = _resolve_latent_color_values(adata, data, color_by)
    patient_col = _resolve_patient_col_for_plot(adata, data)
    patient_ids = None if patient_col is None else adata.obs[patient_col].astype(str).to_numpy()
    if selected_patients is not None and patient_ids is not None:
        _validate_selected_patients(patient_ids, selected_patients)

    gmm_cfg = adata.uns[resolved_gmm_key]
    sharing = str(gmm_cfg.get("component_sharing", "sample_specific"))
    point_group_ids = None

    if component_view == "grouped":
        if sharing != "sample_specific":
            raise ValueError(
                "component_view='grouped' is only supported for sample_specific GMMs. "
                "# FUTURE: reinterpret grouped selection controls as raw component ids."
            )
        if patient_col is None:
            raise ValueError(
                "A patient column is required for component_view='grouped'. "
                "Pass selected_patients or ensure adata stores a patient column."
            )
        from ggml_ot.gene._grouping import _resolve_grouping_for_consumer

        grouping = _resolve_grouping_for_consumer(
            adata,
            resolved_gmm_key,
            grouping_method=grouping_method,
            n_groups=n_groups,
            group_representative=group_representative,
            barycenter_weighting=barycenter_weighting,
            grouping_key=grouping_key,
        )
        means = np.asarray(grouping["grouped_mu"], dtype=np.float64)
        covariances = np.asarray(grouping["grouped_var"], dtype=np.float64)
        n_groups_resolved = int(grouping["n_groups"])
        point_group_ids = _cell_group_assignments(
            adata,
            gmm_key=resolved_gmm_key,
            label_matrix=np.asarray(grouping["label_matrix"], dtype=int),
            distribution_ids=[str(x) for x in grouping["distribution_ids"]],
            patient_col=patient_col,
        )
        ellipse_labels = np.arange(n_groups_resolved, dtype=int)
        ellipse_palette = _resolve_group_palette(
            ellipse_labels=ellipse_labels,
            selected_groups=selected_groups,
            group_display=group_display,
        )
        if selected_groups is not None:
            selected_groups = np.asarray(selected_groups, dtype=int)
            _validate_selected_groups(selected_groups, n_groups=n_groups_resolved)
            if group_display == "only":
                keep = np.isin(ellipse_labels, selected_groups)
                means = means[keep]
                covariances = covariances[keep]
                ellipse_labels = ellipse_labels[keep]
    else:
        if sharing == "sample_specific":
            means, covariances, ellipse_labels = _raw_sample_specific_components(
                gmm_cfg=gmm_cfg,
                selected_patients=selected_patients,
            )
            ellipse_palette = None
        else:
            means, covariances = _raw_global_components(gmm_cfg)
            ellipse_labels = np.arange(means.shape[0], dtype=int)
            ellipse_palette = None

    projection = _latent_projection_for_gmm(adata, rep_dim=int(means.shape[-1]), latent_dim=int(X_latent.shape[1]))
    if projection is not None:
        means_2d, covariances_2d = _project_gmm(means, covariances, projection, dims)
    else:
        means_2d = np.asarray(means[:, list(dims)], dtype=np.float64)
        covariances_2d = np.asarray(
            covariances[:, list(dims), :][:, :, list(dims)],
            dtype=np.float64,
        )

    background_mask, emphasis_mask = _resolve_point_masks(
        patient_ids=patient_ids,
        selected_patients=selected_patients,
        point_view=point_view,
        point_group_ids=point_group_ids,
        selected_groups=selected_groups,
        group_display=group_display,
        n_points=int(X_latent.shape[0]),
    )
    _scatter_latent_points(
        ax,
        X_latent,
        dims=dims,
        color_values=color_values,
        palette=palette,
        background_mask=background_mask,
        emphasis_mask=emphasis_mask,
    )

    ellipse_overlay(
        means_2d,
        covariances_2d,
        ax,
        labels=ellipse_labels,
        palette=ellipse_palette,
        color_mode="all",
        n_signal=0,
        diagonal_approx=False,
        annotate=False,
    )

    x1, x2, y1, y2 = _get_tight_bounds(means_2d, covariances_2d)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax.axvline(0, color="k", linestyle=":", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

    savefig_or_show(ax, default_name="latent_gmm", show=show, save=save)
    return ax


def _resolve_latent_gmm_input(data, gmm_key: str | None) -> tuple:
    from ggml_ot.gene._gmm_summary import resolve_gmm_key
    from ggml_ot.gene._grouping import _resolve_adata

    adata = _resolve_adata(data)
    if isinstance(data, AnnData):
        use_rep = adata.uns.get("ggml_params", {}).get("use_rep")
    else:
        use_rep = getattr(data, "use_rep", None)
    return adata, resolve_gmm_key(adata, gmm_key, use_rep)


def _validate_dims(*, dims: tuple[int, int], latent_dim: int) -> None:
    if len(dims) != 2:
        raise ValueError(f"dims must contain exactly two latent dimensions. Got {dims!r}.")
    if dims[0] == dims[1]:
        raise ValueError(f"dims entries must be distinct. Got {dims!r}.")
    for dim in dims:
        if dim < 0 or dim >= latent_dim:
            raise ValueError(f"Latent dimension {dim} is out of range for X_ggml with shape (_, {latent_dim}).")


def _resolve_latent_color_values(adata, data, color_by):
    if color_by is None:
        label_col = getattr(data, "label_col", None)
        if label_col is None:
            label_col = adata.uns.get("ggml_params", {}).get("label_col")
        if label_col is None:
            return None
        if label_col not in adata.obs.columns:
            raise KeyError(f"Default label column {label_col!r} not found in adata.obs.")
        return np.asarray(adata.obs[label_col])
    if isinstance(color_by, str):
        if color_by not in adata.obs.columns:
            raise KeyError(f"Column {color_by!r} not found in adata.obs.")
        return np.asarray(adata.obs[color_by])
    values = np.asarray(color_by)
    if values.shape[0] != adata.n_obs:
        raise ValueError(f"color_by array must have length adata.n_obs={adata.n_obs}. Got shape={values.shape}.")
    return values


def _resolve_patient_col_for_plot(adata, data) -> str | None:
    patient_col = getattr(data, "patient_col", None)
    if patient_col is None:
        patient_col = adata.uns.get("ggml_params", {}).get("patient_col")
    if patient_col is not None and patient_col in adata.obs.columns:
        return str(patient_col)
    for candidate in ("sample", "patient"):
        if candidate in adata.obs.columns:
            return candidate
    return None


def _validate_selected_patients(patient_ids: np.ndarray, selected_patients: Sequence[str]) -> None:
    available = set(np.asarray(patient_ids, dtype=str).tolist())
    missing = [str(pid) for pid in selected_patients if str(pid) not in available]
    if missing:
        raise ValueError(f"selected_patients contains unknown ids: {missing}.")


def _raw_global_components(gmm_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(gmm_cfg["model"]["mu"], dtype=np.float64)
    covariances = canonicalize_covariances(
        np.asarray(gmm_cfg["model"]["var"], dtype=np.float64),
        means,
        covariance_type=str(gmm_cfg.get("covariance_type", "full")),
    )
    if means.ndim == 3 and means.shape[0] == 1:
        means = means[0]
    if covariances.ndim == 4 and covariances.shape[0] == 1:
        covariances = covariances[0]
    return means, covariances


def _raw_sample_specific_components(
    *,
    gmm_cfg: dict,
    selected_patients: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray(gmm_cfg["model"]["mu"], dtype=np.float64)
    covariances = canonicalize_covariances(
        np.asarray(gmm_cfg["model"]["var"], dtype=np.float64),
        means,
        covariance_type=str(gmm_cfg.get("covariance_type", "full")),
    )
    distribution_ids = [str(x) for x in gmm_cfg.get("weight_inference", {}).get("distribution_ids", [])]
    distribution_n_components = np.asarray(
        gmm_cfg.get("distribution_n_components", np.full(means.shape[0], means.shape[1])),
        dtype=int,
    )
    selected_set = None if selected_patients is None else {str(x) for x in selected_patients}

    flat_means = []
    flat_covariances = []
    labels = []
    for dist_idx, dist_id in enumerate(distribution_ids):
        if selected_set is not None and dist_id not in selected_set:
            continue
        n_active = int(distribution_n_components[dist_idx])
        for comp_idx in range(n_active):
            flat_means.append(means[dist_idx, comp_idx])
            flat_covariances.append(covariances[dist_idx, comp_idx])
            labels.append(dist_id)
    if len(flat_means) == 0:
        raise ValueError("No sample-specific components selected for plotting.")
    return (
        np.asarray(flat_means, dtype=np.float64),
        np.asarray(flat_covariances, dtype=np.float64),
        np.asarray(labels, dtype=object),
    )


def _latent_projection_for_gmm(
    adata,
    *,
    rep_dim: int,
    latent_dim: int,
) -> np.ndarray | None:
    if rep_dim == latent_dim:
        return None
    if "W_ggml" in adata.varm:
        gene_loadings = np.asarray(adata.varm["W_ggml"], dtype=np.float64)
        if gene_loadings.ndim == 2 and gene_loadings.shape[0] == rep_dim:
            return gene_loadings
    if "W_ggml" not in adata.uns:
        raise ValueError(
            "Cannot project GMM components into the GGML latent space because adata.uns['W_ggml'] is missing."
        )
    map_A = np.asarray(adata.uns["W_ggml"], dtype=np.float64)
    if map_A.ndim != 2:
        raise ValueError(f"Expected adata.uns['W_ggml'] to be a 2-D matrix. Got shape={map_A.shape}.")
    if map_A.shape[1] == rep_dim:
        return map_A.T
    if map_A.shape[0] == rep_dim:
        return map_A
    raise ValueError(
        "Stored GGML projection does not align with the fitted GMM representation. "
        f"rep_dim={rep_dim}, W_ggml shape={map_A.shape}."
    )


def _cell_group_assignments(
    adata,
    *,
    gmm_key: str,
    label_matrix: np.ndarray,
    distribution_ids: list[str],
    patient_col: str | None,
) -> np.ndarray:
    if patient_col is None:
        raise KeyError("A patient column is required to map cells onto grouped components.")
    patient_values = adata.obs[patient_col].astype(str).to_numpy()
    comp_key = f"{gmm_key}_comp"
    if comp_key in adata.obs.columns:
        local_components = np.asarray(adata.obs[comp_key], dtype=int)
    else:
        resp_key = f"{gmm_key}_resp"
        if resp_key not in adata.obsm:
            raise KeyError(
                f"Neither hard assignments adata.obs[{comp_key!r}] nor responsibilities "
                f"adata.obsm[{resp_key!r}] are available."
            )
        local_components = np.asarray(adata.obsm[resp_key], dtype=np.float64).argmax(axis=1)

    dist_to_idx = {str(dist_id): idx for idx, dist_id in enumerate(distribution_ids)}
    group_ids = np.full(adata.n_obs, -1, dtype=int)
    for cell_idx, patient_id in enumerate(patient_values):
        dist_idx = dist_to_idx.get(str(patient_id))
        if dist_idx is None:
            continue
        comp_idx = int(local_components[cell_idx])
        if 0 <= comp_idx < label_matrix.shape[1]:
            group_ids[cell_idx] = int(label_matrix[dist_idx, comp_idx])
    return group_ids


def _resolve_group_palette(
    *,
    ellipse_labels: np.ndarray,
    selected_groups: Sequence[int] | None,
    group_display: str,
) -> dict | None:
    if selected_groups is None or group_display == "all":
        return None
    selected_set = {int(x) for x in selected_groups}
    palette = _resolve_palette(np.asarray(ellipse_labels), None)
    return {
        int(label): palette[int(label)] if int(label) in selected_set else "dimgray"
        for label in np.asarray(ellipse_labels, dtype=int).tolist()
    }


def _validate_selected_groups(selected_groups: np.ndarray, *, n_groups: int) -> None:
    if selected_groups.ndim != 1:
        raise ValueError("selected_groups must be a 1-D sequence of group ids.")
    missing = [int(group) for group in selected_groups.tolist() if int(group) < 0 or int(group) >= n_groups]
    if missing:
        raise ValueError(f"selected_groups contains ids outside [0, {n_groups - 1}]: {missing}.")


def _resolve_point_masks(
    *,
    patient_ids: np.ndarray | None,
    selected_patients: Sequence[str] | None,
    point_view: str,
    point_group_ids: np.ndarray | None,
    selected_groups: Sequence[int] | None,
    group_display: str,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    visible_mask = np.ones(n_points, dtype=bool)
    highlight_constraints = []

    if selected_patients is not None and patient_ids is not None:
        patient_mask = np.isin(np.asarray(patient_ids, dtype=str), np.asarray(selected_patients, dtype=str))
        if point_view == "selected_only":
            visible_mask &= patient_mask
        elif point_view == "highlight_selected":
            highlight_constraints.append(patient_mask)

    if selected_groups is not None and point_group_ids is not None:
        group_mask = np.isin(point_group_ids, np.asarray(selected_groups, dtype=int))
        if group_display == "only":
            visible_mask &= group_mask
        elif group_display == "highlight":
            highlight_constraints.append(group_mask)

    if len(highlight_constraints) > 0:
        emphasis_mask = visible_mask & np.logical_and.reduce(highlight_constraints)
    else:
        emphasis_mask = visible_mask
    background_mask = visible_mask & ~emphasis_mask
    return background_mask, emphasis_mask


def _scatter_latent_points(
    ax: mpl.axes.Axes,
    X_latent: np.ndarray,
    *,
    dims: tuple[int, int],
    color_values: np.ndarray | None,
    palette: dict | str | None,
    background_mask: np.ndarray,
    emphasis_mask: np.ndarray,
    s: float = 15.0,
    point_alpha: float = 0.18,
    background_alpha: float = 0.04,
) -> None:
    X_2d = _resolve_projection(np.asarray(X_latent, dtype=np.float64), None, dims)
    if color_values is None:
        colors = np.asarray(["dimgray"] * X_2d.shape[0], dtype=object)
    else:
        cmap = _resolve_palette(np.asarray(color_values), palette)
        colors = np.asarray([cmap[value] for value in np.asarray(color_values)], dtype=object)

    if np.any(background_mask):
        background_idx = np.where(background_mask)[0]
        ax.scatter(
            X_2d[background_mask, 0],
            X_2d[background_mask, 1],
            c=[colors[idx] for idx in background_idx],
            s=s,
            alpha=background_alpha,
            edgecolors="none",
            zorder=1,
        )
    if np.any(emphasis_mask):
        emphasis_idx = np.where(emphasis_mask)[0]
        ax.scatter(
            X_2d[emphasis_mask, 0],
            X_2d[emphasis_mask, 1],
            c=[colors[idx] for idx in emphasis_idx],
            s=s,
            alpha=point_alpha,
            edgecolors="none",
            zorder=2,
        )


# =========================================================================
# 7. Dataset-level wrapper for synthetic GMM datasets
# =========================================================================


def panel_synth_dataset(
    dataset,
    *,
    palette: dict | str | None = None,
    color_key: str = "labels",
    point_alpha: float = 0.10,
    figsize: tuple[float, float] | None = None,
    unmixed_components: tuple[int, int] = (0, 1),
    show_learned_panel: bool = True,
    fitted_gmm_view: Literal["auto", "selected_patients", "class_average", "all_patients"] = "auto",
    mode_coloring: Literal["signal", "patients", "none"] = "signal",
    selected_patient_ids: Sequence[int] | None = None,
    point_view: Literal["all", "highlight_selected", "selected_only"] = "all",
    point_alpha_background: float = 0.05,
    show: bool | None = None,
    save: str | bool | None = None,
) -> mpl.figure.Figure:
    """Multi-panel figure for a synthetic-GMM ``TripletDataset``.

    Reads generation metadata from ``dataset.synth_data`` (set when the
    dataset was created via :func:`ggml_ot.data.from_synth_gmm`) and
    automatically produces up to three panels:

    * **Panel 1** – Signal subspace (before mixing). Uses the aggregate
      ground-truth GMM for global/class-average views and the final clean-space
      patient-specific GMMs for per-patient views. Always shown.
    * **Panel 2** – Unmixed subspace (selected via ``unmixed_components``)
      with fitted GMMs projected back by undoing ``Q_mixing`` and the
      synthetic signal-plane rotation ``R_rotation``.
      Shown only when ``dataset.covariances`` contains a fitted GMM.
    * **Panel 3** – Learned latent space with the same fitted GMMs.
      Shown only after :func:`ggml_ot.train` has been called and
      ``show_learned_panel=True``.

    Parameters
    ----------
    dataset
        A :class:`~ggml_ot.data.TripletDataset` with a ``synth_data``
        attribute (i.e. created via :func:`ggml_ot.data.from_synth_gmm`).
    palette
        Colour mapping for class labels.
    color_key
        Metadata array in ``dataset.synth_data["samples"]`` used for point
        hue (default ``"labels"``).
    point_alpha
        Opacity for scatter points.
    figsize
        Figure size; auto-set from number of panels when ``None``.
    unmixed_components
        Pair of dimensions in the clean synthetic basis obtained after undoing
        ``Q_mixing`` and ``R_rotation``. Use ``(0, 1)`` for signal dimensions
        (default) or e.g. ``(2, 3)`` for two noise dimensions.
    show_learned_panel
        Whether to include panel 3 when a learned projection is available.
        Set ``False`` to compare only unmixed dimensions (e.g. signal or
        noise) without the learned latent panel.
    fitted_gmm_view
        How fitted GMMs are shown when ``identical_supports=False``:
        ``"auto"`` (default) – uses ``"selected_patients"`` for per-patient
        GMMs and ``"class_average"`` for globally-fitted GMMs;
        ``"class_average"``, ``"selected_patients"``, or ``"all_patients"``.
    mode_coloring
        Colour strategy for fitted GMM overlays in panels 2/3:
        ``"signal"`` colours only signal modes when their identity is known;
        ``"patients"`` colours every fitted component by patient id using
        patient colours derived from the class palette; and
        ``"none"`` draws all fitted components in grey.
    selected_patient_ids
        Patient ids to use when ``fitted_gmm_view="selected_patients"``.
        If ``None``, one patient per class is selected automatically.
    point_view
        How points are shown when selected patients are active:
        ``"all"``, ``"highlight_selected"``, or ``"selected_only"``.
    point_alpha_background
        Background alpha used by ``point_view="highlight_selected"``.
    show, save
        Passed to :func:`~ggml_ot.plot._utils.savefig_or_show`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if fitted_gmm_view not in {"auto", "selected_patients", "class_average", "all_patients"}:
        raise ValueError(
            "`fitted_gmm_view` must be one of {'auto', 'selected_patients', 'class_average', 'all_patients'}."
        )
    if mode_coloring not in {"signal", "patients", "none"}:
        raise ValueError("`mode_coloring` must be one of {'signal', 'patients', 'none'}.")
    # Resolve "auto": per-patient GMMs → selected_patients; global → class_average
    if fitted_gmm_view == "auto":
        _has_covariances = not_none(dataset.covariances) if hasattr(dataset, "covariances") else False
        _per_patient = _has_covariances and not getattr(dataset, "identical_supports", True)
        fitted_gmm_view = "selected_patients" if _per_patient else "class_average"
    if point_view not in {"all", "highlight_selected", "selected_only"}:
        raise ValueError("`point_view` must be one of {'all', 'highlight_selected', 'selected_only'}.")

    sd = dataset.synth_data
    samples = sd["samples"]
    clean_unmix_projection = _synth_clean_unmix_projection(sd)

    X_high = np.asarray(sd["X_high_all"])
    X_unmixed = X_high @ clean_unmix_projection
    X_clean = np.vstack(samples["clean"])
    color_arr = np.array(samples.get(color_key, samples["labels"]))
    patient_ids_arr = np.asarray(samples.get("patient_ids", np.arange(len(color_arr))))

    n_dim = X_high.shape[1]
    c0_unmix, c1_unmix = unmixed_components
    if c0_unmix == c1_unmix:
        raise ValueError("`unmixed_components` must contain two distinct indices.")
    if not (0 <= c0_unmix < n_dim and 0 <= c1_unmix < n_dim):
        raise ValueError(f"`unmixed_components` {unmixed_components} out of bounds for n_dim={n_dim}.")

    color_scheme = _build_synth_color_scheme(
        sample_labels=np.asarray(samples["labels"]),
        sample_patient_ids=patient_ids_arr,
        palette=palette,
    )
    point_palette = _resolve_synth_point_palette(
        color_key=color_key,
        color_values=color_arr,
        color_scheme=color_scheme,
        palette=palette,
    )

    # ── GT ellipse metadata ───────────────────────────────────────────────
    n_noise = sd["n_noise"]
    gt_means = sd["agg_means_2d"]
    gt_covs = sd["agg_covs_2d"]
    n_signal = len(gt_means) - n_noise
    ell_labels_gt = np.array([0 if k in [0, 2] else 1 for k in range(n_signal)] + [0] * n_noise)
    mode_labels_gt = [
        r"$\mathcal{X}_i$",
        r"$\mathcal{Y}_i$",
        r"$\mathcal{X}_j$",
        r"$\mathcal{Y}_j$",
    ]

    # ── Detect dataset state ──────────────────────────────────────────────
    has_gmm = not_none(dataset.covariances)
    has_learned = getattr(dataset, "_map_A", None) is not None
    show_panel3 = has_learned and show_learned_panel

    if unmixed_components == (0, 1):
        panel2_subspace = "Signal Subspace (after unmixing)"
        panel2_xlabel = "Signal Dim 1"
        panel2_ylabel = "Signal Dim 2"
    elif c0_unmix >= 2 and c1_unmix >= 2:
        panel2_subspace = "Noise Subspace (after unmixing)"
        panel2_xlabel = "Noise Dim 1"
        panel2_ylabel = "Noise Dim 2"
    else:
        panel2_subspace = f"Unmixed Subspace ({c0_unmix}, {c1_unmix})"
        panel2_xlabel = "Unmixed Dim 1"
        panel2_ylabel = "Unmixed Dim 2"

    n_panels = 1 + int(has_gmm) + int(show_panel3)
    if figsize is None:
        figsize = (6.5 * n_panels, 5.5)

    fig, axes_arr = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes_arr = [axes_arr]

    # ── Build fitted-GMM overlays and optional patient focus ──────────────
    means_hd = covs_hd = None
    fit_color_labels = None
    fit_color_palette = None
    fit_color_mode: Literal["signal", "all", "none"] = "none"
    n_signal_fit = 0
    focus_patient_ids: list[int] | None = None
    focus_patient_note = None
    panel_gmm_line = "Patient GMMs"
    panel1_means = gt_means
    panel1_covs = gt_covs
    panel1_labels = ell_labels_gt
    panel1_palette = color_scheme.class_palette
    panel1_color_mode: Literal["signal", "all", "none"] = "signal"
    panel1_n_signal = n_signal
    panel1_mode_labels: Sequence[str] | None = mode_labels_gt
    panel1_model_line = "Ground-truth GMM"

    if has_gmm:
        sup = _to_np(dataset.supports)
        cov = _to_np(dataset.covariances)

        if dataset.identical_supports:
            if mode_coloring == "patients":
                raise ValueError("`mode_coloring='patients'` requires per-patient fitted GMMs.")
            means_hd = sup  # (K, D)
            covs_hd = cov  # (K, D, D)
            # For shared/global refits there is no reliable mapping from fitted
            # components to synthetic mode identities.
            n_signal_fit = 0
            panel_gmm_line = _synth_gmm_view_label(
                provenance="unknown",
                fitted_gmm_view="class_average",
            )
        else:
            dist_labels = np.asarray(dataset.distribution_labels)
            n_dist = sup.shape[0]
            unique_patient_ids = np.unique(patient_ids_arr)
            # Determine integer patient ID for each distribution *in distribution order*.
            # AnnData_TripletDataset sorts patients alphabetically by name (e.g.
            # "patient_0", "patient_1", "patient_10", ...) which differs from
            # numerical order when n_patients >= 10. Derive ordering from the adata.
            if hasattr(dataset, "adata") and dataset.adata is not None and "sample" in dataset.adata.obs.columns:
                try:
                    _pnames = sorted(dataset.adata.obs["sample"].unique())
                    dist_patient_ids = np.array([int(p.split("_")[-1]) for p in _pnames])
                except Exception:
                    dist_patient_ids = unique_patient_ids if unique_patient_ids.size == n_dist else np.arange(n_dist)
            elif "patient_gmms" in sd and len(sd["patient_gmms"]) == n_dist:
                dist_patient_ids = np.array([pg.get("patient_id", i) for i, pg in enumerate(sd["patient_gmms"])])
            elif unique_patient_ids.size == n_dist:
                dist_patient_ids = unique_patient_ids
            else:
                dist_patient_ids = np.arange(n_dist)
            fitted_gmm_provenance = _resolve_fitted_gmm_provenance(
                dataset=dataset,
                synth_data=sd,
                supports=sup,
                covariances=cov,
                dist_patient_ids=dist_patient_ids,
            )
            signal_mode_identity_known = fitted_gmm_provenance == "synthetic_ground_truth"

            if fitted_gmm_view == "class_average":
                if mode_coloring == "patients":
                    raise ValueError("`mode_coloring='patients'` is not defined for `fitted_gmm_view='class_average'`.")
                unique_cls = sorted(np.unique(dist_labels), key=str)
                K_pp = sup.shape[1]
                n_sig_pp = max(0, K_pp - n_noise)
                cls_means, cls_covs = {}, {}
                for cls in unique_cls:
                    mask = dist_labels == cls
                    cls_means[cls] = sup[mask].mean(axis=0)
                    cls_covs[cls] = cov[mask].mean(axis=0)

                sig_m = np.concatenate([cls_means[c][:n_sig_pp] for c in unique_cls], axis=0)
                sig_c = np.concatenate([cls_covs[c][:n_sig_pp] for c in unique_cls], axis=0)
                noise_m = np.mean([cls_means[c][n_sig_pp:] for c in unique_cls], axis=0)
                noise_c = np.mean([cls_covs[c][n_sig_pp:] for c in unique_cls], axis=0)

                means_hd = np.concatenate([sig_m, noise_m], axis=0)
                covs_hd = np.concatenate([sig_c, noise_c], axis=0)
                n_signal_fit_total = len(unique_cls) * n_sig_pp
                n_signal_fit = n_signal_fit_total if signal_mode_identity_known else 0
                if signal_mode_identity_known:
                    fit_color_labels = np.array(
                        [cls for cls in unique_cls for _ in range(n_sig_pp)] + [unique_cls[0]] * n_noise
                    )
                if mode_coloring == "signal" and fit_color_labels is not None:
                    fit_color_palette = color_scheme.class_palette
                    fit_color_mode = "signal"
                else:
                    fit_color_labels = None
                panel_gmm_line = _synth_gmm_view_label(
                    provenance=fitted_gmm_provenance,
                    fitted_gmm_view="class_average",
                )
            else:
                if fitted_gmm_view == "all_patients":
                    selected_dist_idx = list(range(n_dist))
                else:
                    if selected_patient_ids is None:
                        selected_dist_idx = []
                        for cls in sorted(np.unique(dist_labels), key=str):
                            cls_idx = np.where(dist_labels == cls)[0]
                            if cls_idx.size > 0:
                                selected_dist_idx.append(int(cls_idx[0]))
                    else:
                        pid_to_idx = {pid: i for i, pid in enumerate(dist_patient_ids.tolist())}
                        selected_dist_idx = []
                        for pid in selected_patient_ids:
                            if pid not in pid_to_idx:
                                raise ValueError(
                                    f"Requested patient_id={pid!r} not found in available ids "
                                    f"{dist_patient_ids.tolist()}."
                                )
                            idx = pid_to_idx[pid]
                            if idx not in selected_dist_idx:
                                selected_dist_idx.append(idx)
                    if len(selected_dist_idx) == 0:
                        raise ValueError("No patients selected for fitted GMM visualisation.")

                K_pp = sup.shape[1]
                n_sig_pp = max(0, K_pp - n_noise)
                sig_m, sig_c, noise_m, noise_c = [], [], [], []
                signal_class_labels, noise_class_labels = [], []
                signal_patient_labels, noise_patient_labels = [], []
                for idx in selected_dist_idx:
                    lbl = dist_labels[idx]
                    patient_id = int(dist_patient_ids[idx])
                    if n_sig_pp > 0:
                        sig_m.append(sup[idx][:n_sig_pp])
                        sig_c.append(cov[idx][:n_sig_pp])
                        signal_class_labels.extend([lbl] * n_sig_pp)
                        signal_patient_labels.extend([patient_id] * n_sig_pp)
                    if n_noise > 0:
                        noise_m.append(sup[idx][n_sig_pp : n_sig_pp + n_noise])
                        noise_c.append(cov[idx][n_sig_pp : n_sig_pp + n_noise])
                        noise_class_labels.extend([lbl] * n_noise)
                        noise_patient_labels.extend([patient_id] * n_noise)

                means_blocks, cov_blocks = [], []
                if len(sig_m) > 0:
                    means_blocks.append(np.concatenate(sig_m, axis=0))
                    cov_blocks.append(np.concatenate(sig_c, axis=0))
                if len(noise_m) > 0:
                    means_blocks.append(np.concatenate(noise_m, axis=0))
                    cov_blocks.append(np.concatenate(noise_c, axis=0))
                if len(means_blocks) == 0:
                    raise ValueError("Selected fitted GMM view produced no components to plot.")

                means_hd = np.concatenate(means_blocks, axis=0)
                covs_hd = np.concatenate(cov_blocks, axis=0)
                n_signal_fit_total = len(signal_class_labels)
                n_signal_fit = n_signal_fit_total if signal_mode_identity_known else 0
                if mode_coloring == "patients":
                    fit_color_labels = np.array(signal_patient_labels + noise_patient_labels)
                    fit_color_palette = color_scheme.patient_palette
                    fit_color_mode = "all"
                elif signal_mode_identity_known and (len(signal_class_labels) + len(noise_class_labels) > 0):
                    fit_color_labels = np.array(signal_class_labels + noise_class_labels)
                    fit_color_palette = color_scheme.class_palette
                    fit_color_mode = "signal"
                else:
                    fit_color_labels = None

                selected_ids = dist_patient_ids[selected_dist_idx].tolist()
                panel_gmm_line = _synth_gmm_view_label(
                    provenance=fitted_gmm_provenance,
                    fitted_gmm_view=fitted_gmm_view,
                )
                if fitted_gmm_view == "selected_patients":
                    focus_patient_ids = selected_ids
                    focus_patient_note = ", ".join([f"p{pid}" for pid in selected_ids])

            if fitted_gmm_view in {"selected_patients", "all_patients"}:
                panel1_means, panel1_covs = _project_gmm(
                    means_hd,
                    covs_hd,
                    clean_unmix_projection,
                    components=(0, 1),
                )
                panel1_labels = fit_color_labels
                panel1_palette = fit_color_palette
                panel1_color_mode = fit_color_mode
                panel1_n_signal = n_signal_fit
                panel1_mode_labels = None
                if fitted_gmm_view == "selected_patients":
                    panel1_model_line = _synth_gmm_view_label(
                        provenance=fitted_gmm_provenance,
                        fitted_gmm_view="selected_patients",
                    )
                else:
                    panel1_model_line = _synth_gmm_view_label(
                        provenance=fitted_gmm_provenance,
                        fitted_gmm_view="all_patients",
                    )

    panel_idx = 0

    # ── Panel 1: signal subspace before mixing ────────────────────────────
    ax = axes_arr[panel_idx]
    panel_idx += 1
    _scatter_subspace_with_patient_focus(
        X_clean,
        color_by=color_arr,
        patient_ids=patient_ids_arr,
        selected_patient_ids=focus_patient_ids,
        point_view=point_view,
        palette=point_palette,
        ax=ax,
        alpha=point_alpha,
        background_alpha=point_alpha_background,
        xlabel="Signal Dim 1",
        ylabel="Signal Dim 2",
        title=f"Signal Subspace (before mixing)\n{panel1_model_line}",
    )
    ellipse_overlay(
        panel1_means,
        panel1_covs,
        ax,
        labels=panel1_labels,
        palette=panel1_palette,
        color_mode=panel1_color_mode,
        n_signal=panel1_n_signal,
        mode_labels=panel1_mode_labels,
    )
    x1, x2, y1, y2 = _get_tight_bounds(panel1_means, panel1_covs)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax.axvline(0, color="k", linestyle=":", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    # ── Panel 2: selected unmixed subspace with fitted GMM model ──────────
    if has_gmm:
        ax = axes_arr[panel_idx]
        panel_idx += 1
        means_2d_fit, covs_2d_fit = _project_gmm(
            means_hd,
            covs_hd,
            clean_unmix_projection,
            components=unmixed_components,
        )
        _scatter_subspace_with_patient_focus(
            X_unmixed,
            color_by=color_arr,
            patient_ids=patient_ids_arr,
            selected_patient_ids=focus_patient_ids,
            point_view=point_view,
            components=unmixed_components,
            palette=point_palette,
            ax=ax,
            alpha=point_alpha,
            background_alpha=point_alpha_background,
            xlabel=panel2_xlabel,
            ylabel=panel2_ylabel,
            title=f"{panel2_subspace}\n{panel_gmm_line}",
        )
        ellipse_overlay(
            means_2d_fit,
            covs_2d_fit,
            ax,
            labels=fit_color_labels,
            palette=fit_color_palette,
            color_mode=fit_color_mode,
            n_signal=n_signal_fit,
            lw_noise=2.0,
            alpha_noise=0.7,
            fill_alpha_noise=0.08,
        )
        x1, x2, y1, y2 = _get_tight_bounds(means_2d_fit, covs_2d_fit)
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.axhline(0, color="k", linestyle=":", alpha=0.3)
        ax.axvline(0, color="k", linestyle=":", alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        if focus_patient_note is not None and point_view != "all":
            ax.text(
                0.02,
                0.98,
                f"Highlighted cells: {focus_patient_note}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    # ── Panel 3: learned latent space with fitted GMM model ───────────────
    if show_panel3:
        ax = axes_arr[panel_idx]
        panel_idx += 1
        W = _to_np(dataset._map_A)  # (n_comps, D)
        P = W.T  # (D, n_comps)
        _scatter_subspace_with_patient_focus(
            X_high,
            color_by=color_arr,
            patient_ids=patient_ids_arr,
            selected_patient_ids=focus_patient_ids,
            point_view=point_view,
            projection=P,
            components=(0, 1),
            palette=point_palette,
            ax=ax,
            alpha=point_alpha,
            background_alpha=point_alpha_background,
            xlabel="Latent Dim 1",
            ylabel="Latent Dim 2",
            title=f"Learned Latent Space\n{panel_gmm_line}",
        )
        if has_gmm:
            means_2d_w, covs_2d_w = _project_gmm(means_hd, covs_hd, P)
            ellipse_overlay(
                means_2d_w,
                covs_2d_w,
                ax,
                labels=fit_color_labels,
                palette=fit_color_palette,
                color_mode=fit_color_mode,
                n_signal=n_signal_fit,
                lw_noise=2.0,
                alpha_noise=0.7,
                fill_alpha_noise=0.08,
            )
        ax.axhline(0, color="k", linestyle=":", alpha=0.3)
        ax.axvline(0, color="k", linestyle=":", alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        if focus_patient_note is not None and point_view != "all":
            ax.text(
                0.02,
                0.98,
                f"Highlighted cells: {focus_patient_note}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    shown = savefig_or_show(fig, default_name="panel_synth_dataset", show=show, save=save)
    return None if shown else fig
