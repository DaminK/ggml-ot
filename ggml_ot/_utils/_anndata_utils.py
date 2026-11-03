"""AnnData read / write helpers.

All functions that touch ``adata.obs``, ``adata.obsm``, ``adata.X``, etc.
are centralised here so that the rest of the package never accesses AnnData
internals directly.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Column / index helpers
# ---------------------------------------------------------------------------


def validate_anndata_inputs(
    adata, patient_col: str, label_col: str, use_rep: str | None, layer: str | None = None
) -> None:
    """Assert that required columns and representations exist in *adata*."""
    if patient_col not in adata.obs:
        raise ValueError(f"patient_col {patient_col} not in adata.obs columns {adata.obs.columns}")
    if label_col not in adata.obs:
        raise ValueError(f"label_col {label_col} not in adata.obs columns {adata.obs.columns}")
    if use_rep is not None and use_rep not in adata.obsm:
        raise ValueError(f"use_rep {use_rep} not in adata.obsm keys {adata.obsm.keys()}")
    if layer is not None:
        if use_rep is not None:
            raise ValueError(f"Only one of use_rep ({use_rep}) and layer ({layer}) can be provided.")
        if layer not in adata.layers:
            raise ValueError(f"layer {layer} not in adata.layers keys {adata.layers.keys()}")


def get_obs_matrix_key(use_rep: str | None, layer: str | None = None) -> str:
    """Return the key of the data matrix selected by :func:`get_obs_matrix`, e.g. for ``gmm_<key>`` names."""
    if use_rep is not None:
        return use_rep
    if layer is not None:
        return f"layer_{layer}"
    return "X"


def label_index_map(adata, label_col: str):
    """Return ``(ordered_labels, {label: int_index})`` for *label_col*.

    The canonical ordering is ``np.unique`` over the column values.
    """
    ordered_labels = np.unique(adata.obs[label_col])
    return ordered_labels, {label: i for i, label in enumerate(ordered_labels.tolist())}


def patient_labels(adata, patient_col: str) -> np.ndarray:
    """Return sorted unique patient / sample identifiers."""
    return np.unique(adata.obs[patient_col].to_numpy())


def get_distribution_ids(adata, distribution_col: str) -> np.ndarray:
    """Return sorted unique distribution identifiers from ``adata.obs``."""
    if distribution_col not in adata.obs.columns:
        raise KeyError(f"distribution_col {distribution_col!r} not found in adata.obs.")
    return np.sort(np.unique(adata.obs[distribution_col].to_numpy()))


def get_distribution_index(
    adata,
    *,
    distribution_col: str,
    distribution_ids: np.ndarray,
) -> np.ndarray:
    """Return integer distribution index per cell matching *distribution_ids* order."""
    dist_to_idx = {str(d): i for i, d in enumerate(distribution_ids)}
    distribution_values = adata.obs[distribution_col].to_numpy()
    return np.array([dist_to_idx[str(v)] for v in distribution_values], dtype=int)


def get_patient_labels(adata, *, patient_col: str, label_col: str) -> tuple[list[int], list[str]]:
    """Return ``(encoded_indices, original_strings)`` per patient, in patient order.

    ``encoded_indices`` are integer class indices; ``original_strings`` are the
    raw ``label_col`` values stringified. Warns once per patient with mixed labels.
    """
    _, label_to_idx = label_index_map(adata, label_col)
    indices: list[int] = []
    strings: list[str] = []

    for patient in patient_labels(adata, patient_col):
        patient_adata = adata[adata.obs[patient_col] == patient]
        labels = np.unique(patient_adata.obs[label_col].to_numpy())
        if len(labels) > 1:
            warnings.warn(f"Cells from one sample {patient_col}={patient} contain multiple labels {label_col}={labels}")
        indices.append(label_to_idx[labels[0]])
        strings.append(str(labels[0]))
    return indices, strings


# ---------------------------------------------------------------------------
# Matrix extraction
# ---------------------------------------------------------------------------


def get_obs_matrix(adata, use_rep: str | None, layer: str | None = None):
    """Return the raw (possibly sparse) cell × feature matrix from *adata*.

    Uses ``adata.obsm[use_rep]`` when *use_rep* is provided, ``adata.layers[layer]``
    when *layer* is provided, otherwise ``adata.X``.
    """
    if use_rep is not None:
        return adata.obsm[use_rep]
    if layer is not None:
        return adata.layers[layer]
    return adata.X


def get_cell_feature_matrix(adata, use_rep: str | None, layer: str | None = None) -> np.ndarray:
    """Return the cell × feature matrix from *adata* as a dense ``float64`` array.

    See :func:`get_obs_matrix` for how the matrix is selected.
    Sparse matrices are automatically converted to dense.
    """
    x = get_obs_matrix(adata, use_rep, layer)
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# ggml_dataset payload
# ---------------------------------------------------------------------------


def store_ggml_dataset(
    adata,
    *,
    supports,
    covariances,
    weights,
    distribution_labels,
    identical_supports: bool,
    use_rep: str | None,
    index_mask=None,
    label_type: str = "categorical",
    distribution_labels_str=None,
    layer: str | None = None,
):
    """Write the dataset fields into ``adata.uns["ggml_dataset"]``.

    Stores the per-distribution arrays (supports, weights, labels, ...) that
    define the :class:`~ggml_ot.data.TripletDataset`.

    Parameters
    ----------
    adata
        AnnData object whose ``.uns`` dict will be updated.
    supports
        Per-distribution support points (centroids or empirical cells).
    covariances
        Per-distribution covariance matrices, or ``None``.
    weights
        Per-distribution mixture weights, or ``None`` (uniform).
    distribution_labels
        Per-distribution label, as integer class indices.
    identical_supports
        Whether all distributions share the same support set.
    use_rep
        The ``obsm`` key used, or ``None`` when raw ``.X`` (or a layer) was used.
    index_mask
        Boolean mask over ``adata.obs`` indicating which cells were
        retained.  Defaults to ``[True] * adata.n_obs``.
    label_type
        Supervision mode for *distribution_labels* (default
        ``"categorical"``).
    distribution_labels_str
        Original per-distribution labels as strings, aligned to
        *distribution_labels*. ``None`` lets the dataset stringify
        *distribution_labels* as a fallback.
    layer
        The ``layers`` key used, or ``None`` when ``.X`` (or ``use_rep``) was used.

    Returns
    -------
    adata
        The same object, mutated in place for convenience.
    """
    if index_mask is None:
        index_mask = [True] * adata.n_obs

    adata.uns["ggml_dataset"] = {
        "supports": supports,
        "covariances": covariances,
        "weights": weights,
        "distribution_labels": distribution_labels,
        "distribution_labels_str": distribution_labels_str,
        "index_mask": index_mask,
        "identical_supports": identical_supports,
        "use_rep": use_rep,
        "layer": layer,
        "label_type": label_type,
    }
    return adata


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------


def extract_centroids(adata, group_by: str, use_rep: str | None, layer: str | None = None) -> np.ndarray:
    """Compute per-group centroids (mean) from *adata*.

    Parameters
    ----------
    adata
        AnnData object.
    group_by
        Column in ``adata.obs`` whose unique values define the groups.
    use_rep
        If provided, use ``adata.obsm[use_rep]``; otherwise ``adata.X``.
    layer
        If provided, use ``adata.layers[layer]`` instead of ``adata.X``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_groups, n_features)`` with one centroid per group.
    """
    x = get_obs_matrix(adata, use_rep, layer)
    groups = adata.obs[group_by].to_numpy()
    supports = []

    for cluster in np.unique(groups):
        cluster_cells = x[groups == cluster]
        if sp.issparse(cluster_cells):
            cluster_cells = cluster_cells.toarray()

        supports.append(np.mean(np.asarray(cluster_cells, dtype="f"), axis=0))
    return np.asarray(supports)
