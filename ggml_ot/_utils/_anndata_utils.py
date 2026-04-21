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


def validate_anndata_inputs(adata, patient_col: str, label_col: str, use_rep: str | None) -> None:
    """Assert that required columns and representations exist in *adata*."""
    if patient_col not in adata.obs:
        raise ValueError(f"patient_col {patient_col} not in adata.obs columns {adata.obs.columns}")
    if label_col not in adata.obs:
        raise ValueError(f"label_col {label_col} not in adata.obs columns {adata.obs.columns}")
    if use_rep is not None and use_rep not in adata.obsm:
        raise ValueError(f"use_rep {use_rep} not in adata.obsm keys {adata.obsm.keys()}")


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


def get_patient_label_indices(adata, *, patient_col: str, label_col: str) -> list[int]:
    """Return integer labels in patient order, warning on mixed labels per patient."""
    _, label_to_idx = label_index_map(adata, label_col)
    indices: list[int] = []

    for patient in patient_labels(adata, patient_col):
        patient_adata = adata[adata.obs[patient_col] == patient]
        labels = np.unique(patient_adata.obs[label_col].to_numpy())
        if len(labels) == 0:
            raise ValueError(f"No labels found for {patient_col}={patient!r}.")
        if len(labels) > 1:
            warnings.warn(f"Cells from one sample {patient_col}={patient} contain multiple labels {label_col}={labels}")
        indices.append(label_to_idx[labels[0]])
    return indices


# ---------------------------------------------------------------------------
# Matrix extraction
# ---------------------------------------------------------------------------


def get_cell_feature_matrix(adata, use_rep: str | None) -> np.ndarray:
    """Return the cell × feature matrix from *adata* as a dense ``float64`` array.

    Uses ``adata.obsm[use_rep]`` when *use_rep* is provided, otherwise ``adata.X``.
    Sparse matrices are automatically converted to dense.
    """
    x = adata.obsm[use_rep] if use_rep is not None else adata.X
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------


def extract_centroids(adata, group_by: str, use_rep: str | None) -> np.ndarray:
    """Compute per-group centroids (mean) from *adata*.

    Parameters
    ----------
    adata
        AnnData object.
    group_by
        Column in ``adata.obs`` whose unique values define the groups.
    use_rep
        If provided, use ``adata.obsm[use_rep]``; otherwise ``adata.X``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_groups, n_features)`` with one centroid per group.
    """
    clusters = np.unique(adata.obs[group_by])
    supports = []

    for cluster in clusters:
        cluster_cells = (
            adata[adata.obs[group_by] == cluster].X
            if use_rep is None
            else adata[adata.obs[group_by] == cluster].obsm[use_rep]
        )
        if sp.issparse(cluster_cells):
            cluster_cells = cluster_cells.toarray()

        supports.append(np.mean(np.asarray(cluster_cells, dtype="f"), axis=0))
    return np.asarray(supports)
