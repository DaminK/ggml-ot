"""Closed-form, cell-type-anchored GMM fitting.

In anchored mode every mixture component *is* a cell type: its mean is fixed to
the per-distribution centroid of that cell type and only the covariance and
mixing weight are estimated. Because the component assignment is given by the
label (hard responsibilities), there is no EM and no k-selection — each
``(distribution, cell type)`` Gaussian is estimated in a single closed-form
pass.

The covariance mode is one of:

- ``"full"``  — label-conditional sample covariance per component.
- ``"diag"``  — per-feature variance per component.
- ``"adaptive"`` — full covariance where the component has enough cells to
  estimate a full-rank matrix (``n_k >= n_features + 1``), diagonal otherwise.
  Diagonal fallbacks are materialized as full ``(D, D)`` matrices so the
  persisted model, K_max padding, and downstream OT see a homogeneous
  ``"full"`` representation.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from ggml_ot._utils._covariance import apply_singularity_handling
from ggml_ot._utils._weights import normalize_weight_rows, normalize_weight_vector

AnchoredCovarianceMode = Literal["full", "diag", "adaptive"]


def storage_covariance_type(covariance_mode: AnchoredCovarianceMode) -> Literal["full", "diag"]:
    """Materialized covariance layout for an anchored covariance mode.

    ``"adaptive"`` and ``"full"`` both materialize full ``(D, D)`` matrices, so
    they are stored as ``"full"``; ``"diag"`` keeps diagonal-vector storage.
    """
    if covariance_mode not in ("full", "diag", "adaptive"):
        if covariance_mode == "lowrank":
            raise NotImplementedError(
                "Anchored GMM fitting does not support covariance_type='lowrank'. "
                "A future per-component factor-analysis solve would be required."
            )
        raise ValueError(
            f"Unsupported anchored covariance mode: {covariance_mode!r}. Expected 'full', 'diag', or 'adaptive'."
        )
    return "diag" if covariance_mode == "diag" else "full"


def _component_full_covariance(cells: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Biased (MLE) sample covariance about a fixed mean ``mu``."""
    diff = cells - mu
    return (diff.T @ diff) / cells.shape[0]


def _component_diag_variance(cells: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Biased (MLE) per-feature variance about a fixed mean ``mu``."""
    diff = cells - mu
    return np.mean(diff * diff, axis=0)


def fit_anchored_sample_specific(
    *,
    x: np.ndarray,
    distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
    component_codes: np.ndarray,
    category_names: list[str],
    covariance_mode: AnchoredCovarianceMode,
    eps: float,
    singularity_handling: Literal["guarded", "robust", "strict"],
    verbose: bool = False,
) -> dict[str, Any]:
    """Fit a cell-type-anchored GMM per distribution in closed form.

    Parameters
    ----------
    x
        Cell feature matrix with shape ``(n_cells, n_features)``.
    distribution_index
        Integer distribution (patient) index per cell, shape ``(n_cells,)``.
    distribution_ids
        Distribution identifiers aligned to ``distribution_index`` codes.
    component_codes
        Integer cell-type code per cell, shape ``(n_cells,)``. Cells with a
        negative code (unassigned / NaN label) are excluded from estimation and
        receive hard component ``-1`` with zero responsibility.
    category_names
        Mapping from non-negative code to cell-type name (``category_names[c]``).
    covariance_mode
        ``"full"``, ``"diag"``, or ``"adaptive"`` (see module docstring).
    eps
        Numerical floor for covariances (eigenvalue floor for full matrices,
        minimum variance for diagonal).
    singularity_handling
        Passed to :func:`apply_singularity_handling` for full covariances.
    verbose
        Emit a per-fit summary line.

    Returns
    -------
    dict
        Canonical anchored fit outputs:
        ``n_components`` (K_max), ``distribution_n_components`` ``(D,)``,
        ``model_payload`` (``mu``/``var``/``pi`` stacked over distributions),
        ``distribution_weights`` ``(D, K_max)``, ``hard_components``
        ``(n_cells,)``, ``responsibilities`` ``(n_cells, K_max)``,
        ``component_cell_types`` (ragged per-distribution name lists),
        ``storage_covariance_type``, ``adaptive_diag_fallback_count``, and
        ``degenerate_full_count``.
    """
    x = np.asarray(x, dtype=np.float64)
    distribution_index = np.asarray(distribution_index, dtype=int)
    component_codes = np.asarray(component_codes, dtype=int)
    if x.ndim != 2:
        raise ValueError(f"x must have shape (n_cells, n_features). Got {x.shape}.")
    if component_codes.shape[0] != x.shape[0]:
        raise ValueError("component_codes length must match the number of rows in x.")

    n_cells, n_features = int(x.shape[0]), int(x.shape[1])
    n_distributions = int(len(distribution_ids))
    store_type = storage_covariance_type(covariance_mode)
    full_rank_min_cells = n_features + 1  # n_k >= D + 1 needed for a full-rank covariance

    per_dist_mu: list[np.ndarray] = []
    per_dist_var: list[np.ndarray] = []
    per_dist_pi: list[np.ndarray] = []
    per_dist_names: list[list[str]] = []
    per_dist_k: list[int] = []
    per_dist_cell_slots: list[tuple[np.ndarray, np.ndarray]] = []  # (global_idx, slot)

    adaptive_diag_fallback_count = 0
    degenerate_full_count = 0
    n_unassigned = int(np.count_nonzero(component_codes < 0))

    for d_idx in range(n_distributions):
        dist_mask = distribution_index == d_idx
        global_idx = np.nonzero(dist_mask)[0]
        codes_d = component_codes[global_idx]
        x_d = x[global_idx]

        valid = codes_d >= 0
        present_codes = np.unique(codes_d[valid])
        if present_codes.shape[0] == 0:
            raise ValueError(
                f"Distribution {distribution_ids[d_idx]!r} has no cells with an assigned cell type; "
                "cannot anchor any component."
            )
        n_valid = int(np.count_nonzero(valid))

        mus: list[np.ndarray] = []
        cov_mats: list[np.ndarray] = []  # full (D, D) per component when store_type == "full"
        diag_vars: list[np.ndarray] = []  # (D,) per component when store_type == "diag"
        pis: list[float] = []
        names: list[str] = []
        code_to_slot: dict[int, int] = {}

        for slot, code in enumerate(present_codes):
            code = int(code)
            comp_mask = codes_d == code
            n_k = int(np.count_nonzero(comp_mask))
            cells = x_d[comp_mask]
            mu_k = cells.mean(axis=0)
            mus.append(mu_k)
            pis.append(n_k / n_valid)
            names.append(category_names[code])
            code_to_slot[code] = slot

            if store_type == "diag":
                diag_vars.append(_component_diag_variance(cells, mu_k))
                continue

            use_full = covariance_mode == "full" or (covariance_mode == "adaptive" and n_k >= full_rank_min_cells)
            if use_full:
                cov_mats.append(_component_full_covariance(cells, mu_k))
                if n_k < full_rank_min_cells:
                    degenerate_full_count += 1
            else:
                # adaptive diagonal fallback, materialized as a full matrix
                cov_mats.append(np.diag(_component_diag_variance(cells, mu_k)))
                adaptive_diag_fallback_count += 1

        k_d = len(mus)
        mu_d = np.stack(mus, axis=0)  # (k_d, D)
        if store_type == "diag":
            var_d = np.clip(np.stack(diag_vars, axis=0), a_min=float(eps), a_max=None)  # (k_d, D)
        else:
            var_d = apply_singularity_handling(
                np.stack(cov_mats, axis=0),  # (k_d, D, D)
                covariance_type="full",
                eps=float(eps),
                singularity_handling=singularity_handling,
                n_components=k_d,
            )
        pi_d = normalize_weight_vector(np.asarray(pis, dtype=np.float64))

        per_dist_mu.append(mu_d)
        per_dist_var.append(var_d)
        per_dist_pi.append(pi_d)
        per_dist_names.append(names)
        per_dist_k.append(k_d)

        slot_per_cell = np.full(global_idx.shape[0], -1, dtype=int)
        for local_pos, code in enumerate(codes_d):
            if code >= 0:
                slot_per_cell[local_pos] = code_to_slot[int(code)]
        per_dist_cell_slots.append((global_idx, slot_per_cell))

    k_max = int(max(per_dist_k))

    mu = np.zeros((n_distributions, k_max, n_features), dtype=np.float64)
    if store_type == "diag":
        var = np.zeros((n_distributions, k_max, n_features), dtype=np.float64)
    else:
        var = np.zeros((n_distributions, k_max, n_features, n_features), dtype=np.float64)
    pi = np.zeros((n_distributions, k_max), dtype=np.float64)
    distribution_weights = np.zeros((n_distributions, k_max), dtype=np.float64)
    component_cell_types: list[list[str]] = []

    responsibilities = np.zeros((n_cells, k_max), dtype=np.float64)
    hard_components = np.full(n_cells, -1, dtype=int)

    for d_idx in range(n_distributions):
        k_d = per_dist_k[d_idx]
        mu[d_idx, :k_d] = per_dist_mu[d_idx]
        var[d_idx, :k_d] = per_dist_var[d_idx]
        pi[d_idx, :k_d] = per_dist_pi[d_idx]
        distribution_weights[d_idx, :k_d] = per_dist_pi[d_idx]
        component_cell_types.append(per_dist_names[d_idx] + [""] * (k_max - k_d))

        global_idx, slot_per_cell = per_dist_cell_slots[d_idx]
        assigned = slot_per_cell >= 0
        hard_components[global_idx[assigned]] = slot_per_cell[assigned]
        responsibilities[global_idx[assigned], slot_per_cell[assigned]] = 1.0

    if covariance_mode == "full" and degenerate_full_count > 0:
        warnings.warn(
            f"Anchored full-covariance fit: {degenerate_full_count} component(s) had fewer than "
            f"{full_rank_min_cells} cells (n_features + 1) and are rank-deficient; covariances were "
            "eigenvalue-floored but remain degenerate. Consider covariance_type='adaptive' or 'diag'.",
            UserWarning,
            stacklevel=2,
        )
    if n_unassigned > 0:
        warnings.warn(
            f"Anchored fit: {n_unassigned} cell(s) had an unassigned/NaN cell type and were excluded "
            "from component estimation (hard component -1, zero responsibility).",
            UserWarning,
            stacklevel=2,
        )

    if verbose:
        print(
            f"[fit_gmm] anchored sample-specific: distributions={n_distributions}, "
            f"K_max={k_max}, covariance_mode={covariance_mode}, "
            f"adaptive_diag_fallbacks={adaptive_diag_fallback_count}, "
            f"degenerate_full={degenerate_full_count}, unassigned_cells={n_unassigned}"
        )

    return {
        "n_components": k_max,
        "distribution_n_components": np.asarray(per_dist_k, dtype=int),
        "model_payload": {
            "mu": mu,
            "var": var,
            "pi": normalize_weight_rows(pi),
        },
        "distribution_weights": normalize_weight_rows(distribution_weights),
        "hard_components": hard_components,
        "responsibilities": responsibilities,
        "component_cell_types": component_cell_types,
        "storage_covariance_type": store_type,
        "adaptive_diag_fallback_count": int(adaptive_diag_fallback_count),
        "degenerate_full_count": int(degenerate_full_count),
        "n_unassigned_cells": int(n_unassigned),
    }


__all__ = ["fit_anchored_sample_specific", "storage_covariance_type"]
