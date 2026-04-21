"""GMM component summary and gene-score utilities."""

from __future__ import annotations
from typing import Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData


from ._grouping import (
    BarycenterWeighting,
    GroupingMethod,
    GroupRepresentative,
    _resolve_adata,
    _resolve_grouping_for_consumer,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def resolve_gmm_key(adata: AnnData, gmm_key: str | None, use_rep: str | None) -> str:
    """Resolve a GMM key, falling back to the default naming convention."""
    available_keys = [k for k in adata.uns if k.startswith("gmm_")]
    if gmm_key is not None:
        key = gmm_key
    else:
        rep_label = "X" if use_rep is None else use_rep
        key = f"gmm_{rep_label}"
        if key not in adata.uns and len(available_keys) == 1:
            key = str(available_keys[0])
    if key not in adata.uns:
        raise KeyError(
            f"GMM key {key!r} not found in adata.uns. "
            f"Available keys: {available_keys}. "
            "Fit a GMM first with dataset.fit_gmm()."
        )
    return key


def require_global_or_grouped(
    adata: AnnData,
    gmm_key: str,
    grouping_method: str | None,
) -> None:
    """Raise if the GMM is sample-specific and no grouping is requested."""
    sharing = adata.uns[gmm_key].get("component_sharing", "sample_specific")
    if sharing == "sample_specific" and grouping_method is None:
        raise ValueError(
            f"GMM {gmm_key!r} uses sample-specific components. "
            "Cross-patient component methods require either:\n"
            "  - a GMM fitted with component_sharing='global', or\n"
            "  - an explicit grouping_method='mean' or 'bures-wasserstein' argument.\n"
            "Example: dataset.summarize_gmm_components(..., grouping_method='mean')"
        )


# ---------------------------------------------------------------------------
# Component summaries
# ---------------------------------------------------------------------------


def summarize_gmm_components(
    data,
    gmm_key: str | None = None,
    *,
    groupby: str,
    weighting: Literal["responsibility", "hard"] = "responsibility",
    normalize: Literal["component", "cell_type", "none"] = "component",
    grouping_method: GroupingMethod | None = None,
    n_groups: int | None = None,
    group_representative: GroupRepresentative = "mean",
    barycenter_weighting: BarycenterWeighting = "component_weights",
    grouping_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Summarize GMM components by cell-type composition and patient weights."""
    adata, resolved_gmm_key = _resolve_adata_and_gmm_key(data, gmm_key)
    require_global_or_grouped(adata, resolved_gmm_key, grouping_method)

    gmm_cfg = adata.uns[resolved_gmm_key]
    sharing = gmm_cfg.get("component_sharing", "sample_specific")
    if sharing == "sample_specific" and grouping_method is not None:
        return _summarize_grouped(
            adata,
            resolved_gmm_key,
            groupby=groupby,
            weighting=weighting,
            normalize=normalize,
            grouping_method=grouping_method,
            n_groups=n_groups,
            group_representative=group_representative,
            barycenter_weighting=barycenter_weighting,
            grouping_key=grouping_key,
        )

    n_components = int(gmm_cfg["n_components"])
    celltype_table = _celltype_composition(adata, resolved_gmm_key, n_components, groupby, weighting, normalize)
    patient_weights = _patient_weight_table(adata, resolved_gmm_key)
    purity = _component_purity(celltype_table)
    return {
        "celltype_table": celltype_table,
        "patient_weights": patient_weights,
        "purity": purity,
    }


def component_gene_scores(
    data,
    gmm_key: str | None = None,
    *,
    contrast: str = "latent_mean_shift",
    reference: str = "rest",
    gene_symbols: str | None = None,
    grouping_method: GroupingMethod | None = None,
    n_groups: int | None = None,
    group_representative: GroupRepresentative = "mean",
    barycenter_weighting: BarycenterWeighting = "component_weights",
    grouping_key: str | None = None,
) -> pd.DataFrame:
    """Compute per-component gene scores via latent-space contrast."""
    adata, resolved_gmm_key = _resolve_adata_and_gmm_key(data, gmm_key)
    require_global_or_grouped(adata, resolved_gmm_key, grouping_method)

    if contrast != "latent_mean_shift":
        raise ValueError(f"Unknown contrast method: {contrast!r}. Use 'latent_mean_shift'.")

    gmm_cfg = adata.uns[resolved_gmm_key]
    sharing = gmm_cfg.get("component_sharing", "sample_specific")
    if sharing == "sample_specific" and grouping_method is not None:
        return _grouped_gene_scores(
            adata,
            resolved_gmm_key,
            reference=reference,
            gene_symbols=gene_symbols,
            grouping_method=grouping_method,
            n_groups=n_groups,
            group_representative=group_representative,
            barycenter_weighting=barycenter_weighting,
            grouping_key=grouping_key,
        )

    mu = _global_component_means(np.asarray(gmm_cfg["model"]["mu"], dtype=np.float64))
    weights = np.asarray(gmm_cfg["distribution_weights"], dtype=np.float64)
    avg_weights = weights.mean(axis=0)
    projection = _get_gene_projection_matrix(
        adata,
        rep_dim=mu.shape[1],
        use_rep=gmm_cfg.get("use_rep"),
    )
    gene_names = _resolve_gene_names(adata, gene_symbols)

    rows: list[dict[str, Any]] = []
    for component_idx in range(mu.shape[0]):
        delta = _compute_contrast(mu, avg_weights, component_idx, reference)
        gene_scores = delta if projection is None else projection @ delta
        order = np.argsort(-np.abs(gene_scores))
        for rank_idx, gene_idx in enumerate(order):
            score = float(gene_scores[gene_idx])
            rows.append(
                {
                    "component": component_idx,
                    "gene": gene_names[gene_idx],
                    "score": score,
                    "abs_score": abs(score),
                    "rank": rank_idx + 1,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_adata_and_gmm_key(data, gmm_key: str | None) -> tuple[AnnData, str]:
    adata = _resolve_adata(data)
    if isinstance(data, AnnData):
        use_rep = adata.uns.get("ggml_params", {}).get("use_rep")
    else:
        use_rep = getattr(data, "use_rep", None)
    resolved_key = resolve_gmm_key(adata, gmm_key, use_rep)
    return adata, resolved_key


def _get_gene_projection_matrix(
    adata: AnnData,
    *,
    rep_dim: int,
    use_rep: str | None,
) -> np.ndarray | None:
    """Return a gene-space projection for vectors in the GMM representation space."""
    if use_rep is None:
        if rep_dim != adata.n_vars:
            raise ValueError(f"GMM representation dimension {rep_dim} does not match adata.n_vars={adata.n_vars}.")
        return None

    if "W_ggml" in adata.varm and "W_ggml" in adata.uns:
        ggml_loadings = np.asarray(adata.varm["W_ggml"])
        map_A = np.asarray(adata.uns["W_ggml"])
        if (
            ggml_loadings.ndim == 2
            and map_A.ndim == 2
            and map_A.shape[1] == rep_dim
            and ggml_loadings.shape[1] == map_A.shape[0]
        ):
            return ggml_loadings @ map_A

    candidate_keys = []
    if use_rep in adata.varm:
        candidate_keys.append(use_rep)
    if "pca" in use_rep.lower() and "PCs" in adata.varm:
        candidate_keys.append("PCs")
    if "W_ggml" in adata.varm:
        candidate_keys.append("W_ggml")

    seen: set[str] = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        loadings = np.asarray(adata.varm[key])
        if loadings.ndim == 2 and loadings.shape[1] == rep_dim:
            return loadings

    available = {key: tuple(np.asarray(adata.varm[key]).shape) for key in adata.varm.keys()}
    raise ValueError(
        "Could not project GMM components back to gene space. "
        f"GMM use_rep={use_rep!r} has dimension {rep_dim}, "
        f"but no compatible loadings were found in adata.varm. Available shapes: {available}"
    )


def _resolve_gene_names(adata: AnnData, gene_symbols: str | None) -> np.ndarray:
    if gene_symbols is not None:
        return np.asarray(adata.var[gene_symbols])
    return np.asarray(adata.var_names)


def _compute_contrast(
    mu: np.ndarray,
    avg_weights: np.ndarray,
    component_idx: int,
    reference: str,
) -> np.ndarray:
    """Compute ``mu_component - mu_reference`` under the requested baseline."""
    if reference == "rest":
        mask = np.ones(len(avg_weights), dtype=bool)
        mask[component_idx] = False
        w_rest = avg_weights[mask]
        if float(w_rest.sum()) < 1e-12:
            mu_ref = np.zeros_like(mu[component_idx])
        else:
            mu_ref = (w_rest[:, None] * mu[mask]).sum(axis=0) / float(w_rest.sum())
    elif reference == "global_mean":
        mu_ref = (avg_weights[:, None] * mu).sum(axis=0) / float(avg_weights.sum())
    else:
        raise ValueError(f"Unknown reference: {reference!r}. Use 'rest' or 'global_mean'.")
    return mu[component_idx] - mu_ref


def _celltype_composition(
    adata: AnnData,
    gmm_key: str,
    n_components: int,
    groupby: str,
    weighting: str,
    normalize: str,
) -> pd.DataFrame:
    """Build a component-by-cell-type composition table."""
    if groupby not in adata.obs.columns:
        raise KeyError(f"Column {groupby!r} not found in adata.obs.")

    cell_types = adata.obs[groupby].values
    unique_types = np.unique(cell_types)
    table = np.zeros((n_components, len(unique_types)), dtype=np.float64)

    if weighting == "responsibility":
        resp_key = f"{gmm_key}_resp"
        if resp_key not in adata.obsm:
            raise KeyError(
                f"Responsibilities not found at adata.obsm[{resp_key!r}]. Refit the GMM with responsibilities stored."
            )
        responsibilities = np.nan_to_num(
            np.asarray(adata.obsm[resp_key]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for type_idx, cell_type in enumerate(unique_types):
            mask = cell_types == cell_type
            table[:, type_idx] = responsibilities[mask].sum(axis=0)
    elif weighting == "hard":
        comp_key = f"{gmm_key}_comp"
        if comp_key not in adata.obs.columns:
            raise KeyError(f"Hard assignments not found at adata.obs[{comp_key!r}].")
        hard = np.asarray(adata.obs[comp_key])
        for type_idx, cell_type in enumerate(unique_types):
            mask = cell_types == cell_type
            for component_idx in range(n_components):
                table[component_idx, type_idx] = np.sum(mask & (hard == component_idx))
    else:
        raise ValueError(f"Unknown weighting: {weighting!r}. Use 'responsibility' or 'hard'.")

    return _to_component_table(
        table,
        index=[f"comp_{component_idx}" for component_idx in range(n_components)],
        columns=unique_types,
        normalize=normalize,
    )


def _to_component_table(
    values: np.ndarray,
    *,
    index: list[str],
    columns: np.ndarray,
    normalize: str,
) -> pd.DataFrame:
    table = np.asarray(values, dtype=np.float64)
    if normalize == "component":
        row_sums = table.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        table = table / row_sums
    elif normalize == "cell_type":
        col_sums = table.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        table = table / col_sums
    elif normalize != "none":
        raise ValueError(f"Unknown normalize: {normalize!r}.")
    return pd.DataFrame(table, index=index, columns=columns)


def _patient_weight_table(adata: AnnData, gmm_key: str) -> pd.DataFrame:
    """Build a patient-by-component weight table from stored distribution weights."""
    gmm_cfg = adata.uns[gmm_key]
    distribution_weights = np.asarray(gmm_cfg["distribution_weights"], dtype=np.float64)
    distribution_ids = gmm_cfg.get("weight_inference", {}).get("distribution_ids")
    if distribution_ids is None:
        distribution_ids = [str(i) for i in range(distribution_weights.shape[0])]
    return pd.DataFrame(
        distribution_weights,
        index=[str(x) for x in distribution_ids],
        columns=[f"comp_{component_idx}" for component_idx in range(distribution_weights.shape[1])],
    )


def _component_purity(celltype_table: pd.DataFrame) -> pd.DataFrame:
    """Compute per-component purity (max proportion) and entropy."""
    values = celltype_table.values
    max_prop = values.max(axis=1)
    dominant = celltype_table.columns[values.argmax(axis=1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vals = np.where(values > 0, np.log2(values), 0.0)
    entropy = -(values * log_vals).sum(axis=1)
    return pd.DataFrame(
        {
            "dominant_type": dominant,
            "purity": max_prop,
            "entropy": entropy,
        },
        index=celltype_table.index,
    )


def _summarize_grouped(
    adata: AnnData,
    gmm_key: str,
    *,
    groupby: str,
    weighting: str,
    normalize: str,
    grouping_method: GroupingMethod,
    n_groups: int | None,
    group_representative: GroupRepresentative,
    barycenter_weighting: BarycenterWeighting,
    grouping_key: str | None,
) -> dict[str, Any]:
    grouping = _resolve_grouping_for_consumer(
        adata,
        gmm_key,
        grouping_method=grouping_method,
        n_groups=n_groups,
        group_representative=group_representative,
        barycenter_weighting=barycenter_weighting,
        grouping_key=grouping_key,
    )
    n_groups_resolved = int(grouping["n_groups"])
    label_matrix = np.asarray(grouping["label_matrix"], dtype=int)
    distribution_ids = [str(x) for x in grouping["distribution_ids"]]
    celltype_table = _grouped_celltype_composition(
        adata,
        gmm_key,
        label_matrix=label_matrix,
        distribution_ids=distribution_ids,
        n_groups=n_groups_resolved,
        groupby=groupby,
        weighting=weighting,
        normalize=normalize,
    )
    patient_weights = pd.DataFrame(
        np.asarray(grouping["grouped_weights"], dtype=np.float64),
        index=distribution_ids,
        columns=[f"group_{group_idx}" for group_idx in range(n_groups_resolved)],
    )
    purity = _component_purity(celltype_table)
    return {
        "celltype_table": celltype_table,
        "patient_weights": patient_weights,
        "purity": purity,
        "grouping": grouping,
    }


def _grouped_celltype_composition(
    adata: AnnData,
    gmm_key: str,
    *,
    label_matrix: np.ndarray,
    distribution_ids: list[str],
    n_groups: int,
    groupby: str,
    weighting: str,
    normalize: str,
) -> pd.DataFrame:
    if groupby not in adata.obs.columns:
        raise KeyError(f"Column {groupby!r} not found in adata.obs.")

    patient_col = _resolve_patient_col(adata)
    cell_types = adata.obs[groupby].values
    unique_types = np.unique(cell_types)
    table = np.zeros((n_groups, len(unique_types)), dtype=np.float64)
    patients = adata.obs[patient_col].astype(str).values

    if weighting == "hard":
        comp_key = f"{gmm_key}_comp"
        if comp_key not in adata.obs.columns:
            raise KeyError(f"Hard assignments not found at adata.obs[{comp_key!r}].")
        hard = np.asarray(adata.obs[comp_key], dtype=int)
        for dist_idx, dist_id in enumerate(distribution_ids):
            patient_mask = patients == str(dist_id)
            for type_idx, cell_type in enumerate(unique_types):
                combined = patient_mask & (cell_types == cell_type)
                if not np.any(combined):
                    continue
                for comp_idx in range(label_matrix.shape[1]):
                    group_idx = int(label_matrix[dist_idx, comp_idx])
                    if group_idx < 0:
                        continue
                    table[group_idx, type_idx] += np.sum(combined & (hard == comp_idx))
    elif weighting == "responsibility":
        resp_key = f"{gmm_key}_resp"
        if resp_key not in adata.obsm:
            raise KeyError(f"Responsibilities not found at adata.obsm[{resp_key!r}].")
        responsibilities = np.nan_to_num(
            np.asarray(adata.obsm[resp_key]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for dist_idx, dist_id in enumerate(distribution_ids):
            patient_mask = patients == str(dist_id)
            for type_idx, cell_type in enumerate(unique_types):
                combined = patient_mask & (cell_types == cell_type)
                if not np.any(combined):
                    continue
                resp_subset = responsibilities[combined]
                for comp_idx in range(label_matrix.shape[1]):
                    group_idx = int(label_matrix[dist_idx, comp_idx])
                    if group_idx < 0:
                        continue
                    table[group_idx, type_idx] += resp_subset[:, comp_idx].sum()
    else:
        raise ValueError(f"Unknown weighting: {weighting!r}.")

    return _to_component_table(
        table,
        index=[f"group_{group_idx}" for group_idx in range(n_groups)],
        columns=unique_types,
        normalize=normalize,
    )


def _resolve_patient_col(adata: AnnData) -> str:
    for candidate in ("sample", "patient"):
        if candidate in adata.obs.columns:
            return candidate
    ggml_params = dict(adata.uns.get("ggml_params", {}))
    patient_col = ggml_params.get("patient_col")
    if patient_col is not None:
        return str(patient_col)
    raise KeyError(
        "Cannot determine patient column for grouped component summaries. "
        "Ensure adata.obs contains 'sample' or 'patient', or store ggml_params['patient_col']."
    )


def _grouped_gene_scores(
    adata: AnnData,
    gmm_key: str,
    *,
    reference: str,
    gene_symbols: str | None,
    grouping_method: GroupingMethod,
    n_groups: int | None,
    group_representative: GroupRepresentative,
    barycenter_weighting: BarycenterWeighting,
    grouping_key: str | None,
) -> pd.DataFrame:
    grouping = _resolve_grouping_for_consumer(
        adata,
        gmm_key,
        grouping_method=grouping_method,
        n_groups=n_groups,
        group_representative=group_representative,
        barycenter_weighting=barycenter_weighting,
        grouping_key=grouping_key,
    )
    mu = np.asarray(grouping["grouped_mu"], dtype=np.float64)
    grouped_weights = np.asarray(grouping["grouped_weights"], dtype=np.float64)
    avg_weights = grouped_weights.mean(axis=0)

    gmm_cfg = adata.uns[gmm_key]
    projection = _get_gene_projection_matrix(
        adata,
        rep_dim=mu.shape[1],
        use_rep=gmm_cfg.get("use_rep"),
    )
    gene_names = _resolve_gene_names(adata, gene_symbols)

    rows: list[dict[str, Any]] = []
    for component_idx in range(mu.shape[0]):
        delta = _compute_contrast(mu, avg_weights, component_idx, reference)
        gene_scores = delta if projection is None else projection @ delta
        order = np.argsort(-np.abs(gene_scores))
        for rank_idx, gene_idx in enumerate(order):
            score = float(gene_scores[gene_idx])
            rows.append(
                {
                    "component": component_idx,
                    "gene": gene_names[gene_idx],
                    "score": score,
                    "abs_score": abs(score),
                    "rank": rank_idx + 1,
                }
            )
    return pd.DataFrame(rows)


def _global_component_means(mu: np.ndarray) -> np.ndarray:
    """Return globally shared component means in canonical shape ``(K, d)``."""
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim == 3 and mu.shape[0] == 1:
        return mu[0]
    if mu.ndim != 2:
        raise ValueError(f"Global/shared GMM component means must have shape (K, d) or (1, K, d). Got {mu.shape}.")
    return mu


__all__ = [
    "component_gene_scores",
    "resolve_gmm_key",
    "summarize_gmm_components",
]
