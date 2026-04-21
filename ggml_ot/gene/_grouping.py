"""Component grouping / alignment for sample-specific GMMs.

Groups patient-local components into cross-patient "families"
by clustering their stored GMM representations in the original
representation space used for GMM fitting.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any, Literal

import numpy as np
from anndata import AnnData

from ggml_ot._utils._covariance import canonicalize_covariances
from ggml_ot.settings import settings


GroupingMethod = Literal["mean", "bures-wasserstein"]
GroupRepresentative = Literal["mean", "gaussian"]
BarycenterWeighting = Literal["uniform", "component_weights"]


def group_components(
    data,
    gmm_key: str,
    *,
    grouping_method: GroupingMethod = "mean",
    n_groups: int | None = None,
    group_representative: GroupRepresentative = "mean",
    barycenter_weighting: BarycenterWeighting = "component_weights",
    grouping_key: str | None = None,
) -> dict[str, Any]:
    """Group sample-specific components into cross-patient families.

    Parameters
    ----------
    data
        Trained ``AnnData_TripletDataset`` or raw ``AnnData`` containing a
        sample-specific GMM under ``gmm_key``.
    gmm_key
        Key in ``adata.uns`` for the source GMM schema.
    grouping_method
        Group assignment strategy.
    n_groups
        Number of cross-patient groups. Defaults to
        ``max(distribution_n_components)``.
    group_representative
        Group representative to compute after assignment.
    barycenter_weighting
        Weighting scheme for grouped Gaussian representatives.
    grouping_key
        Explicit key under ``adata.uns`` where the grouped result is stored.
        When ``None``, a derived sibling key is used.

    Returns
    -------
    dict
        Stable grouped representation containing:
        ``label_matrix``, ``distribution_ids``, ``grouped_mu``,
        ``grouped_var``, ``grouped_weights``, grouping metadata,
        source metadata, and the resolved ``grouping_key``.
    """
    adata = _resolve_adata(data)
    _validate_grouping_options(
        grouping_method=grouping_method,
        group_representative=group_representative,
        barycenter_weighting=barycenter_weighting,
    )

    gmm_cfg = adata.uns[gmm_key]
    sharing = str(gmm_cfg.get("component_sharing", "sample_specific"))
    if sharing != "sample_specific":
        raise ValueError(
            "Component grouping is only supported for sample_specific GMMs, "
            f"but gmm_key={gmm_key!r} has component_sharing={sharing!r}."
        )

    mu = np.asarray(gmm_cfg["model"]["mu"], dtype=np.float64)
    if mu.ndim != 3:
        raise ValueError(f"Sample-specific grouping expects model['mu'] with shape (D, K_max, d). Got {mu.shape}.")
    covariances = canonicalize_covariances(
        np.asarray(gmm_cfg["model"]["var"], dtype=np.float64),
        mu,
        covariance_type=str(gmm_cfg.get("covariance_type", "full")),
    )
    distribution_weights = np.asarray(gmm_cfg["distribution_weights"], dtype=np.float64)
    if distribution_weights.ndim != 2 or distribution_weights.shape[:2] != mu.shape[:2]:
        raise ValueError(
            "distribution_weights must have shape (D, K_max) aligned with model['mu']. "
            f"Got weights={distribution_weights.shape}, mu={mu.shape}."
        )

    distribution_ids = _resolve_distribution_ids(gmm_cfg)
    distribution_n_components = _resolve_distribution_n_components(gmm_cfg, k_max=mu.shape[1])
    if len(distribution_ids) != mu.shape[0]:
        raise ValueError(
            "Stored distribution_ids must align with the first axis of model['mu']. "
            f"Got len(distribution_ids)={len(distribution_ids)}, mu.shape[0]={mu.shape[0]}."
        )

    if n_groups is None:
        n_groups = int(np.max(distribution_n_components))
    n_groups = _validate_n_groups(n_groups=n_groups, n_active=int(distribution_n_components.sum()))

    active_dist_idx, active_comp_idx = _active_component_indices(distribution_n_components, k_max=mu.shape[1])
    active_mu = mu[active_dist_idx, active_comp_idx]
    active_covariances = covariances[active_dist_idx, active_comp_idx]
    active_weights = distribution_weights[active_dist_idx, active_comp_idx]

    labels = _cluster_active_components(
        active_mu,
        active_covariances,
        grouping_method=grouping_method,
        n_groups=n_groups,
    )
    label_matrix = np.full((mu.shape[0], mu.shape[1]), -1, dtype=int)
    label_matrix[active_dist_idx, active_comp_idx] = labels

    grouped_mu, grouped_var = _build_group_representatives(
        active_mu=active_mu,
        active_covariances=active_covariances,
        active_weights=active_weights,
        labels=labels,
        n_groups=n_groups,
        group_representative=group_representative,
        barycenter_weighting=barycenter_weighting,
    )
    grouped_weights = _build_grouped_weight_table(
        distribution_weights=distribution_weights,
        label_matrix=label_matrix,
        n_groups=n_groups,
    )
    label_matrix, grouped_mu, grouped_var, grouped_weights = _canonicalize_group_order(
        label_matrix=label_matrix,
        grouped_mu=grouped_mu,
        grouped_var=grouped_var,
        grouped_weights=grouped_weights,
    )

    resolved_grouping_key = grouping_key or _default_grouping_key(
        gmm_key=gmm_key,
        grouping_method=grouping_method,
        group_representative=group_representative,
        barycenter_weighting=barycenter_weighting,
        n_groups=n_groups,
    )
    if resolved_grouping_key == gmm_key:
        raise ValueError(
            "grouping_key must not overwrite the original fitted GMM key. "
            f"Got grouping_key={resolved_grouping_key!r} equal to gmm_key={gmm_key!r}."
        )

    source_checksum = _gmm_source_checksum(
        gmm_cfg=gmm_cfg,
        mu=mu,
        covariances=covariances,
        distribution_weights=distribution_weights,
        distribution_n_components=distribution_n_components,
    )

    grouping = {
        "label_matrix": label_matrix,
        "distribution_ids": list(distribution_ids),
        "grouped_mu": grouped_mu,
        "grouped_var": grouped_var,
        "grouped_weights": grouped_weights,
        "grouping_method": grouping_method,
        "group_representative": group_representative,
        "barycenter_weighting": barycenter_weighting,
        "n_groups": int(n_groups),
        "source_gmm_key": str(gmm_key),
        "source_checksum": source_checksum,
        "grouping_key": resolved_grouping_key,
    }
    adata.uns[resolved_grouping_key] = grouping
    return grouping


def _resolve_grouping_for_consumer(
    data,
    gmm_key: str,
    *,
    grouping_method: GroupingMethod,
    n_groups: int | None,
    group_representative: GroupRepresentative,
    barycenter_weighting: BarycenterWeighting,
    grouping_key: str | None,
) -> dict[str, Any]:
    """Return a stored grouping when valid, otherwise compute and persist it."""
    adata = _resolve_adata(data)
    gmm_cfg = adata.uns[gmm_key]
    distribution_n_components = _resolve_distribution_n_components(
        gmm_cfg,
        k_max=int(np.asarray(gmm_cfg["model"]["mu"]).shape[1]),
    )
    resolved_n_groups = int(np.max(distribution_n_components)) if n_groups is None else int(n_groups)
    resolved_grouping_key = grouping_key or _default_grouping_key(
        gmm_key=gmm_key,
        grouping_method=grouping_method,
        group_representative=group_representative,
        barycenter_weighting=barycenter_weighting,
        n_groups=resolved_n_groups,
    )

    stored = adata.uns.get(resolved_grouping_key)
    if stored is None:
        return group_components(
            adata,
            gmm_key,
            grouping_method=grouping_method,
            n_groups=resolved_n_groups,
            group_representative=group_representative,
            barycenter_weighting=barycenter_weighting,
            grouping_key=resolved_grouping_key,
        )

    _validate_stored_grouping(
        stored_grouping=stored,
        gmm_key=gmm_key,
        grouping_method=grouping_method,
        n_groups=resolved_n_groups,
        group_representative=group_representative,
        barycenter_weighting=barycenter_weighting,
    )
    current_checksum = _gmm_source_checksum(
        gmm_cfg=gmm_cfg,
        mu=np.asarray(gmm_cfg["model"]["mu"], dtype=np.float64),
        covariances=canonicalize_covariances(
            np.asarray(gmm_cfg["model"]["var"], dtype=np.float64),
            np.asarray(gmm_cfg["model"]["mu"], dtype=np.float64),
            covariance_type=str(gmm_cfg.get("covariance_type", "full")),
        ),
        distribution_weights=np.asarray(gmm_cfg["distribution_weights"], dtype=np.float64),
        distribution_n_components=distribution_n_components,
    )
    stored_checksum = str(stored.get("source_checksum", ""))
    if stored_checksum != current_checksum:
        warnings.warn(
            f"Stored grouping {resolved_grouping_key!r} is outdated relative to the current "
            f"GMM {gmm_key!r}. Rerun group_components(...) to refresh it.",
            UserWarning,
            stacklevel=2,
        )
    return stored


def _resolve_adata(data) -> AnnData:
    if isinstance(data, AnnData):
        return data
    adata = getattr(data, "adata", None)
    if isinstance(adata, AnnData):
        return adata
    raise TypeError("Expected an AnnData object or AnnData-backed dataset with an `.adata` attribute.")


def _validate_stored_grouping(
    *,
    stored_grouping: dict[str, Any],
    gmm_key: str,
    grouping_method: str,
    n_groups: int,
    group_representative: str,
    barycenter_weighting: str,
) -> None:
    expected = {
        "source_gmm_key": gmm_key,
        "grouping_method": grouping_method,
        "group_representative": group_representative,
        "barycenter_weighting": barycenter_weighting,
        "n_groups": int(n_groups),
    }
    mismatches = []
    for key, expected_value in expected.items():
        stored_value = stored_grouping.get(key)
        if stored_value != expected_value:
            mismatches.append(f"{key}: stored={stored_value!r}, expected={expected_value!r}")
    if mismatches:
        raise ValueError("Stored grouping does not match the requested grouping parameters. " + "; ".join(mismatches))


def _validate_grouping_options(
    *,
    grouping_method: str,
    group_representative: str,
    barycenter_weighting: str,
) -> None:
    if grouping_method not in {"mean", "bures-wasserstein"}:
        raise ValueError(f"Unknown grouping_method={grouping_method!r}. Use 'mean' or 'bures-wasserstein'.")
    if group_representative not in {"mean", "gaussian"}:
        raise ValueError(f"Unknown group_representative={group_representative!r}. Use 'mean' or 'gaussian'.")
    if barycenter_weighting not in {"uniform", "component_weights"}:
        raise ValueError(
            f"Unknown barycenter_weighting={barycenter_weighting!r}. Use 'uniform' or 'component_weights'."
        )


def _resolve_distribution_ids(gmm_cfg: dict[str, Any]) -> list[str]:
    distribution_ids = gmm_cfg.get("weight_inference", {}).get("distribution_ids")
    if distribution_ids is None:
        raise KeyError(
            "Missing stored distribution ids under `adata.uns[gmm_key]['weight_inference']['distribution_ids']`."
        )
    return [str(x) for x in distribution_ids]


def _resolve_distribution_n_components(gmm_cfg: dict[str, Any], *, k_max: int) -> np.ndarray:
    distribution_n_components = np.asarray(
        gmm_cfg.get("distribution_n_components", np.full(1, k_max)),
        dtype=int,
    )
    if distribution_n_components.ndim != 1:
        raise ValueError(
            f"distribution_n_components must be a 1-D vector. Got shape={distribution_n_components.shape}."
        )
    if np.any(distribution_n_components < 0):
        raise ValueError("distribution_n_components must be non-negative.")
    if np.any(distribution_n_components > int(k_max)):
        raise ValueError(
            "distribution_n_components cannot exceed the stored K_max. "
            f"Got max={int(np.max(distribution_n_components))}, K_max={k_max}."
        )
    return distribution_n_components


def _validate_n_groups(*, n_groups: int, n_active: int) -> int:
    try:
        resolved = int(n_groups)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n_groups must be a positive integer. Got {n_groups!r}.") from exc
    if resolved <= 0:
        raise ValueError(f"n_groups must be a positive integer. Got {resolved}.")
    if n_active <= 0:
        raise ValueError("Cannot group components because there are no active components.")
    if resolved > n_active:
        raise ValueError(f"n_groups={resolved} cannot exceed the number of active components ({n_active}).")
    return resolved


def _active_component_indices(
    distribution_n_components: np.ndarray,
    *,
    k_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    dist_indices: list[int] = []
    comp_indices: list[int] = []
    for dist_idx, n_active in enumerate(distribution_n_components.tolist()):
        for comp_idx in range(int(n_active)):
            dist_indices.append(dist_idx)
            comp_indices.append(comp_idx)
    if len(dist_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.asarray(dist_indices, dtype=int), np.asarray(comp_indices, dtype=int)


def _cluster_active_components(
    active_mu: np.ndarray,
    active_covariances: np.ndarray,
    *,
    grouping_method: GroupingMethod,
    n_groups: int,
) -> np.ndarray:
    n_active = int(active_mu.shape[0])
    if n_active == 0:
        raise ValueError("Cannot group zero active components.")
    if n_groups == 1:
        return np.zeros(n_active, dtype=int)
    if n_groups == n_active:
        return np.arange(n_active, dtype=int)
    if grouping_method == "mean":
        labels, _ = _cluster_by_mean(active_mu, n_groups)
        return labels.astype(int, copy=False)
    labels, _ = _cluster_by_bures(active_mu, active_covariances, n_groups)
    return labels.astype(int, copy=False)


def _cluster_by_mean(flat_mu: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means on component means."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_groups, n_init=10, random_state=settings.random_seed)
    labels = km.fit_predict(flat_mu)
    return labels, np.asarray(km.cluster_centers_, dtype=np.float64)


def _cluster_by_bures(
    flat_mu: np.ndarray,
    flat_covariances: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster Gaussian components using Bures-Wasserstein distance."""
    from sklearn.cluster import SpectralClustering

    n = int(flat_mu.shape[0])
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    is_diag = flat_covariances.ndim == 2
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = _bures_w2_squared(
                flat_mu[i],
                flat_mu[j],
                flat_covariances[i],
                flat_covariances[j],
                is_diag=is_diag,
            )
            dist_matrix[i, j] = dist_ij
            dist_matrix[j, i] = dist_ij

    positive = dist_matrix[dist_matrix > 0]
    if positive.size == 0:
        labels = np.arange(n, dtype=int) % n_groups
    else:
        sigma = float(np.median(positive)) + 1e-10
        affinity = np.exp(-dist_matrix / (2.0 * sigma))
        sc = SpectralClustering(
            n_clusters=n_groups,
            affinity="precomputed",
            random_state=settings.random_seed,
            n_init=10,
        )
        labels = np.asarray(sc.fit_predict(affinity), dtype=int)

    centroids = np.zeros((n_groups, flat_mu.shape[1]), dtype=np.float64)
    for group_idx in range(n_groups):
        mask = labels == group_idx
        if np.any(mask):
            centroids[group_idx] = flat_mu[mask].mean(axis=0)
    return labels, centroids


def _bures_w2_squared(
    mu1: np.ndarray,
    mu2: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    *,
    is_diag: bool,
) -> float:
    """Squared Bures-Wasserstein distance between two Gaussian components."""
    mean_diff_sq = float(np.sum((mu1 - mu2) ** 2))
    if is_diag:
        bures = float(np.sum(cov1) + np.sum(cov2) - 2.0 * np.sum(np.sqrt(cov1 * cov2)))
    else:
        sqrt_cov1 = _matrix_sqrt(cov1)
        inner = sqrt_cov1 @ cov2 @ sqrt_cov1
        bures = float(np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(_matrix_sqrt(inner)))
    return mean_diff_sq + max(bures, 0.0)


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _build_group_representatives(
    *,
    active_mu: np.ndarray,
    active_covariances: np.ndarray,
    active_weights: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
    group_representative: GroupRepresentative,
    barycenter_weighting: BarycenterWeighting,
) -> tuple[np.ndarray, np.ndarray | None]:
    n_dim = int(active_mu.shape[1])
    grouped_mu = np.zeros((n_groups, n_dim), dtype=np.float64)
    grouped_var = None if group_representative == "mean" else np.zeros((n_groups, n_dim, n_dim), dtype=np.float64)

    for group_idx in range(n_groups):
        mask = labels == group_idx
        group_mu = active_mu[mask]
        group_covariances = active_covariances[mask]
        group_component_weights = active_weights[mask]
        if group_mu.shape[0] == 0:
            warnings.warn(
                f"Group {group_idx} has no active members; its representative will be zero-filled.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if group_representative == "gaussian":
            rep_weights = _representative_weights(
                group_component_weights,
                scheme=barycenter_weighting,
            )
            grouped_mu[group_idx] = np.sum(rep_weights[:, None] * group_mu, axis=0)
            grouped_var[group_idx] = _bures_wasserstein_group_covariance(
                group_mu=group_mu,
                group_covariances=group_covariances,
                weights=rep_weights,
            )
        else:
            rep_weights = _representative_weights(
                group_component_weights,
                scheme=barycenter_weighting,
            )
            grouped_mu[group_idx] = np.sum(rep_weights[:, None] * group_mu, axis=0)

    return grouped_mu, grouped_var


def _representative_weights(
    weights: np.ndarray,
    *,
    scheme: BarycenterWeighting,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"Expected 1-D component weights, got shape={weights.shape}.")
    if weights.size == 0:
        raise ValueError("Cannot build representative weights for an empty group.")

    if scheme == "uniform":
        return np.full(weights.shape, 1.0 / float(weights.size), dtype=np.float64)

    total = float(weights.sum())
    if total <= 0:
        return np.full(weights.shape, 1.0 / float(weights.size), dtype=np.float64)
    return weights / total


def _bures_wasserstein_group_covariance(
    *,
    group_mu: np.ndarray,
    group_covariances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the Bures-Wasserstein barycenter covariance for one group."""
    if group_covariances.shape[0] == 1:
        return np.asarray(group_covariances[0], dtype=np.float64)
    import ot

    _, covariance = ot.gaussian.bures_wasserstein_barycenter(
        np.asarray(group_mu, dtype=np.float64),
        np.asarray(group_covariances, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
    )
    return np.asarray(covariance, dtype=np.float64)


def _build_grouped_weight_table(
    *,
    distribution_weights: np.ndarray,
    label_matrix: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    grouped_weights = np.zeros((distribution_weights.shape[0], n_groups), dtype=np.float64)
    for dist_idx in range(label_matrix.shape[0]):
        for comp_idx in range(label_matrix.shape[1]):
            group_idx = int(label_matrix[dist_idx, comp_idx])
            if group_idx < 0:
                continue
            grouped_weights[dist_idx, group_idx] += float(distribution_weights[dist_idx, comp_idx])
    return grouped_weights


def _canonicalize_group_order(
    *,
    label_matrix: np.ndarray,
    grouped_mu: np.ndarray,
    grouped_var: np.ndarray | None,
    grouped_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Reorder groups by descending total mass, breaking ties by ascending mean coordinates (dim 0 first)."""
    total_mass = grouped_weights.sum(axis=0)
    if grouped_mu.shape[0] <= 1:
        return label_matrix, grouped_mu, grouped_var, grouped_weights

    sort_keys = [grouped_mu[:, dim] for dim in range(grouped_mu.shape[1] - 1, -1, -1)]
    sort_keys.append(-total_mass)
    order = np.lexsort(tuple(sort_keys))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=int)

    reordered_matrix = np.array(label_matrix, copy=True)
    active_mask = reordered_matrix >= 0
    reordered_matrix[active_mask] = inverse[reordered_matrix[active_mask]]

    reordered_mu = grouped_mu[order]
    reordered_var = None if grouped_var is None else grouped_var[order]
    reordered_weights = grouped_weights[:, order]
    return reordered_matrix, reordered_mu, reordered_var, reordered_weights


def _default_grouping_key(
    *,
    gmm_key: str,
    grouping_method: str,
    group_representative: str,
    barycenter_weighting: str,
    n_groups: int,
) -> str:
    return f"{gmm_key}_grouped_{grouping_method}_{group_representative}_{barycenter_weighting}_{int(n_groups)}"


def _gmm_source_checksum(
    *,
    gmm_cfg: dict[str, Any],
    mu: np.ndarray,
    covariances: np.ndarray,
    distribution_weights: np.ndarray,
    distribution_n_components: np.ndarray,
) -> str:
    hasher = hashlib.sha1()
    hasher.update(str(gmm_cfg.get("component_sharing", "")).encode("utf-8"))
    hasher.update(str(gmm_cfg.get("covariance_type", "")).encode("utf-8"))
    for array in (mu, covariances, distribution_weights, distribution_n_components):
        arr = np.ascontiguousarray(array)
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(arr.tobytes())
    distribution_ids = _resolve_distribution_ids(gmm_cfg)
    hasher.update("|".join(distribution_ids).encode("utf-8"))
    return hasher.hexdigest()


__all__ = [
    "group_components",
]
