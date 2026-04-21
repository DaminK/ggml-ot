from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from ggml_ot._utils._anndata_utils import (
    get_cell_feature_matrix,
    get_distribution_ids,
    get_distribution_index,
    get_patient_label_indices,
    patient_labels,
)
from ggml_ot._utils._covariance import canonicalize_covariances
from ggml_ot._utils._weights import (
    canonicalize_model_weights,
    get_distribution_weights,
    normalize_weight_rows,
    normalize_weights,
)
from ggml_ot._utils._validate import sanitize_degenerate_sample_specific_gmm_components
from ggml_ot.data._parameter_setup import set_preprocessed
from ._GaussianMixture import GaussianMixture


def _store_gmm_in_adata(
    *,
    adata,
    gmm_key: str,
    component_sharing: Literal["global", "sample_specific"],
    use_rep: str | None,
    covariance_type: Literal["diag", "full"],
    n_components: int,
    distribution_n_components: np.ndarray,
    fit_params: dict[str, Any],
    selection: dict[str, Any] | None,
    backend: str,
    backend_metadata: dict[str, Any],
    model_payload: dict[str, Any],
    distribution_weights,
    distribution_ids,
    hard_components: np.ndarray | None = None,
    responsibilities: np.ndarray | None = None,
    weights_fit_scope: str = "all_cells",
    weights_source: str = "stored",
) -> None:
    """Persist a canonical GMM schema under ``adata.uns[gmm_key]``.

    Parameters
    ----------
    adata
        AnnData object receiving the persisted schema.
    gmm_key
        Key under ``adata.uns`` where the schema is written.
    component_sharing
        GMM component sharing mode.
    use_rep
        Representation used for the persisted GMM, or ``None`` for raw ``.X``.
    covariance_type
        Covariance parameterization of the persisted model.
    n_components
        Maximum number of components in the persisted representation.
    distribution_n_components
        Per-distribution component counts with shape ``(D,)``.
    fit_params
        Metadata describing fit-time parameters.
    selection
        Optional K-selection metadata.
    backend
        Backend/provenance label stored in the schema.
    backend_metadata
        Additional backend metadata stored verbatim in the schema.
    model_payload
        Model parameter payload containing ``mu``, ``var``, and ``pi``.
    distribution_weights
        Per-distribution component weights.
    distribution_ids
        Distribution identifiers aligned to the first axis of the schema.
    hard_components
        Optional per-cell hard assignments written to ``adata.obs``.
    responsibilities
        Optional per-cell responsibilities written to ``adata.obsm``.
    weights_fit_scope
        Provenance label for how the distribution weights were obtained.
    weights_source
        Provenance label for where the stored distribution weights came from.
    """
    distribution_n_components = np.asarray(distribution_n_components, dtype=int)
    distribution_ids = [str(x) for x in np.asarray(distribution_ids).tolist()]
    if distribution_n_components.ndim != 1 or distribution_n_components.shape[0] != len(distribution_ids):
        raise ValueError(
            "distribution_n_components must be a 1D vector with one entry per distribution. "
            f"Got shape={distribution_n_components.shape}, expected ({len(distribution_ids)},)."
        )

    distribution_weights = normalize_weight_rows(np.asarray(distribution_weights, dtype=np.float64))
    mu_payload = np.asarray(model_payload["mu"], dtype=np.float64)
    var_payload = canonicalize_covariances(
        np.asarray(model_payload["var"], dtype=np.float64),
        mu_payload,
        covariance_type=covariance_type,
    )
    pi_payload = np.asarray(model_payload["pi"], dtype=np.float64)
    if pi_payload.ndim == 1:
        pi_payload = pi_payload.reshape(1, -1)
    if pi_payload.ndim == 2:
        pi_payload = normalize_weight_rows(pi_payload)

    if hard_components is not None:
        adata.obs[f"{gmm_key}_comp"] = np.asarray(hard_components)
    if responsibilities is not None:
        adata.obsm[f"{gmm_key}_resp"] = np.asarray(responsibilities, dtype=np.float64)

    adata.uns[gmm_key] = {
        "component_sharing": component_sharing,
        "use_rep": use_rep,
        "covariance_type": covariance_type,
        "n_components": int(n_components),
        "distribution_n_components": distribution_n_components.tolist(),
        "fit_params": dict(fit_params),
        "selection": selection,
        "backend": backend,
        "model_family": "gmm",
        "representation_space": "use_rep",
        "backend_metadata": dict(backend_metadata),
        "model": {
            "mu": mu_payload,
            "var": var_payload,
            "pi": pi_payload,
        },
        "distribution_weights": distribution_weights,
        "weight_inference": {
            "weights_fit_scope": weights_fit_scope,
            "weights_source": weights_source,
            "distribution_ids": distribution_ids,
        },
    }


def _distribution_ids_for_gmm(gmm_cfg: dict, *, gmm_key: str) -> list[str]:
    """Read persisted distribution ids from ``adata.uns[gmm_key]`` schema."""
    distribution_ids = gmm_cfg.get("weight_inference", {}).get("distribution_ids")
    if distribution_ids is None:
        raise KeyError(
            f"Missing adata.uns[{gmm_key!r}]['weight_inference']['distribution_ids']; "
            "cannot align per-distribution GMM outputs."
        )
    return [str(x) for x in distribution_ids]


def _validate_component_sharing_shapes(
    *, supports: np.ndarray, covariances: np.ndarray, component_sharing: str
) -> None:
    """Validate persisted model tensor shapes for selected sharing mode."""
    if component_sharing == "sample_specific":
        if supports.ndim != 3:
            raise ValueError("For sample_specific schema, model['mu'] must have shape (D, K, d).")
        if covariances.ndim not in (3, 4):
            raise ValueError("For sample_specific schema, model['var'] must have shape (D, K, d) or (D, K, d, d).")
        return
    if supports.ndim != 2:
        raise ValueError("For global/shared-support schema, model['mu'] must have shape (K, d).")


def _distribution_model_parameters(
    *,
    supports: np.ndarray,
    covariances: np.ndarray,
    component_sharing: str,
    dist_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, var)`` arrays for one distribution in canonical shape."""
    if component_sharing == "sample_specific":
        return (
            np.asarray(supports[dist_idx], dtype=np.float64),
            np.asarray(covariances[dist_idx], dtype=np.float64),
        )
    return np.asarray(supports, dtype=np.float64), np.asarray(covariances, dtype=np.float64)


def _infer_weights_via_component_prediction(
    adata,
    *,
    gmm_cfg,
    patient_col: str,
    distribution_ids: list[str],
    n_components: int,
    supports: np.ndarray,
    covariances: np.ndarray,
    component_sharing: str,
    covariance_type: str,
    use_rep,
) -> dict[str, np.ndarray]:
    x = get_cell_feature_matrix(adata, use_rep=use_rep)
    patient_values = adata.obs[patient_col].astype(str).to_numpy()
    weight_rows = canonicalize_model_weights(gmm_cfg.get("model", {}).get("pi", None), n_components=n_components)
    n_distributions = len(distribution_ids)
    weights_map: dict[str, np.ndarray] = {}
    _validate_component_sharing_shapes(
        supports=supports,
        covariances=covariances,
        component_sharing=component_sharing,
    )

    for d_idx, dist_id in enumerate(distribution_ids):
        mask = np.asarray(patient_values == dist_id)
        if not np.any(mask):
            raise ValueError(f"Distribution id {dist_id!r} has no cells in adata.obs[{patient_col!r}].")
        weights = get_distribution_weights(
            weight_rows,
            dist_idx=d_idx,
            n_distributions=n_distributions,
            n_components=n_components,
        )
        mu_dist, cov_dist = _distribution_model_parameters(
            supports=supports,
            covariances=covariances,
            component_sharing=component_sharing,
            dist_idx=d_idx,
        )
        model = GaussianMixture.from_dict(
            {
                "mu": mu_dist,
                "var": cov_dist,
                "pi": np.asarray(weights, dtype=np.float64),
                "covariance_type": covariance_type,
            }
        )
        hard = model.predict_hard_components_numpy(x[mask])
        counts = np.bincount(hard, minlength=n_components).astype(np.float64)
        weights_map[dist_id] = normalize_weights(counts)
    return weights_map


def _infer_gmm_distribution_weights(
    adata,
    *,
    gmm_cfg,
    gmm_key: str,
    patient_col: str,
    distribution_ids: list[str],
    n_components: int,
    gmm_weights_source: str,
    supports: np.ndarray,
    covariances: np.ndarray,
    component_sharing: str,
    covariance_type: str,
    use_rep,
) -> dict[str, np.ndarray]:
    valid_sources = {"auto", "stored", "components"}
    if gmm_weights_source not in valid_sources:
        raise ValueError(
            f"Unsupported gmm_weights_source={gmm_weights_source!r}. Expected one of {sorted(valid_sources)}."
        )

    dist_weights = gmm_cfg.get("distribution_weights", None)
    if dist_weights is not None and gmm_weights_source in {"auto", "stored"}:
        dist_weights = np.asarray(dist_weights, dtype=np.float64)
        if dist_weights.ndim != 2:
            raise ValueError("distribution_weights must be a 2D array.")
        if dist_weights.shape[0] != len(distribution_ids):
            raise ValueError("distribution_weights rows must match length of distribution_ids.")
        if dist_weights.shape[1] != n_components:
            raise ValueError("distribution_weights columns must match number of GMM components.")
        return {distribution_ids[i]: normalize_weights(dist_weights[i]) for i in range(len(distribution_ids))}

    if gmm_weights_source == "stored":
        raise ValueError(
            f"gmm_weights_source='stored' requested, but adata.uns[{gmm_key!r}]['distribution_weights'] is missing."
        )

    component_error: Exception | None = None
    if gmm_weights_source in {"auto", "components"}:
        try:
            return _infer_weights_via_component_prediction(
                adata,
                gmm_cfg=gmm_cfg,
                patient_col=patient_col,
                distribution_ids=distribution_ids,
                n_components=n_components,
                supports=supports,
                covariances=covariances,
                component_sharing=component_sharing,
                covariance_type=covariance_type,
                use_rep=use_rep,
            )
        except Exception as exc:
            component_error = exc
            if gmm_weights_source == "components":
                raise ValueError(
                    "gmm_weights_source='components' requested, but component assignments "
                    "could not be inferred from stored GMM parameters."
                ) from exc

    raise ValueError(
        "Could not infer distribution weights for GMM dataset loading. "
        f"Expected one of: adata.uns[{gmm_key!r}]['distribution_weights'], "
        "or inferable component assignments from stored GMM parameters."
    ) from component_error


def _extract_fit_data_from_anndata(
    adata,
    *,
    use_rep: str | None,
    distribution_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract matrix-level fit inputs from AnnData."""
    x = get_cell_feature_matrix(adata, use_rep=use_rep)
    distribution_ids = get_distribution_ids(adata, distribution_col=distribution_col)
    distribution_index = get_distribution_index(
        adata,
        distribution_col=distribution_col,
        distribution_ids=distribution_ids,
    )
    return x, distribution_index, distribution_ids


def _resolve_gmm_use_rep(gmm_cfg: dict, use_rep):
    gmm_use_rep = gmm_cfg.get("use_rep", None)
    if use_rep is not None:
        if gmm_use_rep != use_rep:
            raise ValueError(f"use_rep {use_rep} does not match the one used for GMM fitting {gmm_use_rep}")
        return use_rep
    if gmm_use_rep is not None:
        warnings.warn(
            f"use_rep not provided, but GMM was fitted with use_rep {gmm_use_rep}. Setting use_rep to {gmm_use_rep}."
        )
        return gmm_use_rep
    return None


def _extract_loaded_model(gmm_cfg: dict) -> tuple[np.ndarray, np.ndarray, str, str]:
    model = gmm_cfg["model"]
    supports = np.asarray(model["mu"], dtype=np.float64)
    covariances = canonicalize_covariances(
        np.asarray(model["var"], dtype=np.float64),
        supports,
        covariance_type=str(gmm_cfg["covariance_type"]),
    )
    if supports.ndim == 3 and supports.shape[0] == 1:
        supports = supports[0]
    if covariances.ndim in (3, 4) and covariances.shape[0] == 1:
        covariances = covariances[0]
    return supports, covariances, str(gmm_cfg["component_sharing"]), str(gmm_cfg["covariance_type"])


def _patient_distribution_indices(*, patients: np.ndarray, distribution_ids: list[str], gmm_key: str) -> list[int]:
    dist_to_idx = {distribution_ids[i]: i for i in range(len(distribution_ids))}
    indices: list[int] = []
    for patient in patients:
        patient_key = str(patient)
        d_idx = dist_to_idx.get(patient_key)
        if d_idx is None:
            raise ValueError(f"Distribution id {patient_key!r} not found in stored distribution_ids for {gmm_key!r}.")
        indices.append(int(d_idx))
    return indices


def _build_patient_gmm_dicts(
    *,
    supports: np.ndarray,
    covariances: np.ndarray,
    weights_map: dict[str, np.ndarray],
    patients: np.ndarray,
    distribution_ids: list[str],
    component_sharing: str,
    covariance_type: str,
    gmm_key: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict[str, np.ndarray | str]], bool]:
    patient_keys = [str(p) for p in patients]
    _validate_component_sharing_shapes(
        supports=supports,
        covariances=covariances,
        component_sharing=component_sharing,
    )

    if component_sharing == "sample_specific":
        dist_indices = _patient_distribution_indices(
            patients=patients,
            distribution_ids=distribution_ids,
            gmm_key=gmm_key,
        )
    else:
        dist_indices = [0 for _ in patient_keys]

    supports_by_patient: list[np.ndarray] = []
    covariances_by_patient: list[np.ndarray] = []
    weights_by_patient: list[np.ndarray] = []
    gmm_dicts: list[dict[str, np.ndarray | str]] = []
    for i, patient_key in enumerate(patient_keys):
        mu_dist, cov_dist = _distribution_model_parameters(
            supports=supports,
            covariances=covariances,
            component_sharing=component_sharing,
            dist_idx=int(dist_indices[i]),
        )
        weights_dist = np.asarray(weights_map[patient_key], dtype=np.float64)
        supports_by_patient.append(np.asarray(mu_dist, dtype="f"))
        covariances_by_patient.append(np.asarray(cov_dist, dtype="f"))
        weights_by_patient.append(weights_dist)
        gmm_dicts.append(
            {
                "mu": np.asarray(mu_dist, dtype=np.float64),
                "var": np.asarray(cov_dist, dtype=np.float64),
                "pi": weights_dist,
                "covariance_type": covariance_type,
            }
        )
    return supports_by_patient, covariances_by_patient, weights_by_patient, gmm_dicts, component_sharing == "global"


def _set_preprocessed_from_patient_gmm_dicts(
    adata,
    *,
    supports_by_patient: list[np.ndarray],
    covariances_by_patient: list[np.ndarray],
    weights_by_patient: list[np.ndarray],
    labels: list[int],
    use_rep,
    sample_gmm: bool,
    gmm_dicts: list[dict[str, np.ndarray | str]],
    n_cells: int,
    identical_supports: bool,
):
    if sample_gmm:
        # GaussianMixture.from_dict().sample() is the canonical sampling call.
        # sample() returns (points, labels); we take points as float32 numpy.
        sampled_supports = [
            GaussianMixture.from_dict(d).sample(n_cells)[0].detach().cpu().float().numpy() for d in gmm_dicts
        ]
        return set_preprocessed(
            adata,
            supports=sampled_supports,
            covariances=None,
            weights=None,
            distribution_labels=labels,
            identical_supports=False,
            use_rep=use_rep,
        )

    if identical_supports:
        return set_preprocessed(
            adata,
            supports=np.asarray(supports_by_patient[0], dtype="f"),
            covariances=np.asarray(covariances_by_patient[0], dtype="f"),
            weights=weights_by_patient,
            distribution_labels=labels,
            identical_supports=True,
            use_rep=use_rep,
        )

    return set_preprocessed(
        adata,
        supports=supports_by_patient,
        covariances=covariances_by_patient,
        weights=weights_by_patient,
        distribution_labels=labels,
        identical_supports=False,
        use_rep=use_rep,
    )


def _extract_fitted_anndata_fields(dataset, *, gmm_key: str):
    """Extract fitted supports/covariances/weights from ``dataset.adata.uns[gmm_key]``."""
    gmm_cfg = dataset.adata.uns[gmm_key]
    model = gmm_cfg["model"]
    component_sharing = gmm_cfg["component_sharing"]

    supports = np.asarray(model["mu"], dtype=np.float64)
    covariances = canonicalize_covariances(
        np.asarray(model["var"], dtype=np.float64),
        supports,
        covariance_type=str(gmm_cfg.get("covariance_type", "full")),
    )
    distribution_weights = np.asarray(gmm_cfg["distribution_weights"], dtype=np.float64)
    if distribution_weights.ndim == 1:
        distribution_weights = distribution_weights.reshape(1, -1)
    distribution_weights = normalize_weight_rows(distribution_weights)

    distribution_ids = gmm_cfg.get("weight_inference", {}).get("distribution_ids")
    if distribution_ids is None:
        distribution_ids = list(np.sort(np.unique(dataset.adata.obs[dataset.patient_col].astype(str).to_numpy())))
    distribution_ids = [str(x) for x in distribution_ids]

    patient_order = [str(p) for p in dataset.patient_labels]
    dist_to_idx = {distribution_ids[i]: i for i in range(len(distribution_ids))}

    missing = [p for p in patient_order if p not in dist_to_idx]
    if missing:
        raise ValueError(f"Patients missing in fitted distribution ids for gmm_key={gmm_key!r}: {missing}")

    if component_sharing == "global":
        if supports.ndim == 3 and supports.shape[0] == 1:
            supports = supports[0]
        if covariances.ndim in (3, 4) and covariances.shape[0] == 1:
            covariances = covariances[0]
        weights = np.stack([distribution_weights[dist_to_idx[p]] for p in patient_order], axis=0)
        return supports, covariances, weights, True

    supports = np.stack([supports[dist_to_idx[p]] for p in patient_order], axis=0)
    covariances = np.stack([covariances[dist_to_idx[p]] for p in patient_order], axis=0)
    weights = np.stack([distribution_weights[dist_to_idx[p]] for p in patient_order], axis=0)
    return supports, covariances, weights, False


def gmm_from_anndata(
    adata,
    patient_col,
    label_col,
    use_rep,
    gmm_key,
    n_cells,
    sample_gmm,
    gmm_weights_source,
):
    """Reconstruct GMM-based dataset inputs from ``adata.uns[gmm_key]``.

    Ownership: this function loads the stored GMM schema, reconstructs per-patient
    GaussianMixture objects, infers distribution weights, and either samples new
    supports (sample_gmm=True) or uses the stored component parameters directly.
    The canonical sampling call is ``GaussianMixture.from_dict(d).sample(n)[0].float().numpy()``.
    """
    if gmm_key not in adata.uns:
        raise KeyError(f"GMM not fitted yet; gmm_key {gmm_key} not in adata.uns keys {adata.uns.keys()}")
    gmm_cfg = adata.uns[gmm_key]
    use_rep = _resolve_gmm_use_rep(gmm_cfg, use_rep=use_rep)
    supports, covariances, component_sharing, covariance_type = _extract_loaded_model(gmm_cfg)
    distribution_ids = _distribution_ids_for_gmm(gmm_cfg, gmm_key=gmm_key)
    patients = patient_labels(adata, patient_col)
    labels = get_patient_label_indices(adata, patient_col=patient_col, label_col=label_col)

    n_components = int(supports.shape[1] if supports.ndim == 3 else supports.shape[0])
    weights_map = _infer_gmm_distribution_weights(
        adata,
        gmm_cfg=gmm_cfg,
        gmm_key=gmm_key,
        patient_col=patient_col,
        distribution_ids=distribution_ids,
        n_components=n_components,
        gmm_weights_source=gmm_weights_source,
        supports=supports,
        covariances=covariances,
        component_sharing=component_sharing,
        covariance_type=covariance_type,
        use_rep=use_rep,
    )
    if component_sharing == "sample_specific":
        sanitized_weights = sanitize_degenerate_sample_specific_gmm_components(
            supports=np.asarray(supports, dtype=np.float64),
            distribution_weights=np.stack(
                [np.asarray(weights_map[str(dist_id)], dtype=np.float64) for dist_id in distribution_ids],
                axis=0,
            ),
            distribution_ids=distribution_ids,
            eps=float(gmm_cfg.get("fit_params", {}).get("eps", 1.0e-4)),
            stage=f"loading gmm_key={gmm_key!r}",
        )
        weights_map = {
            str(distribution_ids[idx]): np.asarray(sanitized_weights[idx], dtype=np.float64)
            for idx in range(len(distribution_ids))
        }
    supports_by_patient, covariances_by_patient, weights_by_patient, gmm_dicts, identical_supports = (
        _build_patient_gmm_dicts(
            supports=supports,
            covariances=covariances,
            weights_map=weights_map,
            patients=patients,
            distribution_ids=distribution_ids,
            component_sharing=component_sharing,
            covariance_type=covariance_type,
            gmm_key=gmm_key,
        )
    )
    return _set_preprocessed_from_patient_gmm_dicts(
        adata,
        supports_by_patient=supports_by_patient,
        covariances_by_patient=covariances_by_patient,
        weights_by_patient=weights_by_patient,
        labels=labels,
        use_rep=use_rep,
        sample_gmm=sample_gmm,
        gmm_dicts=gmm_dicts,
        n_cells=n_cells,
        identical_supports=identical_supports,
    )


__all__ = [
    "_extract_fit_data_from_anndata",
    "_extract_fitted_anndata_fields",
    "_store_gmm_in_adata",
    "gmm_from_anndata",
]
