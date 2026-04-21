from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from ggml_ot._utils._array import to_numpy as _to_numpy
from ggml_ot._utils._covariance import canonicalize_covariances
from ggml_ot._utils._weights import normalize_weight_rows


def _iter_distribution_supports(dataset) -> list[np.ndarray]:
    """Normalize dataset supports to a list of numpy arrays."""
    supports = dataset.supports
    if isinstance(supports, torch.Tensor):
        if supports.ndim == 2:
            return [np.asarray(supports.detach().cpu().numpy(), dtype=np.float64)]
        return [np.asarray(supports[i].detach().cpu().numpy(), dtype=np.float64) for i in range(supports.shape[0])]
    if isinstance(supports, np.ndarray):
        if supports.ndim == 2:
            return [np.asarray(supports, dtype=np.float64)]
        return [np.asarray(supports[i], dtype=np.float64) for i in range(supports.shape[0])]
    if isinstance(supports, (list, tuple)):
        return [np.asarray(_to_numpy(s), dtype=np.float64) for s in supports]
    raise TypeError(f"Unsupported supports container type: {type(supports)!r}")


def _distribution_labels(dataset, n_distributions: int):
    """Return distribution labels and validate their cardinality."""
    labels = dataset.distribution_labels
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    if labels.shape[0] != n_distributions:
        raise ValueError(
            "distribution_labels length must match number of distributions in supports. "
            f"Got {labels.shape[0]} labels for {n_distributions} distributions."
        )
    return labels.tolist()


def _apply_gmm_fields_in_place(
    dataset,
    *,
    supports,
    covariances,
    weights,
    identical_supports: bool,
    gmm_provenance: str | None = None,
):
    """Apply fitted GMM fields directly on an existing dataset object."""
    dataset.supports = torch.as_tensor(np.asarray(supports), dtype=torch.float32)
    dataset.covariances = torch.as_tensor(np.asarray(covariances), dtype=torch.float32)
    dataset.weights = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
    dataset.identical_supports = bool(identical_supports)
    dataset.dim = int(dataset.supports.shape[-1])
    if hasattr(dataset, "_map_A"):
        dataset._map_A = None
    if gmm_provenance is not None and hasattr(dataset, "synth_data") and isinstance(dataset.synth_data, dict):
        dataset.synth_data["fitted_gmm_provenance"] = gmm_provenance
    return dataset


def _extract_fit_data_from_dataset(
    dataset,
    *,
    component_sharing: Literal["global", "sample_specific"],
):
    """Build matrix-level fit inputs from generic dataset supports."""
    if bool(dataset.identical_supports):
        raise ValueError(
            "TripletDataset.fit_gmm with identical_supports=True is unsupported for generic datasets, "
            "because only shared representative supports are available. "
            "Use AnnData_TripletDataset.fit_gmm(...) to refit from full AnnData."
        )

    supports = _iter_distribution_supports(dataset)
    labels = _distribution_labels(dataset, len(supports))

    if component_sharing == "sample_specific" and any(s.shape[0] == 0 for s in supports):
        raise ValueError("All distributions must contain at least one support point for sample-specific fitting.")

    distribution_ids = np.arange(len(supports), dtype=int)
    # Flatten per-distribution supports into matrix + distribution index encoding.
    x = np.vstack([np.asarray(s, dtype=np.float64) for s in supports])
    distribution_index = np.concatenate(
        [np.full(s.shape[0], i, dtype=int) for i, s in enumerate(supports)],
        axis=0,
    )
    return x, distribution_index, distribution_ids, np.asarray(labels)


def _store_gmm_in_dataset(
    dataset,
    *,
    fit_outputs: dict,
    covariance_type: Literal["diag", "full"],
    component_sharing: Literal["global", "sample_specific"],
    distribution_labels: np.ndarray,
):
    """Apply matrix-level GMM fit outputs to a generic dataset object."""
    model_payload = fit_outputs["model_payload"]
    weights_out = np.asarray(fit_outputs["distribution_weights"], dtype=np.float64)
    if weights_out.ndim == 1:
        weights_out = weights_out.reshape(1, -1)
    weights_out = normalize_weight_rows(weights_out)

    if component_sharing == "global":
        supports_out = np.asarray(model_payload["mu"], dtype=np.float64)
        covariances_out = np.asarray(model_payload["var"], dtype=np.float64)
        # Global fits may still be serialized with a singleton distribution axis.
        if supports_out.ndim == 3 and supports_out.shape[0] == 1:
            supports_out = supports_out[0]
        if covariances_out.ndim in (3, 4) and covariances_out.shape[0] == 1:
            covariances_out = covariances_out[0]
        identical_supports_out = True
    else:
        supports_out = np.asarray(model_payload["mu"], dtype=np.float64)
        covariances_out = np.asarray(model_payload["var"], dtype=np.float64)
        identical_supports_out = False

    covariances_out = canonicalize_covariances(
        covariances_out,
        supports_out,
        covariance_type=covariance_type,
    )

    dataset.distribution_labels = np.asarray(distribution_labels)
    return _apply_gmm_fields_in_place(
        dataset,
        supports=supports_out,
        covariances=covariances_out,
        weights=weights_out,
        identical_supports=identical_supports_out,
        gmm_provenance="fit_gmm",
    )


__all__ = [
    "_apply_gmm_fields_in_place",
    "_extract_fit_data_from_dataset",
    "_store_gmm_in_dataset",
]
