from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm

from ggml_ot._utils._splits import split_by_group
from ggml_ot._utils._weights import normalize_weight_rows, normalize_weight_vector


def fit_sample_specific_models(
    *,
    x_fit: np.ndarray,
    fit_distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
    config,
    has_selection: bool,
    refit: Literal["full", "none"],
    selected_sample_results,
    selected_k_by_distribution: dict[str, int] | None,
    verbose: bool,
) -> dict[str, Any]:
    """Return fitted sample-specific models for each distribution id."""
    from ._GaussianMixture import GaussianMixture

    if has_selection and refit == "none":
        if selected_sample_results is None:
            raise RuntimeError("Expected selected sample-specific results when refit='none'.")
        return selected_sample_results

    x_by_distribution = split_by_group(
        x=x_fit,
        distribution_index=fit_distribution_index,
        distribution_ids=distribution_ids,
    )

    if has_selection and selected_k_by_distribution is not None:
        results: dict[str, Any] = {}
        dist_iter = (
            tqdm(x_by_distribution.items(), desc="[fit_gmm] refit distributions", leave=False)
            if verbose
            else x_by_distribution.items()
        )
        for dist_id, x_d in dist_iter:
            k_d = int(selected_k_by_distribution[str(dist_id)])
            cfg_d = replace(config, n_components=k_d)
            results[str(dist_id)] = GaussianMixture.fit_from_numpy(x=x_d, config=cfg_d)
        return results

    return GaussianMixture.fit_many_from_numpy(
        x_by_distribution=x_by_distribution,
        config=config,
        verbose=verbose,
    )


def build_padded_sample_specific_outputs(
    *,
    results: dict[str, Any],
    distribution_ids: np.ndarray,
    x: np.ndarray,
    full_distribution_index: np.ndarray,
    covariance_type: Literal["diag", "full"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Pad per-distribution fitted outputs to ``K_max`` and assemble canonical payload."""
    per_dist_mu: list[np.ndarray] = []
    per_dist_var: list[np.ndarray] = []
    per_dist_pi: list[np.ndarray] = []
    per_dist_k: list[int] = []

    for dist in distribution_ids:
        result = results[str(dist)]
        mu_d = np.asarray(result.mu, dtype=np.float64).reshape(-1, x.shape[1])
        k_d = int(mu_d.shape[0])
        if k_d < 1:
            raise RuntimeError(f"Fitted sample-specific model for distribution {dist!r} has no components.")
        if covariance_type == "diag":
            var_d = np.asarray(result.var, dtype=np.float64).reshape(k_d, x.shape[1])
        else:
            var_d = np.asarray(result.var, dtype=np.float64).reshape(k_d, x.shape[1], x.shape[1])
        pi_d = normalize_weight_vector(np.asarray(result.pi).reshape(-1))
        if pi_d.shape[0] != k_d:
            raise RuntimeError(
                f"Distribution {dist!r} produced inconsistent component count between mu ({k_d}) and pi ({pi_d.shape[0]})."
            )
        per_dist_mu.append(mu_d)
        per_dist_var.append(var_d)
        per_dist_pi.append(pi_d)
        per_dist_k.append(k_d)

    k_max = int(max(per_dist_k))
    d = int(x.shape[1])

    mu = []
    var = []
    pi = []
    responsibilities = np.zeros((x.shape[0], k_max), dtype=np.float64)
    hard_components = np.zeros(x.shape[0], dtype=int)
    distribution_weights = np.zeros((len(distribution_ids), k_max), dtype=np.float64)

    for d_idx, dist in enumerate(distribution_ids):
        result = results[str(dist)]
        k_d = int(per_dist_k[d_idx])

        mu_pad = np.zeros((k_max, d), dtype=np.float64)
        mu_pad[:k_d] = per_dist_mu[d_idx]
        if covariance_type == "diag":
            var_pad = np.zeros((k_max, d), dtype=np.float64)
        else:
            var_pad = np.zeros((k_max, d, d), dtype=np.float64)
        var_pad[:k_d] = per_dist_var[d_idx]
        pi_pad = np.zeros((k_max,), dtype=np.float64)
        pi_pad[:k_d] = per_dist_pi[d_idx]

        mu.append(mu_pad)
        var.append(var_pad)
        pi.append(pi_pad)
        distribution_weights[d_idx] = pi_pad

        mask = full_distribution_index == d_idx
        resp_local = result.model.predict_responsibilities_numpy(x[mask])
        if resp_local.shape[1] != k_d:
            raise RuntimeError(
                f"Distribution {dist!r} produced inconsistent responsibilities width ({resp_local.shape[1]}) vs k={k_d}."
            )
        resp_pad = np.zeros((resp_local.shape[0], k_max), dtype=np.float64)
        resp_pad[:, :k_d] = resp_local
        responsibilities[mask] = resp_pad
        hard_components[mask] = np.argmax(resp_pad, axis=1)

    model_payload = {
        "mu": np.stack(mu, axis=0),
        "var": np.stack(var, axis=0),
        "pi": normalize_weight_rows(np.stack(pi, axis=0)),
    }
    return (
        hard_components,
        responsibilities,
        normalize_weight_rows(distribution_weights),
        model_payload,
        np.asarray(per_dist_k, dtype=int),
    )
