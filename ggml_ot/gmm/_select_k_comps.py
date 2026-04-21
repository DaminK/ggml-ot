from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from ggml_ot._utils._array import to_numpy as _to_numpy
from ggml_ot._utils._splits import (
    split_train_val_indices,
)

if TYPE_CHECKING:
    from ._fit_core import GMMFitConfig, GMMResult


def _iter_ks(k_candidates: list[int]) -> list[int]:
    """Sort and validate candidate component counts."""
    ks = sorted({int(k) for k in k_candidates})
    if len(ks) == 0 or any(k < 1 for k in ks):
        raise ValueError("k_comps must contain positive integers.")
    return ks


def parse_k_comps(k_comps: int | list[int] | tuple[int, ...]) -> tuple[int | None, list[int] | None]:
    """Resolve fixed-K vs candidate-K configuration from ``k_comps``."""
    if isinstance(k_comps, (int, np.integer)):
        k = int(k_comps)
        if k < 1:
            raise ValueError("k_comps must be >= 1.")
        return k, None

    k_candidates = _iter_ks(list(k_comps))
    if len(k_candidates) == 1:
        return int(k_candidates[0]), None
    return None, k_candidates


def total_log_likelihood(model: object, x: np.ndarray) -> float:
    """Compute total log-likelihood for model and samples."""
    x_np = np.asarray(x, dtype=np.float32)

    if hasattr(model, "score_samples"):
        device = model.mu.device if hasattr(model, "mu") else torch.device("cpu")
        x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        ll = model.score_samples(x_t)
        ll_np = _to_numpy(ll)
        return float(np.sum(ll_np))

    if hasattr(model, "log_probability"):
        ll = model.log_probability(x_np)
        ll_np = _to_numpy(ll)
        return float(np.sum(ll_np))

    raise TypeError("Unsupported fitted model type for log-likelihood evaluation.")


def _free_parameter_count(k: int, d: int, covariance_type: str) -> int:
    """Compute free parameter count for information criteria."""
    if covariance_type == "diag":
        cov_params = k * d
    else:
        cov_params = k * (d * (d + 1) // 2)
    return (k * d) + cov_params + (k - 1)


def bic_from_result(result: object, x: np.ndarray, config: object) -> float:
    """Calculate Bayesian Information Criterion (BIC)."""
    if getattr(result, "bic", None) is not None:
        return float(result.bic)
    ll_total = total_log_likelihood(result.model, x)
    n_samples = max(1, int(x.shape[0]))
    p = _free_parameter_count(int(config.n_components), int(x.shape[1]), str(config.covariance_type))
    return float(-2.0 * ll_total + p * np.log(n_samples))


def aic_from_result(result: object, x: np.ndarray, config: object) -> float:
    """Calculate Akaike Information Criterion (AIC)."""
    ll_total = total_log_likelihood(result.model, x)
    p = _free_parameter_count(int(config.n_components), int(x.shape[1]), str(config.covariance_type))
    return float(-2.0 * ll_total + 2.0 * p)


def heldout_nll_from_model(model: object, x_val: np.ndarray) -> float:
    """Calculate normalized negative log-likelihood on held-out samples."""
    ll_val = total_log_likelihood(model, x_val)
    return float(-ll_val / max(1, int(x_val.shape[0])))


def _score_information_metric(metric: str, result: object, x: np.ndarray, config: object) -> float:
    """Score one fitted result with an information-criterion metric."""
    if metric == "bic":
        return bic_from_result(result, x, config)
    if metric == "aic":
        return aic_from_result(result, x, config)
    raise NotImplementedError("k_selection_metric must be one of {'bic', 'aic', 'heldout_nll'}.")


def _fit_and_score_for_k(
    *,
    x: np.ndarray,
    x_val: np.ndarray | None,
    config: GMMFitConfig,
    metric: str,
    fit_fn: Callable[..., GMMResult],
) -> tuple[GMMResult, dict[str, float | int]]:
    """Fit one candidate K on a matrix and return the fitted result plus one score record."""
    if metric == "heldout_nll":
        if x_val is None:
            raise ValueError("heldout_nll scoring requires a validation matrix.")
        result = fit_fn(x=x, config=config)
        train_nll = heldout_nll_from_model(result.model, x)
        validation_nll = heldout_nll_from_model(result.model, x_val)
        return result, {
            "k": int(config.n_components),
            "score": float(validation_nll),
            "train_nll": float(train_nll),
            "validation_nll": float(validation_nll),
            "nll_gap": float(validation_nll - train_nll),
            "n_train_cells": int(x.shape[0]),
            "n_val_cells": int(x_val.shape[0]),
        }

    result = fit_fn(x=x, config=config)
    score = _score_information_metric(metric, result, x, config)
    return result, {"k": int(config.n_components), "score": float(score)}


def _select_best_k_on_matrix(
    *,
    x: np.ndarray,
    base_config: GMMFitConfig,
    metric: str,
    k_candidates: list[int],
    fit_fn: Callable[..., GMMResult],
    return_best_result: bool,
    x_val: np.ndarray | None = None,
    verbose: bool = False,
    tqdm_desc: str | None = None,
) -> tuple[int, list[dict[str, float | int]], GMMResult | None]:
    """Select best K for one matrix input."""
    scores: list[dict[str, float | int]] = []
    best_k: int | None = None
    best_score = np.inf
    best_result: GMMResult | None = None

    ks = _iter_ks(k_candidates)
    k_iter = tqdm(ks, desc=tqdm_desc, leave=False) if verbose and tqdm_desc is not None else ks
    for k in k_iter:
        cfg = replace(base_config, n_components=int(k))
        result, score = _fit_and_score_for_k(
            x=x,
            x_val=x_val,
            config=cfg,
            metric=metric,
            fit_fn=fit_fn,
        )
        scores.append(score)
        score_value = float(score["score"])
        if score_value < float(best_score):
            best_score = float(score_value)
            best_k = int(k)
            best_result = result

    if best_k is None:
        raise RuntimeError("K-selection failed to produce a best_k value.")

    return best_k, scores, best_result if return_best_result else None


def _select_best_k_sample_specific(
    x_by_distribution: dict[str, np.ndarray],
    base_config: GMMFitConfig,
    metric: str,
    k_candidates: list[int],
    *,
    fit_single_fn: Callable[..., GMMResult],
    x_val_by_distribution: dict[str, np.ndarray] | None = None,
    return_best_result: bool = False,
    verbose: bool = False,
) -> tuple[dict[str, int], dict[str, list[dict[str, float | int]]], dict[str, GMMResult] | None]:
    """Select best K independently for each distribution in sample-specific mode."""
    best_k_by_distribution: dict[str, int] = {}
    scores_by_distribution: dict[str, list[dict[str, float | int]]] = {}
    best_results: dict[str, GMMResult] | None = {} if return_best_result else None

    dist_iter = (
        tqdm(x_by_distribution.items(), desc="[fit_gmm] selecting k per distribution", leave=False)
        if verbose
        else x_by_distribution.items()
    )
    for distribution_id, x_d in dist_iter:
        best_k, scores, best_result = _select_best_k_on_matrix(
            x=x_d,
            x_val=None if x_val_by_distribution is None else x_val_by_distribution[distribution_id],
            base_config=base_config,
            metric=metric,
            k_candidates=k_candidates,
            fit_fn=fit_single_fn,
            return_best_result=return_best_result,
        )
        best_k_by_distribution[distribution_id] = int(best_k)
        scores_by_distribution[distribution_id] = scores
        if return_best_result and best_results is not None and best_result is not None:
            best_results[distribution_id] = best_result

    return best_k_by_distribution, scores_by_distribution, best_results


def _scores_to_columnar(scores: list[dict[str, float | int]]) -> dict[str, list[float | int]]:
    """Convert record lists into a columnar layout safe for h5ad/h5py serialization.

    Example:
    ``[{"k": 1, "score": 0.5}, ...]`` -> ``{"k": [1, ...], "score": [0.5, ...]}``
    """
    if not scores:
        return {}
    keys = scores[0].keys()
    columnar: dict[str, list[float | int]] = {}
    for key in keys:
        values: list[float | int] = []
        for record in scores:
            value = record[key]
            if isinstance(value, np.generic):
                value = value.item()
            values.append(value)
        columnar[key] = values
    return columnar


def _aggregate_scores_by_k(
    *,
    k_candidates: list[int],
    scores_by_distribution: dict[str, list[dict[str, float | int]]],
) -> list[dict[str, float | int]]:
    """Aggregate sample-specific per-distribution score records into one mean trace by K."""
    score_groups: dict[int, list[dict[str, float | int]]] = {int(k): [] for k in k_candidates}
    for dist_scores in scores_by_distribution.values():
        for item in dist_scores:
            score_groups[int(item["k"])].append(item)

    aggregated: list[dict[str, float | int]] = []
    for k in sorted(score_groups.keys()):
        records = score_groups[int(k)]
        if len(records) == 0:
            continue
        mean_record: dict[str, float | int] = {"k": int(k)}
        for key in records[0].keys():
            if key == "k":
                continue
            mean_record[key] = float(np.mean([float(record[key]) for record in records]))
        aggregated.append(mean_record)
    return aggregated


def _validate_train_size(train_size: float) -> float:
    """Validate the public training fraction used during k-selection."""
    train_frac = float(train_size)
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_size must satisfy 0 < train_size < 1.")
    return train_frac


def _split_sample_specific_train_val(
    *,
    x: np.ndarray,
    distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
    n_components: int,
    train_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split each distribution into train/validation blocks and return the global train index."""
    train_selection: list[np.ndarray] = []
    x_train_by_distribution: dict[str, np.ndarray] = {}
    x_val_by_distribution: dict[str, np.ndarray] = {}

    for d_idx, distribution_id in enumerate(distribution_ids):
        group_idx = np.where(distribution_index == d_idx)[0]
        train_local, val_local = split_train_val_indices(
            group_idx.shape[0],
            n_components=n_components,
            train_frac=train_frac,
            rng=rng,
        )
        train_idx = group_idx[train_local]
        val_idx = group_idx[val_local]
        distribution_key = str(distribution_id)
        train_selection.append(train_idx)
        x_train_by_distribution[distribution_key] = x[train_idx]
        x_val_by_distribution[distribution_key] = x[val_idx]

    return np.sort(np.concatenate(train_selection)), x_train_by_distribution, x_val_by_distribution


def run_k_selection(
    *,
    x: np.ndarray,
    distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
    component_sharing: str,
    config: GMMFitConfig,
    k_candidates: list[int] | None,
    k_selection_metric: str,
    refit: str,
    train_size: float,
    rng: np.random.Generator,
    verbose: bool = False,
) -> tuple[
    GMMFitConfig,
    dict[str, object] | None,
    np.ndarray | None,
    GMMResult | None,
    dict[str, GMMResult] | None,
    dict[str, int] | None,
]:
    """Run optional K-selection and return updated config + artifacts."""
    if k_candidates is None:
        return config, None, None, None, None, None

    # Local import avoids cycles; this module only orchestrates scoring + selection.
    from ._GaussianMixture import GaussianMixture

    selected_global_result: GMMResult | None = None
    selected_sample_results: dict[str, GMMResult] | None = None
    selected_k_by_distribution: dict[str, int] | None = None
    scores_by_distribution: dict[str, list[dict[str, float | int]]] | None = None
    train_frac = _validate_train_size(train_size)

    max_k = int(max(k_candidates))
    if component_sharing == "global":
        selection_idx, val_idx = split_train_val_indices(
            x.shape[0],
            n_components=max_k,
            train_frac=train_frac,
            rng=rng,
        )
        x_fit = x[selection_idx]
        x_val = x[val_idx] if k_selection_metric == "heldout_nll" else None
        best_k, scores, selected_global_result = _select_best_k_on_matrix(
            x=x_fit,
            x_val=x_val,
            base_config=config,
            metric=k_selection_metric,
            k_candidates=k_candidates,
            fit_fn=GaussianMixture.fit_from_numpy,
            return_best_result=(refit == "none"),
            verbose=verbose,
            tqdm_desc="[fit_gmm] evaluating k (global)",
        )
    else:
        selection_idx, x_by_distribution, x_val_by_distribution = _split_sample_specific_train_val(
            x=x,
            distribution_index=distribution_index,
            distribution_ids=distribution_ids,
            n_components=max_k,
            train_frac=train_frac,
            rng=rng,
        )
        selected_k_by_distribution, scores_by_distribution, selected_sample_results = _select_best_k_sample_specific(
            x_by_distribution=x_by_distribution,
            base_config=config,
            metric=k_selection_metric,
            k_candidates=k_candidates,
            fit_single_fn=GaussianMixture.fit_from_numpy,
            x_val_by_distribution=x_val_by_distribution if k_selection_metric == "heldout_nll" else None,
            return_best_result=(refit == "none"),
            verbose=verbose,
        )
        best_k = int(max(selected_k_by_distribution.values()))
        scores = _aggregate_scores_by_k(
            k_candidates=k_candidates,
            scores_by_distribution=scores_by_distribution,
        )

    updated_config = replace(config, n_components=int(best_k))
    selection_meta = {
        "metric": k_selection_metric,
        "k_comps": [int(k) for k in k_candidates],
        "best_k": int(best_k),
        "scores": _scores_to_columnar(scores),
        "selection_cells_count": int(len(selection_idx)),
        "refit": refit,
        "train_frac": float(train_frac),
        "val_frac": float(1.0 - train_frac),
    }
    if k_selection_metric == "heldout_nll":
        selection_meta["score_name"] = "validation_nll"
    if selected_k_by_distribution is not None:
        selection_meta["best_k_by_distribution"] = {str(k): int(v) for k, v in selected_k_by_distribution.items()}
        selection_meta["scores_by_distribution"] = {
            dist_id: _scores_to_columnar(dist_scores) for dist_id, dist_scores in scores_by_distribution.items()
        }
    return (
        updated_config,
        selection_meta,
        selection_idx,
        selected_global_result,
        selected_sample_results,
        selected_k_by_distribution,
    )


__all__ = [
    "aic_from_result",
    "bic_from_result",
    "heldout_nll_from_model",
    "parse_k_comps",
    "run_k_selection",
    "total_log_likelihood",
]
