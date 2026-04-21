"""Fitting GMMs on datasets."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from ggml_ot import settings
from ggml_ot._utils._validate import (
    sanitize_degenerate_sample_specific_gmm_components,
    validate_fit_inputs,
)
from ggml_ot._utils._weights import (
    aggregate_distribution_weights,
    normalize_weight_rows,
    normalize_weight_vector,
)

from ._anndata_dataset_interface import (
    _extract_fit_data_from_anndata,
    _extract_fitted_anndata_fields,
    _store_gmm_in_adata,
)
from ._fit_core import GMMFitConfig
from ._generic_dataset_interface import (
    _apply_gmm_fields_in_place,
    _extract_fit_data_from_dataset,
    _store_gmm_in_dataset,
)
from ._select_k_comps import parse_k_comps, run_k_selection
from ._sample_specific_fit import (
    build_padded_sample_specific_outputs,
    fit_sample_specific_models,
)


def _resolve_component_sharing(dataset, component_sharing: str) -> Literal["global", "sample_specific"]:
    """Resolve explicit or automatic component sharing mode."""
    if component_sharing in {"global", "sample_specific"}:
        return component_sharing
    if component_sharing != "auto":
        raise ValueError("component_sharing must be 'global', 'sample_specific', or 'auto'.")
    return "global" if bool(dataset.identical_supports) else "sample_specific"


def _select_fit_indices(
    *,
    has_selection: bool,
    refit: Literal["full", "none"],
    n_cells: int,
    selection_idx: np.ndarray | None,
) -> np.ndarray:
    """Select data indices used for the final fit stage."""
    if has_selection and refit == "none":
        if selection_idx is None:
            raise RuntimeError("Expected selection indices when refit='none'.")
        return selection_idx
    return np.arange(n_cells, dtype=int)


def _check_min_cells_per_distribution(
    *,
    distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
    component_sharing: Literal["global", "sample_specific"],
    min_required: int,
) -> None:
    """Validate minimum cell count constraints for requested K."""
    if component_sharing != "sample_specific":
        return
    dist_counts = np.bincount(distribution_index, minlength=len(distribution_ids))
    too_small = [str(distribution_ids[i]) for i, c in enumerate(dist_counts) if int(c) < min_required]
    if too_small:
        raise ValueError(
            "sample_specific fitting requires each distribution to contain at least "
            f"{min_required} cells. Too small: {too_small}"
        )


def _build_fit_config(
    *,
    fixed_k: int | None,
    covariance_type: Literal["diag", "full"],
    max_iter: int,
    tol: float,
    n_init: int,
    eps: float,
    singularity_handling: Literal["guarded", "robust", "strict"],
) -> GMMFitConfig:
    """Build backend fit config from public parameters."""
    return GMMFitConfig(
        n_components=1 if fixed_k is None else int(fixed_k),
        covariance_type=covariance_type,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        eps=eps,
        singularity_handling=singularity_handling,
    )


def _run_global_fit(
    *,
    x: np.ndarray,
    x_fit: np.ndarray,
    distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
    config: GMMFitConfig,
    has_selection: bool,
    refit: Literal["full", "none"],
    selected_global_result,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run global-sharing fit branch and return canonical outputs."""
    from ._GaussianMixture import GaussianMixture

    if has_selection and refit == "none":
        if selected_global_result is None:
            raise RuntimeError("Expected selected global result when refit='none'.")
        result = selected_global_result
    else:
        result = GaussianMixture.fit_from_numpy(x=x_fit, config=config)

    responsibilities = result.model.predict_responsibilities_numpy(x)
    hard_components = np.argmax(responsibilities, axis=1)
    distribution_weights = aggregate_distribution_weights(
        responsibilities=responsibilities,
        distribution_index=distribution_index,
        n_distributions=len(distribution_ids),
    )
    distribution_weights = normalize_weight_rows(distribution_weights)
    pi_global = normalize_weight_vector(np.asarray(result.pi, dtype=np.float64).reshape(-1))
    model_payload = {
        "mu": np.asarray(result.mu, dtype=np.float64),
        "var": np.asarray(result.var, dtype=np.float64),
        "pi": pi_global.reshape(1, -1),
    }
    return hard_components, responsibilities, distribution_weights, model_payload


def _fit_gmm(
    *,
    x: np.ndarray,
    distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
    component_sharing: Literal["global", "sample_specific"],
    k_comps: int | list[int] | tuple[int, ...] | None,
    k_selection_metric: Literal["bic", "aic", "heldout_nll"],
    refit: Literal["full", "none"],
    covariance_type: Literal["diag", "full"],
    train_size: float,
    max_iter: int,
    tol: float,
    n_init: int,
    eps: float,
    singularity_handling: Literal["guarded", "robust", "strict"],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run complete GMM orchestration on matrix inputs."""
    x = np.asarray(x, dtype=np.float64)
    distribution_index = np.asarray(distribution_index, dtype=int)
    distribution_ids = np.asarray(distribution_ids)

    validate_fit_inputs(
        x=x,
        distribution_index=distribution_index,
        distribution_ids=distribution_ids,
        component_sharing=component_sharing,
        k_comps=k_comps,
        refit=refit,
    )

    fixed_k, k_candidates = parse_k_comps(k_comps)
    has_selection = k_candidates is not None

    if not has_selection:
        refit = "full"

    min_required = int(max(k_candidates)) if has_selection else int(fixed_k)
    if x.shape[0] < min_required:
        raise ValueError(f"Not enough cells for requested components: n_cells={x.shape[0]} < required={min_required}.")

    _check_min_cells_per_distribution(
        distribution_index=distribution_index,
        distribution_ids=distribution_ids,
        component_sharing=component_sharing,
        min_required=min_required,
    )
    # FUTURE: tighten cell-count validation — full covariance estimation requires
    # n_cells_per_component >= n_features + 1. Currently we only check total cells >= n_components.
    # A post-fit check using effective sample count (pi_k * N) could provide an earlier,
    # more informative warning before singularity manifests in eigenvalues.
    config = _build_fit_config(
        fixed_k=fixed_k,
        covariance_type=covariance_type,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        eps=eps,
        singularity_handling=singularity_handling,
    )

    seed = settings.random_seed
    if seed is not None:
        seed_int = int(seed)
        # Keep torch-side EM initialization reproducible for fixed ggml_ot.settings.random_seed.
        torch.manual_seed(seed_int)
        rng = np.random.default_rng(seed_int)
    else:
        rng = np.random.default_rng()

    (
        config,
        selection_meta,
        selection_idx,
        selected_global_result,
        selected_sample_results,
        selected_k_by_distribution,
    ) = run_k_selection(
        x=x,
        distribution_index=distribution_index,
        distribution_ids=distribution_ids,
        component_sharing=component_sharing,
        config=config,
        k_candidates=k_candidates,
        k_selection_metric=k_selection_metric,
        refit=refit,
        train_size=train_size,
        rng=rng,
        verbose=verbose,
    )
    fit_indices = _select_fit_indices(
        has_selection=has_selection,
        refit=refit,
        n_cells=x.shape[0],
        selection_idx=selection_idx,
    )

    if fit_indices.shape[0] < config.n_components:
        raise ValueError(
            "Fit subset is too small for requested components: "
            f"selected={fit_indices.shape[0]} < n_components={config.n_components}. "
            "Increase train_size or reduce k_comps."
        )

    x_fit = x[fit_indices]

    if component_sharing == "global":
        hard_components, responsibilities, distribution_weights, model_payload = _run_global_fit(
            x=x,
            x_fit=x_fit,
            distribution_index=distribution_index,
            distribution_ids=distribution_ids,
            config=config,
            has_selection=has_selection,
            refit=refit,
            selected_global_result=selected_global_result,
        )
        distribution_n_components = np.full(len(distribution_ids), int(config.n_components), dtype=int)
    else:
        fit_distribution_index = distribution_index[fit_indices]
        results = fit_sample_specific_models(
            x_fit=x_fit,
            fit_distribution_index=fit_distribution_index,
            distribution_ids=distribution_ids,
            config=config,
            has_selection=has_selection,
            refit=refit,
            selected_sample_results=selected_sample_results,
            selected_k_by_distribution=selected_k_by_distribution,
            verbose=verbose,
        )
        hard_components, responsibilities, distribution_weights, model_payload, distribution_n_components = (
            build_padded_sample_specific_outputs(
                results=results,
                distribution_ids=distribution_ids,
                x=x,
                full_distribution_index=distribution_index,
                covariance_type=config.covariance_type,
            )
        )

    if verbose:
        selection_note = ""
        if selection_meta is not None:
            metric = str(selection_meta["metric"])
            best_k = int(selection_meta["best_k"])
            selection_note = f", k_selection={metric}, best_k={best_k}"
        print(
            "[fit_gmm] "
            f"device={settings.device}, sharing={component_sharing}, "
            f"n_cells={x.shape[0]}, n_features={x.shape[1]}, "
            f"fit_cells={fit_indices.shape[0]}, n_components={config.n_components}"
            f"{selection_note}"
        )

    if component_sharing == "sample_specific":
        distribution_weights = sanitize_degenerate_sample_specific_gmm_components(
            supports=np.asarray(model_payload["mu"], dtype=np.float64),
            distribution_weights=np.asarray(distribution_weights, dtype=np.float64),
            distribution_ids=distribution_ids,
            eps=float(config.eps),
            stage="fit_gmm",
        )
        model_payload["pi"] = np.asarray(distribution_weights, dtype=np.float64)

    return {
        "config": config,
        "refit": refit,
        "selection_meta": selection_meta,
        "fit_indices": fit_indices,
        "hard_components": hard_components,
        "responsibilities": responsibilities,
        "distribution_weights": distribution_weights,
        "model_payload": model_payload,
        "distribution_n_components": distribution_n_components,
    }


def fit_gmm(
    dataset,
    *,
    component_sharing: Literal["global", "sample_specific", "auto"] = "sample_specific",
    k_comps: int | list[int] | tuple[int, ...] | None = None,
    k_selection_metric: Literal["bic", "aic", "heldout_nll"] = "aic",
    refit: Literal["full", "none"] = "none",
    covariance_type: Literal["diag", "full"] = "full",
    train_size: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-3,
    n_init: int = 1,
    eps: float = 1e-4,
    singularity_handling: Literal["guarded", "robust", "strict"] = "guarded",
    gmm_key: str | None = None,
    verbose: bool = True,
):
    """Fit per-patient GMM parameters from a GGML dataset.

    Fits Gaussian mixture models to the per-patient cell distributions in
    ``dataset`` and attaches the result as Gaussian component
    (means/covariances/weights) tensors. The updated dataset can then be
    used directly with :func:`ggml_ot.train_gmm`.

    Parameters
    ----------
    dataset
        A :class:`~ggml_ot.data.TripletDataset` or
        :class:`~ggml_ot.data.AnnData_TripletDataset` whose distributions
        will be modelled as GMMs.
    component_sharing
        How components are shared across patients.
        ``"sample_specific"`` (default) fits an independent model per patient.
        ``"global"`` fits one shared model across all patients.
        ``"auto"`` picks ``"sample_specific"`` for AnnData datasets and
        ``"global"`` for generic datasets.
    k_comps
        Number of GMM components. An int fixes the count; a list/tuple
        triggers automatic k-selection over the provided candidates using
        ``k_selection_metric``. ``None`` uses a default candidate range.
    k_selection_metric
        Criterion used to select the best ``k`` when ``k_comps`` is a
        sequence: ``"aic"`` (default), ``"bic"``, or
        ``"heldout_nll"``. The held-out NLL path uses an internal
        train/validation split of the selected cells and stores the resulting
        diagnostics in the persisted GMM schema.
    refit
        Whether to refit the best selected model on all data after
        k-selection. Only used when ``k_comps`` is a sequence.
        ``"none"`` (default) reuses the model from the selection pass;
        ``"full"`` refits the best ``k`` on the full dataset.
    covariance_type
        Covariance structure of each Gaussian component:
        ``"full"`` (default) or ``"diag"``.
    train_size
        Training fraction used during k-selection. The remaining
        ``1 - train_size`` cells form the validation split for
        ``k_selection_metric="heldout_nll"`` and are ignored by
        in-sample criteria such as ``"aic"`` and ``"bic"``. When
        ``refit="none"``, the selected model is retained from that
        training split; when ``refit="full"``, the best selected ``k``
        is refit on all cells. The default ``0.5`` yields a balanced
        train/validation split during selection.
    max_iter
        Maximum EM iterations per fit.
    tol
        EM convergence tolerance.
    n_init
        Number of random EM restarts. Higher values reduce sensitivity to
        initialization at the cost of runtime.
    eps
        Numerical floor added to diagonal of covariance matrices to
        prevent singularities.
    singularity_handling
        How near-singular projected covariances are treated:
        ``"guarded"`` (default) applies a small jitter and continues,
        ``"robust"`` uses a larger stabilization,
        ``"strict"`` raises an error on any detected singularity.
    gmm_key
        Key under which the fitted GMM is stored in
        ``dataset.adata.uns`` (AnnData datasets only). Defaults to
        ``"gmm_<use_rep>"`` when ``None``.
    verbose
        Print per-patient fit progress.

    Returns
    -------
    TripletDataset | AnnData_TripletDataset
        The input dataset augmented with fitted GMM supports, covariances,
        and weights.
    """
    from ggml_ot.data.anndata import AnnData_TripletDataset

    resolved_sharing = _resolve_component_sharing(dataset=dataset, component_sharing=component_sharing)
    fit_kwargs = {
        "k_comps": k_comps,
        "k_selection_metric": k_selection_metric,
        "refit": refit,
        "covariance_type": covariance_type,
        "train_size": train_size,
        "max_iter": max_iter,
        "tol": tol,
        "n_init": n_init,
        "eps": eps,
        "singularity_handling": singularity_handling,
        "verbose": verbose,
    }

    if isinstance(dataset, AnnData_TripletDataset):
        resolved_gmm_key = gmm_key or f"gmm_{'X' if dataset.use_rep is None else dataset.use_rep}"
        x, distribution_index, distribution_ids = _extract_fit_data_from_anndata(
            dataset.adata,
            use_rep=dataset.use_rep,
            distribution_col=dataset.patient_col,
        )
        fit_outputs = _fit_gmm(
            x=x,
            distribution_index=distribution_index,
            distribution_ids=distribution_ids,
            component_sharing=resolved_sharing,
            **fit_kwargs,
        )
        config = fit_outputs["config"]
        effective_refit = str(fit_outputs["refit"])
        selection_meta = fit_outputs["selection_meta"]
        fit_indices = fit_outputs["fit_indices"]
        _store_gmm_in_adata(
            adata=dataset.adata,
            gmm_key=resolved_gmm_key,
            component_sharing=resolved_sharing,
            use_rep=dataset.use_rep,
            covariance_type=config.covariance_type,
            n_components=int(config.n_components),
            distribution_n_components=np.asarray(
                fit_outputs.get(
                    "distribution_n_components",
                    np.full(len(distribution_ids), int(config.n_components), dtype=int),
                ),
                dtype=int,
            ),
            fit_params={
                "max_iter": int(config.max_iter),
                "tol": float(config.tol),
                "n_init": int(config.n_init),
                "eps": float(config.eps),
                "singularity_handling": str(config.singularity_handling),
            },
            selection=selection_meta,
            backend="native",
            backend_metadata={
                "fit_indices_count": int(len(fit_indices)),
                "refit": effective_refit,
            },
            model_payload=fit_outputs["model_payload"],
            distribution_weights=np.asarray(fit_outputs["distribution_weights"], dtype=np.float64),
            distribution_ids=distribution_ids,
            hard_components=np.asarray(fit_outputs["hard_components"]),
            responsibilities=np.asarray(fit_outputs["responsibilities"], dtype=np.float64),
        )

        # Materialize persisted fields back onto the dataset object to keep both views consistent.
        supports, covariances, weights, identical_supports = _extract_fitted_anndata_fields(
            dataset, gmm_key=resolved_gmm_key
        )
        dataset.index_mask = np.ones(dataset.adata.n_obs, dtype=bool)
        return _apply_gmm_fields_in_place(
            dataset,
            supports=supports,
            covariances=covariances,
            weights=weights,
            identical_supports=identical_supports,
            gmm_provenance="fit_gmm",
        )

    x, distribution_index, distribution_ids, labels = _extract_fit_data_from_dataset(
        dataset,
        component_sharing=resolved_sharing,
    )
    fit_outputs = _fit_gmm(
        x=x,
        distribution_index=distribution_index,
        distribution_ids=distribution_ids,
        component_sharing=resolved_sharing,
        **fit_kwargs,
    )
    return _store_gmm_in_dataset(
        dataset,
        fit_outputs=fit_outputs,
        covariance_type=covariance_type,
        component_sharing=resolved_sharing,
        distribution_labels=labels,
    )


__all__ = ["fit_gmm"]
