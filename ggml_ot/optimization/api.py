"""Public training API for GGML.

This module exposes:
- :func:`train` as the canonical generic dispatcher used by both
  ``ggml_ot.train(dataset, ...)`` and ``dataset.train(...)``.
- Explicit solver entry points (:func:`train_emd2`, :func:`train_sinkhorn`,
  :func:`train_gmm`) for users who want direct control over the OT route.
- :func:`train_history` as a debug-oriented wrapper that returns the
  initialized map plus one snapshot per completed epoch.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal
from dataclasses import asdict

from ggml_ot._utils._covariance import has_dataset_covariances, require_dataset_covariances
import numpy as np

from ggml_ot import settings
from ggml_ot.optimization._ggml import _ggml_generic
from ggml_ot._utils._batch import _setup_dataloader
from ggml_ot.optimization.loss import batch_triplet_loss_emd2, batch_triplet_loss_sinkhorn
from ggml_ot.optimization.params import (
    DataLoaderParams,
    GGMLParams,
    MIRegularizationParams,
    OTParams,
    OptimizationParams,
    RegularizationParams,
    SinkhornParams,
    WassersteinBuresParams,
)
from ggml_ot.optimization.regularization import regularizer_loss, regularizer_loss_mutual_information

if TYPE_CHECKING:
    from ggml_ot.data import AnnData_TripletDataset, TripletDataset


def _warn_emd2_cpu_bound_on_cuda() -> None:
    """Warn that exact EMD2 OT solves are CPU-bound even when using CUDA."""
    if settings.device.type == "cuda":
        warnings.warn(
            "Exact EMD2 solver is CPU-bound; no GPU acceleration is used for OT solves.",
            UserWarning,
        )


def train_emd2(
    dataset: TripletDataset | AnnData_TripletDataset,
    alpha: float = 10.0,
    reg: float = 0.001,
    reg_type: Literal[1, "1", 2, "2", "fro", "nuc"] = "fro",
    n_comps: int = 5,
    lr: float = 0.05,
    max_iter: int = 30,
    stop_thr: float | None = None,
    verbose: bool = False,
    plot_iter: int | bool = -1,
    batch_size: int = 512,
    train_size: float | None = None,
    squared_ground_cost: bool = False,
    eps: float = 1e-4,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Train GGML with exact OT (EMD2).

    Parameters
    ----------
    dataset
        Training dataset with empirical supports or Gaussian component means.
    alpha, reg, reg_type, n_comps
        Triplet margin and regularization controls.
    lr, max_iter, stop_thr
        Outer optimization controls (`stop_thr` is the gradient-norm threshold).
    verbose, plot_iter
        Logging and plotting cadence. Plot behavior:
        `0`/`False` disables plotting, `-1` plots only after the final epoch,
        `1`/`True` plots every epoch, `k>=1` plots every `k` epochs.
    batch_size, train_size
        DataLoader controls.
    squared_ground_cost
        If True on Gaussian datasets, use squared Euclidean distance for the
        mean term instead of Euclidean distance.
    eps
        Numerical floor for covariance regularization and OT stability.

    Returns
    -------
    tuple[np.ndarray, float]
        Learned ground metric (``map_A``) and mean epoch time.
    """
    dataloader = _setup_dataloader(
        DataLoaderParams(dataset=dataset, batch_size=batch_size, train_size=train_size),
        device=settings.device,
    )

    _warn_emd2_cpu_bound_on_cuda()

    ggml_params = GGMLParams(alpha=alpha, n_comps=n_comps)
    reg_params = RegularizationParams(reg=reg, reg_type=reg_type)
    optim_params = OptimizationParams(lr=lr, max_iter=max_iter, stop_thr=stop_thr)
    ot_params = OTParams(
        identical_supports=dataloader.dataset.identical_supports,
        eps=eps,
        squared_ground_cost=squared_ground_cost,
    )
    if has_dataset_covariances(dataset):
        ot_params = WassersteinBuresParams(diag_bures_approx=False, **asdict(ot_params))

    return _ggml_generic(
        dataloader=dataloader,
        ggml_fn=batch_triplet_loss_emd2,
        ggml_params=ggml_params,
        ot_params=ot_params,
        reg_fn=regularizer_loss,
        reg_params=reg_params,
        optim_params=optim_params,
        verbose=verbose,
        plot_iter=plot_iter,
        **kwargs,
    )


def train_sinkhorn(
    dataset: TripletDataset | AnnData_TripletDataset,
    alpha: float = 10.0,
    reg: float = 0.001,
    reg_type: Literal[1, "1", 2, "2", "fro", "nuc"] = "fro",
    n_comps: int = 5,
    entropic_reg: float = 0.5,
    sinkhorn_max_iter: int = 100,
    sinkhorn_stop_thr: float | None = None,
    lr: float = 0.05,
    max_iter: int = 30,
    stop_thr: float | None = None,
    verbose: bool = False,
    plot_iter: int | bool = -1,
    batch_size: int = 512,
    train_size: float | None = None,
    squared_ground_cost: bool = False,
    eps: float = 1e-4,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Train GGML with Sinkhorn-regularized OT.

    Parameters
    ----------
    dataset
        Training dataset with empirical supports or Gaussian component means/covariances.
    alpha, reg, reg_type, n_comps
        Triplet margin and regularization controls.
    entropic_reg
        Entropic OT regularization strength for Sinkhorn (`> 0` required),
        expressed as a fraction of the mean ground-cost value (e.g. ``0.5``
        regularizes at half the mean pairwise cost). The cost matrix is
        normalized internally, so this value is scale-invariant across datasets
        and training stages.
    sinkhorn_max_iter, sinkhorn_stop_thr
        Sinkhorn inner-solver controls.
    lr, max_iter, stop_thr
        Outer optimization controls.
    verbose, plot_iter
        Logging and plotting cadence.
    batch_size, train_size
        DataLoader controls.
    squared_ground_cost
        If True on Gaussian datasets, use squared Euclidean distance for the
        mean term instead of Euclidean distance.
    eps
        Numerical floor for covariance regularization and OT stability.

    Returns
    -------
    tuple[np.ndarray, float]
        Learned ground metric (``map_A``) and mean epoch time.

    Raises
    ------
    ValueError
        If `entropic_reg <= 0`.
    """
    if entropic_reg <= 0:
        raise ValueError("`train_sinkhorn` requires `entropic_reg > 0`.")

    dataloader = _setup_dataloader(
        DataLoaderParams(dataset=dataset, batch_size=batch_size, train_size=train_size),
        device=settings.device,
    )

    ggml_params = GGMLParams(alpha=alpha, n_comps=n_comps)
    reg_params = RegularizationParams(reg=reg, reg_type=reg_type)
    optim_params = OptimizationParams(lr=lr, max_iter=max_iter, stop_thr=stop_thr)
    sinkhorn_params = SinkhornParams(
        entropic_reg=entropic_reg,
        sinkhorn_max_iter=sinkhorn_max_iter,
        sinkhorn_stop_thr=sinkhorn_stop_thr,
        eps=eps,
        squared_ground_cost=squared_ground_cost,
    )
    if has_dataset_covariances(dataset):
        sinkhorn_params = WassersteinBuresParams(diag_bures_approx=False, **asdict(sinkhorn_params))

    return _ggml_generic(
        dataloader=dataloader,
        ggml_fn=batch_triplet_loss_sinkhorn,
        ggml_params=ggml_params,
        ot_params=sinkhorn_params,
        reg_fn=regularizer_loss,
        reg_params=reg_params,
        optim_params=optim_params,
        verbose=verbose,
        plot_iter=plot_iter,
        **kwargs,
    )


def train_gmm(
    dataset: TripletDataset | AnnData_TripletDataset,
    alpha: float = 10.0,
    reg: float = 0.001,
    reg_type: Literal[1, "1", 2, "2", "fro", "nuc"] = "fro",
    n_comps: int = 5,
    lr: float = 0.1,
    max_iter: int = 30,
    verbose: bool = False,
    plot_iter: int | bool = -1,
    mi_reg: float | bool = 0.0,
    mi_sqrt: bool = False,
    diag_bures_approx: bool = False,
    mi_reg_weighting: str = "projection",
    stop_thr: float | None = 1e-6,
    entropic_reg: float = 0.0,
    sinkhorn_max_iter: int = 100,
    sinkhorn_stop_thr: float | None = None,
    batch_size: int = 512,
    train_size: float | None = None,
    squared_ground_cost: bool = False,
    eps: float = 1e-4,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Train GGML with exact OT on GMM datasets.

    This convenience wrapper enables GMM-specific regularization knobs while
    staying on the exact OT (EMD2) path.

    Parameters
    ----------
    dataset
        Training dataset with Gaussian covariances.
    alpha, reg, reg_type, n_comps
        Triplet margin and regularization controls.
    lr, max_iter, stop_thr
        Outer optimization controls.
    verbose, plot_iter
        Logging and plotting cadence.
    mi_reg
        Mutual-information regularization strength. If provided as `bool`, it is
        interpreted as a multiplicative on/off flag in the current implementation.
    mi_sqrt
        If True, minimize ``sum w_k sqrt(MI_k)`` instead of ``sum w_k MI_k``.
        This matches the functional form of the diagonal approximation error
        bound (Theorem diag_mi_bounds).
    diag_bures_approx
        If True, use diagonal approximation for the Bures component.
    squared_ground_cost
        If True, use squared Euclidean distance for the Gaussian mean term.
    mi_reg_weighting
        Weighting mode for MI regularization over GMM components.
    entropic_reg
        If ``> 0``, use the Sinkhorn backend instead of exact EMD2.
    sinkhorn_max_iter, sinkhorn_stop_thr
        Sinkhorn inner-solver controls used when ``entropic_reg > 0``.
    batch_size, train_size
        DataLoader controls.
    eps
        Numerical floor for covariance regularization and OT stability.

    Returns
    -------
    tuple[np.ndarray, float]
        Learned ground metric (``map_A``) and mean epoch time.

    Raises
    ------
    ValueError
        If the dataset has no covariance tensors.
    """
    require_dataset_covariances(dataset, caller="train_gmm")

    dataloader = _setup_dataloader(
        DataLoaderParams(dataset=dataset, batch_size=batch_size, train_size=train_size),
        device=settings.device,
    )
    ggml_params = GGMLParams(alpha=alpha, n_comps=n_comps)
    optim_params = OptimizationParams(lr=lr, max_iter=max_iter, stop_thr=stop_thr)

    # OT Backend
    ot_params = OTParams(
        identical_supports=dataloader.dataset.identical_supports,
        eps=eps,
        squared_ground_cost=squared_ground_cost,
    )
    if entropic_reg > 0:
        # Entropic regularization, dispatch to Sinkhorn training route.
        ot_params = SinkhornParams(
            entropic_reg=entropic_reg,
            sinkhorn_max_iter=sinkhorn_max_iter,
            sinkhorn_stop_thr=sinkhorn_stop_thr,
            **asdict(ot_params),
        )
        ggml_backend = batch_triplet_loss_sinkhorn
    else:
        # Exact EMD2-based training route
        _warn_emd2_cpu_bound_on_cuda()
        ggml_backend = batch_triplet_loss_emd2

    # Bures-specific params
    wb_params = WassersteinBuresParams(
        diag_bures_approx=diag_bures_approx,
        **asdict(ot_params),
    )

    # Mutual Information Regularization params
    reg_params = MIRegularizationParams(
        mi_reg=mi_reg,
        mi_reg_weighting=mi_reg_weighting,
        reg=reg,
        reg_type=reg_type,
        eps=eps,
        mi_sqrt=mi_sqrt,
    )

    # Dispatch generic gmml train loop with specified ggml backend and GMM-specific regularizer
    return _ggml_generic(
        dataloader=dataloader,
        ggml_fn=ggml_backend,
        ggml_params=ggml_params,
        ot_params=wb_params,
        reg_fn=regularizer_loss_mutual_information,
        reg_params=reg_params,
        optim_params=optim_params,
        verbose=verbose,
        plot_iter=plot_iter,
        **kwargs,
    )


def train(
    dataset: TripletDataset | AnnData_TripletDataset,
    alpha: float = 1.0,
    reg: float = 0.001,
    reg_type: Literal[1, "1", 2, "2", "fro", "nuc"] = "fro",
    n_comps: int = 5,
    lr: float = 0.1,
    max_iter: int = 30,
    plot_iter: int | bool = -1,
    verbose: bool = False,
    batch_size: int = 512,
    train_size: float | None = None,
    squared_ground_cost: bool = False,
    return_dataset: bool = True,
    measure_time: bool = False,
    mi_reg: float = 0.0,
    diag_bures_approx: bool = False,
    mi_reg_weighting: str | None = None,
    entropic_reg: float = 0.0,
    sinkhorn_max_iter: int = 100,
    sinkhorn_stop_thr: float | None = 1e-6,
    stop_thr: float | None = 1e-6,
    eps: float = 1e-4,
    **kwargs,
) -> TripletDataset | AnnData_TripletDataset | np.ndarray | tuple[np.ndarray, float]:
    """Perform supervised optimal transport by ground metric learning.

    GGML learns a linear ground metric so distributions from the same class are
    closer than distributions from different classes.

    Parameters
    ----------
    dataset
        Dataset containing distributions and labels.
    alpha
        Triplet margin.
    reg
        Regularization strength.
    reg_type
        Type of regularization (`1`/`"1"` for L1, `2`/`"2"`/`"fro"` for
        L2/Frobenius, `"nuc"` for nuclear norm).
    n_comps
        Number of learned components (rank of the linear map).
    lr
        Adam learning rate.
    max_iter
        Number of training epochs.
    plot_iter
        Training progress plot frequency: ``0``/``False`` disables plotting,
        ``-1`` plots only after the final epoch, ``1``/``True`` plots every
        epoch, ``k >= 1`` plots every *k* epochs.  Whether plots are
        displayed or saved follows the ``show``/``save`` conventions
        described in the plotting API (see :func:`ggml_ot.pl.embedding`).
    verbose
        Print optimization progress.
    batch_size
        DataLoader batch size.
    train_size
        Optional train split ratio used to create a training subset.
    return_dataset
        If True, assign learned metric to ``dataset.map_A`` and return dataset.
    measure_time
        If True and ``return_dataset=False``, also return mean epoch time.
    mi_reg
        Mutual-information regularization strength for GMM training.
        In current behavior, `bool` is allowed and treated multiplicatively.
    diag_bures_approx
        If True, use diagonal approximation of the Bures term for GMMs.
    squared_ground_cost
        If True on Gaussian datasets, use squared Euclidean distance for the
        mean term instead of Euclidean distance.
    mi_reg_weighting
        Weighting scheme for MI regularization across Gaussian components.
    entropic_reg
        Entropic OT regularization. If ``> 0``, dispatch to Sinkhorn training.
    sinkhorn_max_iter
        Maximum iterations of the Sinkhorn inner solver.
    sinkhorn_stop_thr
        Stopping threshold for the Sinkhorn inner solver.
    stop_thr
        Stopping threshold on objective gradient norm in outer optimization.
    eps
        Numerical floor for covariance regularization and OT stability.
    Returns
    -------
    TripletDataset | AnnData_TripletDataset
        Returned when ``return_dataset=True``.
    np.ndarray
        Learned ground metric when ``return_dataset=False``.
    tuple[np.ndarray, float]
        Learned ground metric and mean epoch time when
        ``return_dataset=False`` and ``measure_time=True``.

    Raises
    ------
    ValueError
        If GMM-specific knobs are used without covariances.
    """

    # Data & Model setup
    dataloader = _setup_dataloader(
        DataLoaderParams(dataset=dataset, batch_size=batch_size, train_size=train_size),
        device=settings.device,
    )
    ggml_params = GGMLParams(alpha=alpha, n_comps=n_comps)
    optim_params = OptimizationParams(lr=lr, max_iter=max_iter, stop_thr=stop_thr)

    # OT Backend: Sinkhorn or EMD2
    ot_params = OTParams(
        identical_supports=dataloader.dataset.identical_supports,
        eps=eps,
        squared_ground_cost=squared_ground_cost,
    )
    if entropic_reg > 0:
        # Entropic regularization: Sinkhorn training route
        ot_params = SinkhornParams(
            entropic_reg=entropic_reg,
            sinkhorn_max_iter=sinkhorn_max_iter,
            sinkhorn_stop_thr=sinkhorn_stop_thr,
            **asdict(ot_params),
        )
        ggml_backend = batch_triplet_loss_sinkhorn
    else:
        # Exact EMD2-based training route
        _warn_emd2_cpu_bound_on_cuda()
        ggml_backend = batch_triplet_loss_emd2

    if diag_bures_approx:
        require_dataset_covariances(dataset, caller="train with GMM option `diag_bures_approx`")
    if has_dataset_covariances(dataset):
        ot_params = WassersteinBuresParams(diag_bures_approx=diag_bures_approx, **asdict(ot_params))

    # Regularization
    reg_params = RegularizationParams(reg=reg, reg_type=reg_type)
    regularizer_backend = regularizer_loss

    # GMM-specific Mutual Information Regularizer
    if mi_reg > 0 or mi_reg_weighting is not None:
        require_dataset_covariances(dataset, caller="train with GMM options `mi_reg`, `mi_reg_weighting`")
        reg_params = MIRegularizationParams(
            mi_reg=mi_reg,
            # If mi_reg is active but no weighting strategy was specified, default to "projection"
            mi_reg_weighting=mi_reg_weighting if mi_reg_weighting is not None else "projection",
            eps=ot_params.eps,
            **asdict(reg_params),
        )
        regularizer_backend = regularizer_loss_mutual_information

    # Dispatch to generic training loop with selected backends and parameters
    map_A, times = _ggml_generic(
        dataloader=dataloader,
        ggml_fn=ggml_backend,
        ggml_params=ggml_params,
        ot_params=ot_params,
        reg_fn=regularizer_backend,
        reg_params=reg_params,
        optim_params=optim_params,
        verbose=verbose,
        plot_iter=plot_iter,
        **kwargs,
    )

    # FUTURE: clean up return logic, prob. remove dataset return and just do in place
    if return_dataset:
        if bool(kwargs.get("return_map_A_history", False)):
            raise ValueError(
                "`return_dataset` with `return_map_A_history` not supported, use `train_history(...)` instead."
            )
        dataset.map_A = map_A
        return dataset

    if measure_time:
        return map_A, times
    return map_A


def train_history(
    dataset: TripletDataset | AnnData_TripletDataset,
    alpha: float = 10.0,
    reg: float = 0.001,
    reg_type: Literal[1, "1", 2, "2", "fro", "nuc"] = "fro",
    n_comps: int = 5,
    lr: float = 0.1,
    max_iter: int = 30,
    plot_iter: int | bool = -1,
    verbose: bool = False,
    batch_size: int = 512,
    train_size: float | None = None,
    squared_ground_cost: bool = False,
    mi_reg: float = 0.0,
    diag_bures_approx: bool = False,
    mi_reg_weighting: str | None = None,
    entropic_reg: float = 0.0,
    sinkhorn_max_iter: int = 100,
    sinkhorn_stop_thr: float | None = None,
    stop_thr: float | None = 1e-6,
    eps: float = 1e-4,
    **kwargs,
) -> list[np.ndarray]:
    """Perform supervised OT training and return init + per-epoch map snapshots.

    This is a debug/analysis helper that mirrors :func:`train`, but instead of
    assigning the learned map to the dataset it returns a list containing the
    initialized ``map_A`` followed by one snapshot for each completed epoch.
    """
    kwargs.pop("return_dataset", None)
    kwargs.pop("measure_time", None)
    kwargs["return_map_A_history"] = True

    return train(
        dataset=dataset,
        alpha=alpha,
        reg=reg,
        reg_type=reg_type,
        n_comps=n_comps,
        lr=lr,
        max_iter=max_iter,
        plot_iter=plot_iter,
        verbose=verbose,
        batch_size=batch_size,
        train_size=train_size,
        squared_ground_cost=squared_ground_cost,
        return_dataset=False,
        measure_time=False,
        mi_reg=mi_reg,
        diag_bures_approx=diag_bures_approx,
        mi_reg_weighting=mi_reg_weighting,
        entropic_reg=entropic_reg,
        sinkhorn_max_iter=sinkhorn_max_iter,
        sinkhorn_stop_thr=sinkhorn_stop_thr,
        stop_thr=stop_thr,
        eps=eps,
        **kwargs,
    )
