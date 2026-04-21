"""Internal optimization core utilities and training loop."""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

from ggml_ot import settings
from ggml_ot._utils._array import snapshot_numpy
from ggml_ot._utils._batch import (
    move_batch_to_device as _move_batch_to_device,
)
from ggml_ot._utils._validate import assert_finite_tensor
from ggml_ot.optimization.params import (
    GGMLParams,
    MIRegularizationParams,
    OTParams,
    OptimizationParams,
    RegularizationParams,
)
from ggml_ot.optimization.triplets import _create_triplets
from ggml_ot.optimization._triplets import warn_empty_triplets


def _ggml_init(
    dataloader: torch.utils.data.DataLoader,
    n_comps: int | float | None,
):
    """Initialize the ground metric map for GGML training."""
    dim = dataloader.dataset.dim

    if n_comps is None:
        n_comps = dim
    n_comps = int(n_comps)

    if n_comps == 1:
        warnings.warn("n_comps is 1, will be treated as diagonal matrix that scales every dimension")

    map_A = torch.empty((n_comps, dim), device=settings.device)
    strategy = settings.init_strategy
    if strategy == "orthonormal":
        if n_comps <= dim:
            torch.nn.init.orthogonal_(map_A)
        else:
            torch.nn.init.xavier_uniform_(map_A)
    elif strategy == "orthogonal":
        if n_comps <= dim:
            torch.nn.init.orthogonal_(map_A)
            # Scale rows away from unit norm while preserving orthogonality
            map_A.data *= dim**0.5
        else:
            torch.nn.init.xavier_uniform_(map_A)
    elif strategy == "random":
        torch.nn.init.xavier_uniform_(map_A)
    else:
        raise ValueError(f"Unknown init_strategy: {strategy!r}. Use 'orthonormal', 'orthogonal', or 'random'.")

    map_A.requires_grad_(True)
    return map_A


def _ggml_generic(
    dataloader: torch.utils.data.DataLoader,
    ggml_fn: callable,
    ggml_params: GGMLParams,
    ot_params: OTParams | None = None,
    reg_fn: callable = None,
    reg_params: RegularizationParams = None,
    optim_params: OptimizationParams = None,
    verbose: bool = False,
    plot_iter: int | bool = -1,
    return_map_A_history: bool = False,
    **kwargs,
):
    """Train GGML with a generic objective + regularizer loop."""
    map_A = _ggml_init(dataloader, ggml_params.n_comps)
    assert_finite_tensor(map_A.detach(), name="map_A (init)", epoch=0)
    alpha = torch.scalar_tensor(float(ggml_params.alpha), device=settings.device)

    if reg_params is not None and reg_params.reg > 0:
        reg_params.reg = torch.scalar_tensor(float(reg_params.reg), device=settings.device)

    epoch_times = []
    dataset = dataloader.dataset

    if ot_params is not None and ot_params.identical_supports:
        points = dataset.supports.to(settings.device)
        if len(dataset.covariances) > 0:
            covs = dataset.covariances.to(settings.device)
        else:
            covs = torch.tensor([], device=settings.device)

    optimizer = torch.optim.Adam([map_A], lr=optim_params.lr)
    converged = False
    debug_anomaly = os.getenv("GGML_DEBUG_ANOMALY", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    map_A_history = [snapshot_numpy(map_A)] if return_map_A_history else None

    for i in range(1, optim_params.max_iter + 1):
        start_epoch = time.time()

        pbar = tqdm(dataloader, disable=1 - verbose)
        for batch_idx, (distr_points, distr_covs, distr_weights, distr_labels) in enumerate(pbar, start=1):
            triplets_idx = _create_triplets(distr_labels)
            if warn_empty_triplets(triplets_idx, distr_labels, epoch=i, batch_idx=batch_idx):
                continue

            if settings.device.type == "cuda":
                (
                    distr_points,
                    distr_covs,
                    distr_weights,
                    triplets_idx,
                ) = _move_batch_to_device(
                    (distr_points, distr_covs, distr_weights, triplets_idx),
                    device=settings.device,
                )

            if ot_params is not None and ot_params.identical_supports:
                distr_points = points
                distr_covs = covs

            optimizer.zero_grad()
            loss = torch.zeros((), device=settings.device)

            if debug_anomaly:
                with torch.autograd.detect_anomaly(check_nan=True):
                    loss = ggml_fn(
                        loss,
                        distr_points,
                        distr_weights,
                        distr_covs,
                        triplets_idx,
                        map_A,
                        alpha,
                        ot_params=ot_params,
                    )
                    assert_finite_tensor(loss.detach(), name="objective loss", epoch=i, batch_idx=batch_idx)
                    loss.backward()
            else:
                loss = ggml_fn(
                    loss,
                    distr_points,
                    distr_weights,
                    distr_covs,
                    triplets_idx,
                    map_A,
                    alpha,
                    ot_params=ot_params,
                )
                assert_finite_tensor(loss.detach(), name="objective loss", epoch=i, batch_idx=batch_idx)
                loss.backward()
            if map_A.grad is not None:
                assert_finite_tensor(
                    map_A.grad.detach(),
                    name="map_A.grad (after objective)",
                    epoch=i,
                    batch_idx=batch_idx,
                )

            grad_norm = 0.0
            if (verbose or optim_params.stop_thr is not None) and map_A.grad is not None:
                grad_norm = torch.linalg.norm(map_A.grad).item()

            if isinstance(reg_params, MIRegularizationParams):
                regul_loss = reg_fn(map_A, distr_points, distr_covs, distr_weights, reg_params)
            else:
                regul_loss = reg_fn(map_A, reg_params)
            assert_finite_tensor(regul_loss.detach(), name="regularization loss", epoch=i, batch_idx=batch_idx)

            regul_loss.backward()
            if map_A.grad is not None:
                assert_finite_tensor(
                    map_A.grad.detach(),
                    name="map_A.grad (after regularization)",
                    epoch=i,
                    batch_idx=batch_idx,
                )
            optimizer.step()
            assert_finite_tensor(map_A.detach(), name="map_A (post-step)", epoch=i, batch_idx=batch_idx)

            if optim_params.stop_thr is not None and grad_norm < optim_params.stop_thr:
                converged = True

            if verbose:
                loss_val = float(loss.item())
                reg_val = float(regul_loss.item())
                pbar.set_postfix(
                    {
                        "obj_loss": f"{loss_val:.4f}",
                        "reg_loss": f"{reg_val:.4f}",
                        "obj_grad": f"{grad_norm:.3e}",
                    }
                )

        epoch_times.append(time.time() - start_epoch)
        if return_map_A_history:
            map_A_history.append(snapshot_numpy(map_A))

        is_last = converged or i == optim_params.max_iter
        if (plot_iter > 0 and i % plot_iter == 0) or (plot_iter == -1 and is_last):
            tqdm.write(f"Compute all OT distances after {i} iterations")
            _ = dataset.compute_OT(
                ground_metric=map_A.clone().detach(),
                symbols=["train"] * len(dataset.distribution_labels),
            )

        if converged:
            if verbose:
                tqdm.write(
                    f"Converged obj_grad {grad_norm:.3e} < stop_thr {optim_params.stop_thr:.3e} at iteration {i}"
                )
            break

    epoch_time = np.mean(np.asarray(epoch_times))
    if return_map_A_history:
        return map_A_history, epoch_time
    return snapshot_numpy(map_A), epoch_time
