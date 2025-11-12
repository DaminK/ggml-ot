import numpy as np
import torch

import tqdm as tqdm
import time
from typing import Literal

from ggml_ot.data import TripletDataset, AnnData_TripletDataset
from ggml_ot.distances import (
    pairwise_gaussian_distance,
    pairwise_mahalanobis_distance,
    _emd2,
)
from ggml_ot.optimization import regularizer_loss, mutual_information_loss

from ggml_ot.experimental import component_margin


def train(
    dataset: TripletDataset | AnnData_TripletDataset,
    alpha: float = 10.0,
    reg: float = 1.0,
    reg_type: Literal[1, 2, "cos"] = 2,
    n_comps: int = 5,
    lr: float = 0.02,
    max_iter: int = 10,
    plot_iter: int | bool = -1,
    verbose: bool = False,
    n_threads: str | int = 128,
    batch_size: int = 128,
    train_size=None,
    return_dataset=True,
    measure_time=False,
    MI_reg: float = 0,  # not officially supported yet
    hellinger_approx=False,  # not officially supported yet
    weight_MI=False,  # not officially supported yet
    **kwargs,
) -> TripletDataset | AnnData_TripletDataset | np.ndarray | tuple[np.ndarray, float]:
    """Perform Supervised Optimal Transport by Ground Metric Learning.

    GGML learns a suitable ground metric that captures the distribution classes (e.g. patient groups) under Optimal Transport.

    .. note:: This package provides functions for :meth:`hyperparameter tuning <ggml_ot.tune>` and :meth:`cross-validation <ggml_ot.train_test>`.

    Parameters
    ----------
    dataset
        A dataset containing triplets of distributions.

        .. seealso:: The documentation for the provided interfaces to :meth:`AnnData <ggml_ot.from_anndata>` and :meth:`numpy arrays <ggml_ot.from_numpy>`.

    alpha
        Margin between distributions from different classes (e.g. disease states). Large values lead to strong separations on the train set, but potential overfitting.

    reg
        Regularization strength.

    reg_type
        Type of regularization, `1 | 2` use the corresponding matrx norm (oversimplified: use L1 for sparsity and L2 for robustness), `"cos"` uses the cosine similarity between subspace axis if you want to enforce orthogonality.

    n_comps
        Number of components in the learned subspace, i.e., rank of the subspace.

    lr
        Learning rate of Adam Optimizer.

    max_iter
        Max number of iterations through the training data (epochs).

    plot_iter
        If `True` plot OT after each epoch. If `-1` plot OT after last epoch of the training, for other `int` plot OT after every i-th epoch.

    verbose
        Whether to print progress information during training.

    n_threads
       Either "max" to use all available threads during calculation or the specifc number of threads, defaults to 4.

    batch_size
        Batch size for the DataLoader, defaults to 128.

    Returns
    -------
    TripletDataset | AnnData_TripletDataset
        If `return_dataset` is set to True, the dataset is returned with the learned ground metric (`dataset.w_theta`).

        If the dataset is of type `AnnData_TripletDataset`, the cells are projected into the learned gene subspace (`dataset.adata.obsm["X_ggml"]`) and the loadings of the gene subspace are stored in `dataset.adata.varm["W_ggml"]`.

    np.ndarray
        If `return_dataset` is False, the learned ground metric `w_theta` is returned as a numpy array.

    tuple[np.ndarray, float]
        If `return_dataset` is False and `measure_time` is True, a tuple with the learned ground metric and the average epoch time is returned.

    """
    if train_size is None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        split_indices = dataset._train_test_split(n_splits=1, train_size=train_size)
        dataloader = torch.utils.data.DataLoader(
            dataset.subset(split_indices[0][0]), batch_size=batch_size, shuffle=True
        )

    # TODO find cleaner way to have 1: many key word params with clear definition and default values on top level function 2: not endless long tupples of passed params + having redundant changes to the func signaturre and this tupple
    ggml_params = {
        "dataloader": dataloader,
        "alpha": alpha,
        "reg": reg,
        "rank_k": n_comps,
        "lr": lr,
        "norm": reg_type,
        "max_iterations": max_iter,
        "diagonal_only": False,  # deprecated
        "random_init": True,  # deprecated
        "verbose": verbose,
        "plot_i_iterations": plot_iter,
        "n_threads": n_threads,
        "measure_time": measure_time,
        "hellinger_approx": hellinger_approx,
        "MI_reg": MI_reg,
        "weight_MI": weight_MI,
    }

    if verbose:
        print(f"Running GGML with alpha: {alpha}, reg: {reg}, reg_type: {reg_type}, n_comps: {n_comps}")

    if dataset.identical_supports:
        w_theta, times = _ggml_identical_supports(**ggml_params)
    else:
        w_theta, times = _ggml_empirical(**ggml_params)

    dataset._w_theta = w_theta
    if isinstance(dataset, AnnData_TripletDataset):
        dataset.project(w_theta)

    if return_dataset:
        return dataset
    else:
        if measure_time:
            return w_theta, times
        else:
            return w_theta


# Add train method to TripletDataset Class
TripletDataset.train = train


def _ggml_init(
    dataloader: torch.utils.data.DataLoader,
    rank_k: int | float | None,
    diagonal_only: bool,
    random_init: bool,
):
    dim = dataloader.dataset.dim

    if rank_k is None:
        rank_k = dim
    rank_k = int(rank_k)

    if diagonal_only:
        w_theta = torch.distributions.uniform.Uniform(-1, 1).sample([dim]) if random_init else torch.ones((dim))
    else:
        # TODO: warning, for rank 1 subsequent computation interprets 1d vector as diagonal (even if diagonal_only is false)
        w_theta = (
            torch.distributions.uniform.Uniform(-1, 1).sample([rank_k, dim])
            if random_init
            else torch.diag(torch.ones((dim)))[:rank_k, :]
        )
    w_theta.requires_grad_(requires_grad=True)
    w_theta.retain_grad()

    return w_theta


def _ggml_empirical(
    dataloader: torch.utils.data.DataLoader,
    alpha: float,
    reg: float,
    rank_k: int,
    lr: float,
    norm: str,
    max_iterations: int,
    diagonal_only: bool,  # TODO remove?
    random_init: bool,
    verbose: bool,
    plot_i_iterations: int | None,
    n_threads: int,
    **kwargs,
):
    w_theta = _ggml_init(dataloader, rank_k, diagonal_only, random_init)

    alpha = torch.scalar_tensor(alpha)
    lambda_ = torch.scalar_tensor(reg)

    epoch_times = []
    dataset = dataloader.dataset

    # Stochastic Gradient Descent
    for i in range(1, max_iterations + 1):
        # Epoch
        optimizer = torch.optim.Adam([w_theta], lr=lr)
        iteration_loss = []
        start_epoch = time.time()

        for distribution_triplets, weight_triplets, _ in tqdm.tqdm(dataloader, disable=1 - verbose):
            # Minibatches
            optimizer.zero_grad()
            loss = torch.scalar_tensor(0, requires_grad=True)

            for t, triplet_supports in enumerate(distribution_triplets):
                triplet_supports.requires_grad_(requires_grad=True)  # TODO check if needed
                triplet_weights = weight_triplets[t] if dataset.weights is not None else None

                loss = loss + _triplet_loss(
                    triplet_supports,
                    w_theta,
                    triplet_weights,
                    alpha,
                    n_threads=n_threads,
                )

            # Regularization
            loss = loss / len(distribution_triplets) + lambda_ * regularizer_loss(w_theta, loss=norm)

            # Book-keeping
            iteration_loss.append(loss.clone().detach().numpy())

            # Gradient Descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            w_theta.grad = None
            w_theta.requires_grad_(requires_grad=True)
            w_theta.retain_grad()

        epoch_times.append(time.time() - start_epoch)

        if verbose:
            print(f"Iteration {i} with Loss  {np.sum(iteration_loss)}")

        if (plot_i_iterations > 0 and i % plot_i_iterations == 0) or (plot_i_iterations == -1 and i == max_iterations):
            print(f"Compute all OT distances after {i} iterations")
            _ = dataset.compute_OT(
                ground_metric=w_theta.detach(),
                symbols=["train"] * len(dataset.distribution_labels),
            )

    epoch_time = np.mean(np.asarray(epoch_times))
    return w_theta.clone().detach().numpy(), epoch_time


def _ggml_identical_supports(
    dataloader: torch.utils.data.DataLoader,
    alpha: float,
    reg: float,
    rank_k: int,
    lr: float,
    norm: str,
    max_iterations: int,
    diagonal_only: bool,  # TODO remove, likely confusing
    random_init: bool,
    verbose: bool,
    plot_i_iterations: int | None,
    n_threads: int,
    MI_reg: float,
    hellinger_approx: bool,
    weight_MI: bool,
    **kwargs,
):
    w_theta = _ggml_init(dataloader, rank_k, diagonal_only, random_init)

    alpha = torch.scalar_tensor(alpha)
    lambda_ = torch.scalar_tensor(reg)

    if MI_reg > 0:
        MI_reg = torch.scalar_tensor(MI_reg)

    epoch_times = []
    dataset = dataloader.dataset

    # Epochs
    for i in range(1, max_iterations + 1):
        optimizer = torch.optim.Adam([w_theta], lr=lr)
        iteration_loss = []
        start_epoch = time.time()

        # Minibatches
        for weight_triplets, _ in tqdm.tqdm(dataloader, disable=1 - verbose):
            optimizer.zero_grad()
            loss = torch.scalar_tensor(0, requires_grad=True)

            # Precompute ground distance with current w_theta for minibatch
            if dataset.covariances is None:
                precomputed_D = pairwise_mahalanobis_distance(dataset.supports, dataset.supports, w_theta)
            else:
                precomputed_D = pairwise_gaussian_distance(
                    dataset.supports,
                    dataset.covariances,
                    w_theta,
                    hellinger_approx=hellinger_approx,
                )

            # Track contribution of components to the relative margin of each triplet
            margins_comp = torch.zeros(len(dataset.supports)) if weight_MI else None

            # Triplet Loss
            for triplet_weight in weight_triplets:
                triplet_weight.requires_grad_(requires_grad=True)

                triplet_loss, triplet_margin_comp = _triplet_loss_identical_supports(
                    precomputed_D,
                    triplet_weight,
                    alpha,
                    n_threads=n_threads,
                    return_comp_margin=weight_MI,
                )
                loss = loss + triplet_loss / len(weight_triplets)

                if weight_MI:
                    margins_comp = margins_comp + triplet_margin_comp

            # KL_D Regularization (Independence of Gaussian Variables)
            if dataset.covariances is not None and MI_reg > 0:
                loss = MI_reg * mutual_information_loss(dataset.covariances, w_theta, weights=margins_comp)

            # L1,L2 Regularization
            loss = loss + lambda_ * regularizer_loss(w_theta, loss=norm)

            # Gradient Descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            w_theta.grad = None
            w_theta.requires_grad_(requires_grad=True)
            w_theta.retain_grad()

            # Book-keeping
            iteration_loss.append(loss.clone().detach().numpy())

        epoch_times.append(time.time() - start_epoch)

        if verbose:
            print(f"Iteration {i} with Loss  {np.sum(iteration_loss)}")

        if (plot_i_iterations > 0 and i % plot_i_iterations == 0) or (plot_i_iterations == -1 and i == max_iterations):
            print(f"Compute all OT distances after {i} iterations")
            _ = dataset.compute_OT(
                ground_metric=w_theta.detach(),
                symbols=["train"] * len(dataset.distribution_labels),
            )

    epoch_time = np.mean(np.asarray(epoch_times))
    return w_theta.clone().detach().numpy(), epoch_time


def _triplet_loss(triplet, w, weights=None, alpha=torch.scalar_tensor(0.1), n_threads: str | int = 8):
    X_i, X_j, X_k = triplet
    W_i, W_j, W_k = weights if weights is not None else (torch.empty(0), torch.empty(0), torch.empty(0))

    D_ij = pairwise_mahalanobis_distance(X_i, X_j, w)
    D_jk = pairwise_mahalanobis_distance(X_j, X_k, w)

    W_ij = _emd2(W_i, W_j, M=D_ij, log=False, numThreads=n_threads)
    W_jk = _emd2(W_j, W_k, M=D_jk, log=False, numThreads=n_threads)

    return torch.nn.functional.relu(W_ij - W_jk + alpha)


def _triplet_loss_identical_supports(
    precomputed_D,
    weights,
    alpha=torch.scalar_tensor(0.1),
    n_threads: str | int = 8,
    return_comp_margin=False,
):
    W_i, W_j, W_k = weights

    if not return_comp_margin:
        W_ij = _emd2(W_i, W_j, M=precomputed_D, numThreads=n_threads)
        W_jk = _emd2(W_j, W_k, M=precomputed_D, numThreads=n_threads)
        W_margin_comp = None
    else:
        W_ij, Pi_ij = _emd2(W_i, W_j, M=precomputed_D, return_pi=True, numThreads=n_threads)
        W_jk, Pi_jk = _emd2(W_j, W_k, M=precomputed_D, return_pi=True, numThreads=n_threads)
        W_margin_comp = component_margin(Pi_ij, precomputed_D, Pi_jk, precomputed_D, alpha)

    return torch.nn.functional.relu(W_ij - W_jk + alpha), W_margin_comp
