import ot
import numpy as np
import torch
import tqdm as tqdm
import time
import os
from .data import scRNA_Dataset
import scanpy as sc
from anndata import AnnData
from ggml_ot.benchmark import evaluate_generalizability, plot_table
import pandas as pd


def anndata_preprocess(
    adata_path,
    patient_col="sample",
    label_col="patient_group",
    subsample_patient_ratio=1,
    n_cells=1000,
    n_feats=None,
    filter_genes=True,
    max_iterations=5,
    plot_i_iterations=5,
):
    adata = sc.read_h5ad(adata_path)
    training_data = scRNA_Dataset(
        adata,
        patient_col=patient_col,
        label_col=label_col,
        subsample_patient_ratio=subsample_patient_ratio,
        n_cells=n_cells,
        n_feats=n_feats,
        filter_genes=filter_genes,
    )
    train_dataset = torch.utils.data.DataLoader(
        training_data, batch_size=128, shuffle=True
    )
    w_theta = ggml(
        train_dataset,
        l=1,
        max_iterations=max_iterations,
        plot_i_iterations=plot_i_iterations,
    )

    if filter_genes:
        gene_var = np.var(adata.X.toarray(), axis=0)
        thresh = np.mean(gene_var)
        adata = adata[:, gene_var > thresh]

    adata = adata.copy()

    adata.uns["w_theta"] = w_theta
    adata.obsm["X_ggml_ot"] = adata.X @ w_theta.transpose()
    return adata


def ggml(
    data: str | AnnData | torch.utils.data.DataLoader | torch.utils.data.Dataset,
    alpha: float = 10,
    reg: float = 0.1,
    rank_k: int = 5,
    lr: float = 0.02,
    norm: str = "fro",
    max_iterations: int = 30,
    diagonal: bool = False,
    random_init: bool = True,
    verbose: bool = True,
    save_path: str = "",
    save_i_iterations: int | None = None,
    plot_i_iterations: int | None = None,
    dataset: torch.utils.data.Dataset | None = None,
    adata: AnnData | None = None,
    n_threads: str | int = 1,  # "max" or num of threads
    train_size: float = 0.75,
    **kwargs,
):
    """
    A method to train Global Metrics as the Ground Metric of an Optimal Transport distance from class labels of distributions. It provides interfaces to Anndata (scRNA-seq data) and Pytorch Dataset/Dataloader.

    :param data: Input Data
    :type data: Anndata | path to Anndata | torch.utils.data.DataLoader | torch.utils.data.Dataset
    :param alpha: Required distance margin between learned cluster, if alpha is a list of values, hyperparameter tuning is used to find the most suitable alpha and if an empty list is given, hyperparameter tuning is done with default values
    :type alpha: float, array-like or empty list
    :param reg: Regularization parameter, if reg is a list of values, hyperparameter tuning is used to find the most suitable reg and if an empty list is given, hyperparameter tuning is done with default values
    :type reg: float, array-like or empty list
    :param rank_k: Rank of the subspace projection, if rank_k is a list of values, hyperparameter tuning is used to find the most suitable rank_k and if an empty list is given, hyperparameter tuning is done with default values
    :type rank_k: int, array-like or empty list
    :param lr: Learning rate
    :type lr: float
    :param norm: Norm used for loss calculation during learning
    :type norm: str
    :param max_iterations: Amount of learning iterations to perform
    :type max_iterations: int
    :param diagonal: True => initialize the to be learned weights with a diagonal matrix, False => initialize with a full matrix
    :type diagonal: bool
    :param random_init: True => initialize the to be learned weights with random values drawn from [-1, 1], False => initialize with ones
    :type random_init: bool
    :param verbose: True => print debug and progress information during processing
    :type verbose: bool
    :param save_path: path to save weights matrix
    :type save_path: string
    :param save_i_iterations: saves every ith iteration of the learned weights
    :type save_i_iterations: int
    :param plot_i_iterations: plots every ith iteration of the learned weights
    :type plot_i_terations: int
    :param dataset: Only applies when using a DataLoader as input and when plotting during learning is used - contains data to compute the OT distances on
    :type dataset: Dataset
    :param adata: When a Dataset or DataLoader is passed for training, but results should be returned as part of the Anndata object,
    :type adata: Anndata
    :param n_threads: either "max" to use all available threads during calculation or the specifc number of threads, defaults to 1
    :type n_threads: string, int
    :param train_size: train_size for hyperparameter tuning, defaults to 0.75
    :type train_size: float
    :param kwargs: Other keyword arguments are passed through to scRNA_Dataset which provides an Pytorch DataLoader for Anndata objects (see docs of scRNA_Dataset for details)
    :type kwargs: key, value pairings
    :return: if train_size not None, returns the trained w_theta and the hyperparamaters it was trained with, otherwise only returns w_theta
    :rtype: array-like or array-like, dict
    """

    # Anndata objects or path to Anndata objects -> pytorch dataset
    if isinstance(data, str) or isinstance(data, AnnData):
        # TODO we should also accept types of os.pth or Pathlib
        if isinstance(data, str) and not os.path.exists(data):
            raise Exception(f"AnnData File not found under the path {data}")
        data = scRNA_Dataset(data, **kwargs)
        adata = data.adata

    # pytorch dataset -> dataloader
    if isinstance(data, torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
        if hasattr(data, "adata"):
            adata = data.adata
    elif isinstance(data, torch.utils.data.DataLoader):
        dataloader = data
        data = dataset
    else:
        raise Exception(f"Input datatype {type(data)} is not supported yet")

    tune = False
    if isinstance(alpha, list) and len(alpha) == 0:
        alpha = [0.1, 1, 10]
        tune = True
    if isinstance(reg, list) and len(reg) == 0:
        reg = [0.01, 0.1, 1, 10]
        tune = True
    if isinstance(rank_k, list) and len(rank_k) == 0:
        rank_k = [3, 5]
        tune = True

    n_splits = 1
    best_W = None
    best = {"score": -np.inf, "knn_acc": None, "alpha": None, "reg": None, "k": None}
    if not isinstance(alpha, list) and not isinstance(alpha, np.ndarray):
        alpha = [alpha]
    else:
        tune = True
        n_splits = 5
    if not isinstance(reg, list) and not isinstance(reg, np.ndarray):
        reg = [reg]
    else:
        tune = True
        n_splits = 5
    if not isinstance(rank_k, list) and not isinstance(rank_k, np.ndarray):
        rank_k = [rank_k]
    else:
        tune = True
        n_splits = 5

    if tune:
        print("Starting the hyperparameter tuning")

    for a_ in alpha:
        for reg_ in reg:
            for rank_k_ in rank_k:
                method = f"a={a_}, l={reg_}, k={rank_k_}"
                knn = np.zeros(n_splits)
                mi = np.zeros(n_splits)
                ari = np.zeros(n_splits)
                vi = np.zeros(n_splits)
                for i in range(n_splits):
                    if tune:
                        data = scRNA_Dataset(
                            adata, train_size=train_size, filter_genes=False, **kwargs
                        )
                        dataloader = torch.utils.data.DataLoader(
                            data, batch_size=128, shuffle=True
                        )
                    # perform ggml on dataloader
                    w_theta = _ggml(
                        dataloader=dataloader,
                        alpha=a_,
                        reg=reg_,
                        rank_k=rank_k_,
                        lr=lr,
                        norm=norm,
                        max_iterations=max_iterations,
                        diagonal=diagonal,
                        random_init=random_init,
                        verbose=verbose,
                        save_path=save_path,
                        save_i_iterations=save_i_iterations,
                        plot_i_iterations=plot_i_iterations,
                        n_threads=n_threads,
                        dataset=data,
                    )
                    if tune:
                        try:
                            knn_, mi_, ari_, vi_ = evaluate_generalizability(
                                dataset=data,
                                w_theta=w_theta,
                                method=method,
                                verbose=verbose,
                            )
                            knn[i] = knn_
                            mi[i] = mi_
                            ari[i] = ari_
                            vi[i] = vi_
                        except ValueError as e:
                            print(
                                f"Skipping split {i + 1} with config a={a_}, reg={reg_}, rank_k={rank_k_}: {e}"
                            )
                            continue
                if tune:
                    score = (
                        np.mean(knn) / 2
                        + (np.mean(mi) / 3 + np.mean(ari) / 3 + np.mean(vi) / 3) / 2
                    )
                    df = pd.DataFrame(
                        [
                            {
                                "method": f"a={a_}, l={reg_}, k={rank_k_}",
                                "knn_acc": np.mean(knn),
                                "mi": np.mean(mi),
                                "ari": np.mean(ari),
                                "vi": np.mean(vi),
                                "score": score,
                            }
                        ]
                    )
                    plot_table(df, print_latex=False)
                    if score > best["score"]:
                        best["knn_acc"] = np.mean(knn)
                        best["score"] = score
                        best["alpha"] = a_
                        best["reg"] = reg_
                        best["k"] = rank_k_
                        best_W = w_theta

    if adata is None:
        if not tune:
            return w_theta
        else:
            w_theta = best_W
            return w_theta, best
    else:
        if tune:
            w_theta = best_W
        adata.uns["W_ggml"] = w_theta
        if "use_rep_GGML" in adata.uns.keys() and adata.uns["use_rep_GGML"] is not None:
            adata.obsm["X_ggml"] = adata.obsm[adata.uns["use_rep_GGML"]] @ np.transpose(
                w_theta
            )
        else:
            adata.obsm["X_ggml"] = adata.X @ np.transpose(w_theta)
            adata.varm["W_ggml"] = np.transpose(
                w_theta
            )  # cause its neat, but only possible when learning on the full gene space in .X
            # TODO decide whether this just convolutes the code as we set it to the uns field for compability reasons anyway
        if not tune:
            return adata
        else:
            return adata, best


def _ggml(
    dataloader: torch.utils.data.DataLoader,
    alpha: float,
    reg: float,
    rank_k: int,
    lr: float,
    norm: str,
    max_iterations: int,
    diagonal: bool,
    random_init: bool,
    verbose: bool,
    save_path: str,
    save_i_iterations: int | None,
    plot_i_iterations: int | None,
    dataset: type[scRNA_Dataset] | None,
    n_threads: int,
):
    center = next(iter(dataloader))[1]
    if len(center) == 0:
        dim = next(iter(dataloader))[0].shape[-1]
    else:
        center = center[0]
        dim = center.shape[-1]

    if rank_k is None:
        rank_k = dim
        # TODO: warning, for rank 1 subsequent computation interprets 1d vector as diagonal

    if verbose:
        print(f"Running GGML with alpha: {alpha}, reg: {reg}, rank: {rank_k}")

    alpha = torch.scalar_tensor(alpha)
    lambda_ = torch.scalar_tensor(reg)

    torch.manual_seed(42)  # TODO: remove?
    if diagonal:
        w_theta = (
            torch.distributions.uniform.Uniform(-1, 1).sample([dim])
            if random_init
            else torch.ones((dim))
        )
    else:
        w_theta = (
            torch.distributions.uniform.Uniform(-1, 1).sample([rank_k, dim])
            if random_init
            else torch.diag(torch.ones((dim)))[:rank_k, :]
        )

    w_theta.requires_grad_(requires_grad=True)
    w_theta.retain_grad()

    epoch_times = []

    for i in range(1, max_iterations + 1):
        # Iterations
        optimizer = torch.optim.Adam([w_theta], lr=lr)
        iteration_loss = []
        start_epoch = time.time()
        disable = 1 - verbose
        for triplets, centers, labels in tqdm.tqdm(dataloader, disable=disable):
            # Minibatches
            optimizer.zero_grad()
            loss = torch.scalar_tensor(0, requires_grad=True)

            # Centroids distances can be precomputed for each minibatch
            if len(center) > 0:
                precomputed_D = pairwise_mahalanobis_distance(center, center, w_theta)

            for trip, labels in zip(triplets, labels):
                trip.requires_grad_(requires_grad=True)
                if len(center) == 0:
                    loss = loss + triplet_loss(
                        trip, w_theta, alpha, n_threads=n_threads
                    )
                else:
                    loss = loss + triplet_loss_clusters(
                        trip, precomputed_D, w_theta, alpha, n_threads=n_threads
                    )

            # Regularization
            loss = loss / len(triplets) + lambda_ * regularizer_loss(w_theta, loss=norm)
            loss.backward()
            iteration_loss.append(loss.clone().detach().numpy())

            optimizer.step()
            optimizer.zero_grad()

            w_theta.grad = None
            w_theta.requires_grad_(requires_grad=True)
            w_theta.retain_grad()

        epoch_times.append(time.time() - start_epoch)

        if verbose:
            print(f"Iteration {i} with Loss  {np.sum(iteration_loss)}")

        if save_i_iterations is not None and i % save_i_iterations == 0:
            np.save(save_path + f"/theta_{alpha}_{reg}_{rank_k}_iter{i}_L{norm}.npy")

        if (
            dataset is not None
            and plot_i_iterations is not None
            and i % plot_i_iterations == 0
        ):
            print(f"Compute all OT distances after {i} iterations")
            _ = dataset.compute_OT_on_dists(w=w_theta.clone().detach().numpy())

    return w_theta.clone().detach().numpy()


def triplet_loss(triplet, w, alpha=torch.scalar_tensor(0.1), n_threads: str | int = 8):
    X_i, X_j, X_k = triplet

    D_ij = pairwise_mahalanobis_distance(X_i, X_j, w)
    D_jk = pairwise_mahalanobis_distance(X_j, X_k, w)

    W_ij = _emd2(
        torch.empty(0), torch.empty(0), M=D_ij, log=False, numThreads=n_threads
    )  # noqa
    W_jk = _emd2(
        torch.empty(0), torch.empty(0), M=D_jk, log=False, numThreads=n_threads
    )  # noqa

    return torch.nn.functional.relu(W_ij - W_jk + alpha)


def triplet_loss_clusters(
    triplet, precomputed_D, w, alpha=torch.scalar_tensor(0.1), n_threads: str | int = 8
):
    P_i, P_j, P_k = triplet

    W_ij = _emd2(P_i, P_j, M=precomputed_D, log=False, numThreads=n_threads)  # noqa
    W_jk = _emd2(P_j, P_k, M=precomputed_D, log=False, numThreads=n_threads)  # noqa

    return torch.nn.functional.relu(W_ij - W_jk + alpha)


def _emd2(*args, **kwargs):
    # Wrapper for emd2 to decide if we use exact solver on CPU or approximative solver on GPU
    if False:  # torch.cuda.is_available():
        # GPU with CUDA
        return ot.bregman.sinkhorn_stabilized(
            *args, reg=1.0, **kwargs
        )  # TODO test this, it seems to be really unstable
    else:
        # CPU
        return ot.emd2(*args, **kwargs)


def regularizer_loss(w_theta, loss, order=2):
    # TODO make better conventions of loss type and order
    if order is None:
        order = loss
    if loss == "cos":
        if w_theta.shape[0] == 2:
            return torch.abs(
                torch.dot(w_theta[0, :], w_theta[1, :])
                / (
                    torch.linalg.norm(w_theta[0, :], ord=order)
                    * torch.linalg.norm(w_theta[1, :], ord=order)
                )
            )
        else:
            return pairwise_cosine_sim(w_theta, order)
    else:
        return torch.linalg.norm(w_theta, ord=loss)


def pairwise_cosine_sim(Vs, order):
    # Computes sum over pairwise cosine distances of columns in w
    loss = 0
    for i, v1 in enumerate(Vs):
        for j, v2 in enumerate(Vs):
            if i > j:
                loss = loss + torch.abs(
                    torch.dot(v1, v2)
                    / (
                        torch.linalg.norm(v1, ord=order)
                        * torch.linalg.norm(v2, ord=order)
                    )
                )
    return loss


def pairwise_mahalanobis_distance(X_i, X_j, w):
    # W has shape (rank k<=dim) x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m

    # Transform poins of X_i,X_j according to W
    if w.dim() == 1:
        # assume cov=0, scale dims by diagonal
        proj_X_i = X_i * w[None, :]
        proj_X_j = X_j * w[None, :]

    else:
        w = torch.transpose(w, 0, 1)
        proj_X_i = torch.matmul(X_i, w)
        proj_X_j = torch.matmul(X_j, w)

    return torch.linalg.norm(
        proj_X_i[:, torch.newaxis, :] - proj_X_j[torch.newaxis, :, :], dim=-1
    )
