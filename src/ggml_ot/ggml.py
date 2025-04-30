import ot
from scipy.spatial import distance
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
import tqdm as tqdm
import time
import matplotlib.pyplot as plt
from .plot import plot_ellipses
import os
from .data import scRNA_Dataset


def _ggml_anndata(
    data: str,
    a: float,
    l: float,
    k: int,
    lr: float,
    norm: str,
    max_iterations: int,
    diagonal: bool,
    random_init: bool,
    verbose: bool,
    save_path: str,
    save_i_iterations: int,
    plot_i_iterations: int,
    n_threads: str | int,
    patient_col: str | None = None,
    label_col: str | None = None,
    subsample_patient_ratio: float = 1,
    n_cells: int | None = None,
    n_feats: int | None = None,
    filter_genes: bool = False,
):
    # check if data is a valid path
    if not os.path.exists(data):
        print(f"Error: AnnData File not found under the path {data}")
        return

    training_data = scRNA_Dataset(
        data,
        patient_col,
        label_col,
        subsample_patient_ratio,
        n_cells,
        n_feats,
        filter_genes,
    )

    return _ggml_dataset(
        data=training_data,
        a=a,
        l=l,
        k=k,
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
    )


def _ggml_dataloader(
    dataloader: torch.utils.data.DataLoader,
    a: float,
    l: float,
    k: int,
    lr: float,
    norm: str,
    max_iterations: int,
    diagonal: bool,
    random_init: bool,
    verbose: bool,
    save_path: str,
    save_i_iterations: int,
    plot_i_iterations: int,
    dataset: torch.utils.data.Dataset,
    n_threads: str | int,
):
    dim = next(iter(dataloader))[0].shape[-1]
    if k is None:
        k = dim
        # TODO: warning, for rank 1 subsequent computation interprets 1d vector as diagonal

    if verbose:
        print(f"Running GGML with alpha: {a}, lambda: {l}, rank: {k}")

    alpha = torch.scalar_tensor(a)
    lambda_ = torch.scalar_tensor(l)

    torch.manual_seed(42)  # TODO: remove?
    if diagonal:
        w_theta = (
            torch.distributions.uniform.Uniform(-1, 1).sample([dim])
            if random_init
            else torch.ones((dim))
        )
    else:
        w_theta = (
            torch.distributions.uniform.Uniform(-1, 1).sample([k, dim])
            if random_init
            else torch.diag(torch.ones((dim)))[:k, :]
        )

    w_theta.requires_grad_(requires_grad=True)
    w_theta.retain_grad()

    epoch_times = []

    for i in range(1, max_iterations + 1):
        # Iterations
        optimizer = torch.optim.Adam([w_theta], lr=lr)
        iteration_loss = []
        start_epoch = time.time()

        for triplets, labels in tqdm.tqdm(dataloader):
            # Minibatches
            optimizer.zero_grad()
            loss = torch.scalar_tensor(0, requires_grad=True)
            for trip, labels in zip(triplets, labels):
                trip.requires_grad_(requires_grad=True)
                loss = loss + triplet_loss(trip, w_theta, alpha, n_threads=n_threads)

            # Regularization
            loss = loss / len(triplets) + lambda_ * torch.linalg.norm(w_theta, ord=norm)
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
            np.save(save_path + f"/theta_{a}_{l}_{k}_iter{i}_L{norm}.npy")

        if (
            dataset is not None
            and plot_i_iterations is not None
            and i % plot_i_iterations == 0
        ):
            print(f"Compute all OT distances after {i} iterations")
            _ = dataset.compute_OT_on_dists(w=w_theta.clone().detach().numpy())

    return w_theta.clone().detach().numpy()


def _ggml_dataset(
    dataset: torch.utils.data.Dataset,
    a: float,
    l: float,
    k: int,
    lr: float,
    norm: str,
    max_iterations: int,
    diagonal: bool,
    random_init: bool,
    verbose: bool,
    save_path: str,
    save_i_iterations: int,
    plot_i_iterations: int,
    n_threads: str | int,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    return _ggml_dataloader(
        dataloader=dataloader,
        a=a,
        l=l,
        k=k,
        lr=lr,
        norm=norm,
        max_iterations=max_iterations,
        diagonal=diagonal,
        random_init=random_init,
        verbose=verbose,
        save_path=save_path,
        save_i_iterations=save_i_iterations,
        plot_i_iterations=plot_i_iterations,
        dataset=dataset,
        n_threads=n_threads,
    )


def ggml(
    data: str | torch.utils.data.DataLoader | torch.utils.data.Dataset,
    a: float = 10,
    l: float = 0.1,
    k: int = 5,
    lr: float = 0.01,
    norm: str = "fro",
    max_iterations: int = 30,
    diagonal: bool = False,
    random_init: bool = True,
    verbose: bool = True,
    save_path: str = "",
    save_i_iterations: int | None = None,
    plot_i_iterations: int | None = None,
    dataset: torch.utils.data.Dataset | None = None,
    n_threads: str | int = 1,  # "max" or num of threads
    patient_col: str | None = None,
    label_col: str | None = None,
    subsample_patient_ratio: float = 1,
    n_cells: int | None = None,
    n_feats: int | None = None,
    filter_genes: bool = False,
):
    """
    :param data: Input Data
    :type data: path, Dataloader or Dataset
    :param a: Required distance margin between learned cluster
    :type a: float
    :param l: Regularization parameter
    :type l: float
    :param k: Rank of the subspace projection
    :type k: int
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
    :param n_threads: either "max" to use all available threads during calculation or the specifc number of threads, defaults to 1
    :type n_threads: string, int
    :param patient_col: Only applies when using an AnnData file as input - name of the adata.obs that contains the patient label.
    :type patient_col: string
    :param subsample_patient_ratio: ratio of the patient datasets to subsample.
    :type subsample_patient_ratio: float
    :param n_cells: amount of cells to sample per patient
    :type n_cells: int
    :param n_feats: amount of features to sample per cell
    :type n_feats: int
    :param filter_genes: True => do not use features (genes) that are represented with a low variance in the given data.
    :type filter_genes: bool
    """
    if isinstance(data, str):
        return _ggml_anndata(
            data=data,
            a=a,
            l=l,
            k=k,
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
            patient_col=patient_col,
            label_col=label_col,
            subsample_patient_ratio=subsample_patient_ratio,
            n_cells=n_cells,
            n_feats=n_feats,
            filter_genes=filter_genes,
        )
    elif isinstance(data, torch.utils.data.DataLoader):
        return _ggml_dataloader(
            dataloader=data,
            a=a,
            l=l,
            k=k,
            lr=lr,
            norm=norm,
            max_iterations=max_iterations,
            diagonal=diagonal,
            random_init=random_init,
            verbose=verbose,
            save_path=save_path,
            save_i_iterations=save_i_iterations,
            plot_i_iterations=plot_i_iterations,
            dataset=dataset,
            n_threads=n_threads,
        )
    elif isinstance(data, torch.utils.data.Dataset):
        return _ggml_dataset(
            dataset=data,
            a=a,
            l=l,
            k=k,
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
        )
    else:
        print("error - this input datatype is not supported yet")
        return


def triplet_loss(triplet, w, alpha=torch.scalar_tensor(0.1), n_threads: str | int = 8):
    X_i, X_j, X_k = triplet

    D_ij = pairwise_mahalanobis_distance(X_i, X_j, w)
    D_jk = pairwise_mahalanobis_distance(X_j, X_k, w)

    W_ij = ot.emd2([], [], M=D_ij, log=False, numThreads=n_threads)  # noqa
    W_jk = ot.emd2([], [], M=D_jk, log=False, numThreads=n_threads)  # noqa

    return torch.nn.functional.relu(W_ij - W_jk + alpha)


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


def compute_dists(A, B, theta):
    D = np.zeros((len(A), len(B)))
    for i, x in enumerate(A):
        for j, y in enumerate(B):
            D[i, j] = mahalanobis_distance(
                x, y, W=theta
            )  # np.linalg.norm(np.squeeze(np.dot(theta,x))-np.squeeze((np.dot(theta,y))))
    return D


def mahalanobis_distance(x, y, W=None, M=None):
    if M is None:
        assert W is not None, (
            "Atleast one of M or W need to be passed to Mahalanobis distance"
        )
        M = np.dot(W, W.transpose())
        print(f"Dist on M {distance.mahalanobis(x, y, np.linalg.inv(M))}")
        print(
            f"Dist on W {np.linalg.norm(np.squeeze(np.dot(W, x)) - np.squeeze((np.dot(W, y))))}"
        )
        return np.linalg.norm(np.squeeze(np.dot(W, x)) - np.squeeze((np.dot(W, y))))
    else:
        return distance.mahalanobis(x, y, np.linalg.inv(M))


def logistic(x):
    logi = 2 / (1 + np.exp(-x)) - 1
    if logi < 0:
        logi = 0.1 * logi
    return logi


def create_triplets(distributions, labels):
    triplets = []
    for i, _ in enumerate(distributions):
        for j, _ in enumerate(distributions):
            for k, _ in enumerate(distributions):
                if labels[i] == labels[j] and labels[j] != labels[k] and i != j:
                    triplets.append((i, j, k))
    return triplets


def ggml_notorch(
    distributions,
    labels,
    alpha=0.1,
    num_iter=1000,
    temp=0.01,
    threshold=0.01,
    lambda_=0.001,
):
    # OG Implementation using 0th Order SGD

    # create tripplets
    triplets = []
    for i, _ in enumerate(distributions):
        for j, _ in enumerate(distributions):
            for k, _ in enumerate(distributions):
                if labels[i] == labels[j] and labels[j] != labels[k] and i != j:
                    triplets.append((i, j, k))
    print(triplets)

    # init distance
    dims_n = distributions[0].shape[-1]
    theta = np.random.uniform(-1, 1, size=(dims_n, dims_n))
    # theta = np.asarray([[1,0],[0,1]],dtype="f") #np.identity(dims_n)
    # plot_ellipses(theta)

    # plot_ellipses(np.dot(theta,np.transpose(theta)))

    ##theta= np.tril(theta) + np.triu(theta.T, 1)  #enforce symetric
    ##theta = nearestPD(theta) #enforce positive semidefinit
    last_losses = []

    for s in range(num_iter):
        losses = []
        losses_diff = []
        losses_new = []

        t = temp * np.exp(-s / num_iter)

        # Random direction for params

        # x, y = np.random.randint(0,dims_n,size=(2))

        # new_val = np.random.uniform()

        # theta_new[x,y] = new_val

        # theta_new = np.copy(theta) + np.random.normal(0, 1, size=theta.shape)

        random_direction = np.random.normal(0, 0.2, size=theta.shape)
        ##mask = np.random.uniform(0,1,size=theta.shape)<2/3
        ##theta_new[mask] = theta[mask]  #50% chance to change a entry in the params or set change to 0

        ##theta_new= np.tril(theta_new) + np.triu(theta_new.T, 1)

        ##theta_new = np.dot(theta_new,theta_new.transpose()) #to enforce theta_new is positive semi-definite (for mahalanobis norm)

        ##if not isPD(theta_new):
        ##    theta_new = nearestPD(theta_new)

        ###print("Theta update:")
        ###print(np.dot(theta_new-theta,np.transpose(theta_new-theta)))

        for i, j, k in tqdm(triplets):
            # if np.random.random() < 1/2:
            #    continue #only compute 1/3 of dataset in each iteration

            # Compute optimal transport plan for fixed metric
            # d = lambda x,y: distance.mahalanobis(x,y,theta)

            ##assert isPD(theta_new), "Theta(k-1) is not positive semi-definite"
            dist_ij = compute_dists(distributions[i], distributions[j], theta)
            pi_ij = ot.emd([], [], M=dist_ij, log=False)

            dist_jk = compute_dists(distributions[j], distributions[k], theta)
            pi_jk = ot.emd([], [], M=dist_jk, log=False)

            # d_new = lambda x,y: distance.mahalanobis(x,y,theta_new)
            ##assert isPD(theta_new), "Theta(k) is not positive semi-definite"
            dist_ij_new = compute_dists(
                distributions[i], distributions[j], theta + random_direction
            )
            pi_ij_new = ot.emd([], [], M=dist_ij_new, log=False)
            dist_jk_new = compute_dists(
                distributions[j], distributions[k], theta + random_direction
            )
            pi_jk_new = ot.emd([], [], M=dist_jk_new, log=False)

            loss = (
                np.sum(np.multiply(dist_ij, pi_ij))
                - np.sum(alpha * np.multiply(dist_jk, pi_jk))
                + lambda_ * np.linalg.norm(np.dot(theta, theta.transpose()), ord=1)
            )  # min

            # print(loss)
            loss_new = (
                np.sum(np.multiply(dist_ij_new, pi_ij_new))
                - np.sum(alpha * np.multiply(dist_jk_new, pi_jk_new))
                + lambda_
                * np.linalg.norm(
                    np.dot(
                        theta + random_direction, (theta + random_direction).transpose()
                    ),
                    ord=1,
                )
            )  # min

            # print(dist_ij_new)
            # print(dist_jk_new)

            loss_diff = loss - loss_new
            # print(loss_diff)
            # print(f"loss diff {loss_diff}")

            # theta[theta<0]=0 #TODO solve that inverse cov can't contain zeros but we have not modeled bounds

            losses.append(loss)
            losses_new.append(loss_new)
            losses_diff.append(loss_diff)
            # print(f"Triplet ({i,j,k}) loss: {loss} loss_diff: {loss_diff}")

        print("Theta:")
        print(theta)

        # theta_update_scale = logistic(np.quantile(losses_diff,q=0.8)) * t

        theta_update_scale = logistic(np.average(losses_diff)) * t
        # replaced by logistic
        # if theta_update_scale > 1:
        # we can't overshoot our positive semi-definite matrix
        #    theta_update_scale = 1
        # if theta_update_scale < 0:
        #    print(theta_update_scale)

        theta = theta + theta_update_scale * random_direction

        if theta[1, 1] < 0:
            theta = -1 * theta
        # theta = theta / np.linalg.norm(theta) #To avoid overflows
        ##if not isPD(theta):
        ##   theta = nearestPD(theta)

        print(
            f"{s}. iteration | avg loss {np.average(losses)} min {np.min(losses)} max {np.max(losses)}| new avg loss {np.average(losses_new)} | loss change {np.average(losses_diff)} | step {theta_update_scale}"
        )
        print("Theta New:")
        print(theta)
        # plot_ellipses(theta)
        plot_ellipses(np.dot(theta, np.transpose(theta)))
        plt.show()

        last_losses.append(np.average(losses_diff))
        if s > 5:
            print(f"Loss var {np.var(last_losses[-5:-1])}")
            if np.average(last_losses[-3:-1]) < threshold:
                print("converged")
                print(np.var(last_losses[-3:-1]))
                break

    return theta


# theta = gg_ml(dists,labels,alpha=0.1,num_iter=30,temp=1.0,lambda_=0.1)


def get_optimal_pi():
    pass


def get_optimal_theta():
    pass


def pairwise_mahalanobis_distance_npy(X_i, X_j=None, w=None, numThreads=32):
    # W has shape dim x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m
    if X_j is None:
        if w is None or isinstance(w, str):
            return pairwise_distances(
                X_i, metric=w, n_jobs=numThreads
            )  # cdist .. ,X_j)
        else:
            if w.ndim == 2 and w.shape[0] == w.shape[1]:
                return pairwise_distances(
                    X_i, metric="mahalanobis", n_jobs=numThreads, VI=w
                )
            else:
                X_j = X_i
    # Transform poins of X_i,X_j according to W
    # refactor pls
    if w is None or isinstance(w, str):
        return pairwise_distances(
            X_i, X_j, metric=w, n_jobs=numThreads
        )  # cdist .. ,X_j)
    # elif w.ndim == 2 and w.shape[0]==w.shape[1] and len(w)>1000:
    # mahalanobis matrix is to large to compute distances, use cholesky factorization instead M=wT w

    # return pairwise_distances(X_i,metric="mahalanobis",n_jobs=numThreads,VI =w)
    #
    # return scipy.spatial.distance.squareform(scipy.spatial.distance.cdist(X_i,X_j,metric=w))

    # Assume w is cov matrix of mahalanobis distance
    elif w.ndim == 1:
        # assume cov=0, scale dims by diagonal
        w = np.diag(w)
        proj_X_i = np.matmul(X_i, w)
        proj_X_j = np.matmul(X_j, w)

        # proj_X_i = X_i * w[None,:]
        # proj_X_j = X_j * w[None,:]
    else:
        w = np.transpose(w)
        proj_X_i = np.matmul(X_i, w)
        proj_X_j = np.matmul(X_j, w)

    # print("projected shape")
    # print(proj_X_i.shape)
    return np.linalg.norm(
        proj_X_i[:, np.newaxis, :] - proj_X_j[np.newaxis, :, :], axis=-1
    )


def OLD_pairwise_mahalanobis_distance_npy(X_i, X_j, w=None):
    # W has shape dim x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m
    if w is None:
        w = np.identity(X_i.shape[-1])
    else:
        w = w.astype("f")

    X_i = X_i.astype("f")
    X_j = X_j.astype("f")

    # Transform poins of X_i,X_j according to W
    if w.ndim == 1:
        # assume cov=0, scale dims by diagonal
        # w = np.diag(w)
        # proj_X_i = np.matmul(X_i,w)
        # proj_X_j = np.matmul(X_j,w)

        proj_X_i = X_i * w[None, :]
        proj_X_j = X_j * w[None, :]

    else:
        w = np.transpose(w)
        proj_X_i = np.matmul(X_i, w)
        proj_X_j = np.matmul(X_j, w)

    return np.linalg.norm(
        proj_X_i[:, np.newaxis, :] - proj_X_j[np.newaxis, :, :], axis=-1
    )
