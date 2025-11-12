import torch
import numpy as np


def regularizer_loss(w_theta, loss, order=2):
    # TODO make better conventions of loss type and order
    if order is None:
        order = loss
    if loss == "cos":
        if w_theta.shape[0] == 2:
            return torch.abs(
                torch.dot(w_theta[0, :], w_theta[1, :])
                / (torch.linalg.norm(w_theta[0, :], ord=order) * torch.linalg.norm(w_theta[1, :], ord=order))
            )
        else:
            return pairwise_cosine_sim(w_theta, order)
    else:
        return torch.linalg.norm(w_theta, ord=loss)


def pairwise_cosine_sim(w, order):
    """Computes sum over pairwise cosine distances of columns in w"""
    loss = torch.scalar_tensor(0, requires_grad=True)
    for i, v1 in enumerate(w):
        for j, v2 in enumerate(w):
            if i > j:
                loss = loss + torch.abs(
                    torch.dot(v1, v2) / (torch.linalg.norm(v1, ord=order) * torch.linalg.norm(v2, ord=order))
                )
    return loss


def mutual_information(covs):
    """Compute KL of Gaussians"""
    eyes = torch.eye(len(covs[0]))[torch.newaxis, :, :]
    diag_covs = covs * eyes

    n_comps = len(covs)
    KL = torch.zeros((n_comps))
    for i in range(n_comps):
        KL[i] = 0.5 * torch.log(torch.linalg.det(diag_covs[i]) / torch.linalg.det(covs[i]))

    return KL


def mutual_information_loss(sigma, w, weights=None):
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma).to(dtype=torch.float32)
    if isinstance(w, np.ndarray):
        w = torch.from_numpy(w).to(dtype=torch.float32)

    # TODO don't project twice (already project in pairwise emd)
    w_sigma = torch.einsum("kd,cdj->ckj", w.float(), sigma.float())
    w_sigma = torch.einsum("cjd,dk->cjk", w_sigma.float(), torch.transpose(w.float(), 0, 1))
    KL = mutual_information(w_sigma)

    if weights is not None:
        KL = KL * (weights / torch.sum(weights))

    return torch.sum(KL)
