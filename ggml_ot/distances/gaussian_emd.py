import torch
import numpy as np

from ggml_ot.distances.utils import FastMatSqrt


def pairwise_gaussian_distance(mu, sigma, w, hellinger_approx=False):
    if isinstance(mu, np.ndarray):
        mu = torch.from_numpy(mu).to(dtype=torch.float32)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma).to(dtype=torch.float32)
    if isinstance(w, np.ndarray):
        w = torch.from_numpy(w).to(dtype=torch.float32)

    if sigma.dim == 2:
        print(sigma.shape)
        print("not implemented yet")
        sigma = torch.diag_embed(sigma, dim1=1, dim2=2)

    if w == "euclidean":
        # Corresponds to identitiy transformation
        w_mu = mu.float()
        w_sigma = sigma.float()
    else:
        # Transform Gaussians
        w_mu = torch.einsum("kd,cd->ck", w.float(), mu.float())

        # compute matrix product for each entry in first axis c of sigma (same projection for all gaussian components)
        # comp_c x dim_d x dim_d
        # 'ab,cbd,de->cae'
        w_sigma = torch.einsum("kd,cdj->ckj", w.float(), sigma.float())
        w_sigma = torch.einsum("cjd,dk->cjk", w_sigma.float(), torch.transpose(w.float(), 0, 1))

    if not hellinger_approx:
        root_w_sigma = FastMatSqrt(w_sigma)
    else:
        root_w_sigma = torch.sqrt(torch.diagonal(w_sigma, dim1=1, dim2=2))

    precomputed_distance = torch.zeros((len(mu), len(mu)))
    for i in range(len(mu)):
        for j in range(len(mu)):
            if i < j:
                if not hellinger_approx:
                    wasserstein_bures = (
                        torch.trace(w_sigma[i, :, :])
                        + torch.trace(w_sigma[j, :, :])
                        - 2
                        * torch.trace(
                            torch.squeeze(
                                FastMatSqrt(
                                    torch.matmul(
                                        torch.matmul(root_w_sigma[i], w_sigma[j, :, :]),
                                        root_w_sigma[i],
                                    )[None, :, :]
                                )
                            )
                        )
                    )  # TODO process it once for all (i,j) as flattened first array
                else:
                    wasserstein_bures = torch.square(torch.linalg.norm(root_w_sigma[i, :] - root_w_sigma[j, :]))
                precomputed_distance[i, j] = torch.linalg.vector_norm(w_mu[i, :] - w_mu[j, :]) + wasserstein_bures
            else:
                precomputed_distance[i, j] = precomputed_distance[j, i]
    return precomputed_distance
