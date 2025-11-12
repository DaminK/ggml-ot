import numpy as np
import torch
import scipy
import tqdm

from ggml_ot.distances import pairwise_mahalanobis_distance, _emd2


def _c_component(Pi, D, per_component=True):
    """Computes transported cost between components. Returns a vector of the total cost for each component (per_component = True) or a matrix of the transported cost (per_component = False)"""
    C = Pi * D
    if per_component:
        return (torch.sum(C, axis=1) + torch.sum(C, axis=0)) / 2
    else:
        return C


def _component_margin(Pi_ij, D_ij, Pi_jk, D_jk, alpha=None, per_component=True):
    def _c_component(Pi, D, per_component=True):
        """Computes transported cost between components. Returns a vector of the total cost for each component (per_component = True) or a matrix of the transported cost (per_component = False)"""
        C = Pi * D
        if per_component:
            return (torch.sum(C, axis=1) + torch.sum(C, axis=0)) / 2
        else:
            return C

    # Get transported cost for each
    W_ij_comps = _c_component(Pi_ij, D_ij, per_component)
    W_jk_comps = _c_component(Pi_jk, D_jk, per_component)

    # Get relative margin for each component
    W_margin_comp = W_ij_comps - W_jk_comps
    W_margin_comp = W_margin_comp / torch.sum(W_margin_comp)

    return W_margin_comp


def component_margin(dataset, w_theta, per_class_pair=True, per_component=False, n_threads=200):
    precomputed_D = pairwise_mahalanobis_distance(dataset.supports, dataset.supports, w_theta)

    if per_class_pair:
        classes = np.unique(dataset.distribution_labels, sorted=True)
        class_pairs = np.asarray([(c1, c2) for c1 in classes for c2 in classes if c1 < c2])
        class_pairs = {(c1, c2): i for i, (c1, c2) in enumerate(class_pairs)}
        print(len(class_pairs))
        print(class_pairs)

    n_class_pairs = scipy.special.comb(len(classes), 2, exact=True) if per_class_pair else 1
    print(n_class_pairs)

    margins_comp = (
        torch.zeros((n_class_pairs, len(dataset.supports)))
        if per_component
        else torch.zeros((n_class_pairs, len(dataset.supports), len(dataset.supports)))
    )

    print("Compute for all triplets")
    for i in tqdm.tqdm(range(len(dataset))):
        W_i, W_j, W_k = torch.from_numpy(dataset[i][0])

        c_pair_index = class_pairs[tuple(np.unique(dataset[i][1], sorted=True))] if per_class_pair else 1

        W_ij, Pi_ij = _emd2(W_i, W_j, M=precomputed_D, return_pi=True, numThreads=n_threads)  # noqa
        W_jk, Pi_jk = _emd2(W_j, W_k, M=precomputed_D, return_pi=True, numThreads=n_threads)  # noqa
        margins_comp[c_pair_index, :, :] += _component_margin(
            Pi_ij, precomputed_D, Pi_jk, precomputed_D, per_component=per_component
        )

    margins_comp = (margins_comp / len(dataset.supports)).numpy()

    if per_class_pair:
        return {(c1, c2): margins_comp[i, :, :] for i, (c1, c2) in enumerate(class_pairs)}
    else:
        return np.squeeze(margins_comp)
