import numpy as np
import torch

from ggml_ot import settings


def _create_triplets(labels):
    """Creates contrastive triplets (i,j,k) for metric learning where i,j are from the same class and
    j,k are from different classes.

    :param labels: distribution labels
    :type labels: array-like
    :return: tensor containing indices that form contrastive triplets
    :rtype: torch.Tensor
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # 1. Create 3 views of the labels for i, j, k dimensions
    # Shape: (N, 1, 1)
    l_i = labels.view(-1, 1, 1)
    # Shape: (1, N, 1)
    l_j = labels.view(1, -1, 1)
    # Shape: (1, 1, N)
    l_k = labels.view(1, 1, -1)

    # 2. Create the boolean mask using broadcasting
    # Condition 1: c(i) == c(j)
    # Condition 2: c(j) != c(k)
    mask = (l_i == l_j) & (l_j != l_k)

    # 3. Extract indices where the mask is True
    # Returns a tensor of shape (M, 3) where each row is (i, j, k)
    triplets = mask.nonzero(as_tuple=False)

    return triplets


def old_create_triplets(labels, t=5, **kwargs):
    """Creates t triplets for each point for metric learning where i and j are from the same class and
    k is from a different class.

    :param labels: distribution labels to create triplets from
    :type labels: array-like
    :param t: number of neighbors to sample from both the same and different classes, defaults to 5
    :type t: int, optional
    :return: list of created triplets
    :rtype: list of tuples
    """
    labels = np.asarray(labels)
    triplets = []
    replace = any(np.unique(labels, return_counts=True)[1] < t)

    def get_neighbors(class_, skip=None):
        # get t elements from distributions where labels = class
        return settings.numpy_generator.choice(np.where(labels == class_)[0], size=t, replace=replace)

    for j, c_j in enumerate(labels):
        for i in get_neighbors(c_j):
            for c_k in np.unique(labels):
                if c_k != c_j:
                    for k in get_neighbors(c_k):
                        triplets.append((i, j, k))
    return triplets
