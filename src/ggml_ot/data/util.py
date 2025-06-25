import numpy as np
import os
import requests
import scanpy as sc

import warnings


def download_cellxgene(
    dataset_id: str,
    url="https://datasets.cellxgene.cziscience.com",
    save_path=None,
    load=False,
):
    """Downloads a dataset from the CELLxGENE data portal.

    :param dataset_id: the filename of the dataset to download
    :type dataset_id: str
    :param url: base URL of the CELLxGENE dataset repository, defaults to "https://datasets.cellxgene.cziscience.com"
    :type url: str, optional
    :param save_path: local path to save the downloaded dataset to, defaults to None
    :type save_path: str, optional
    :param load: whether to load the dataset into memory, defaults to False
    :type load: bool, optional
    :return: Anndata object if "load=True"
    :rtype: Anndata, optional
    """
    if ".h5ad" not in dataset_id:
        dataset_id = dataset_id + ".h5ad"
    if save_path is None:
        save_path = f"data/{dataset_id}"

    if not os.path.isfile(save_path):
        url = f"{url}/{dataset_id}"

        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # download and save the file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Dataset saved to: {save_path}")
    else:
        print(f"{save_path} already exists")

    if load:
        return sc.read_h5ad(save_path)


def sample_backed_mode(path, sample_size=None, sample_ratio=None):
    """Samples and saves a subset of cells from a large ".h5ad" file using backed mode.

    :param path: path to the input file
    :type path: str
    :param sample_size: number of cells to sample (required if sample_ratio is None), defaults to None
    :type sample_size: int, optional
    :param sample_ratio: proportion of cells to sample (required if sample_size is None), defaults to None
    :type sample_ratio: float, optional
    :raises ValueError: if neither sample_size nor sample_ratio is provided
    :raises Warning: if the sample_size is larger than the number of cells
    """
    adata = sc.read_h5ad(path, backed="r")

    if sample_ratio is None and sample_size is None:
        raise ValueError("One of sample_size or sample_ratio needs to be passed")

    n_cells = adata.n_obs
    if sample_ratio is not None:
        sample_size = int(n_cells * sample_ratio)

    if sample_size < n_cells:
        idx = np.random.choice(n_cells, size=sample_size, replace=False)
    else:
        warnings.warn(
            "Number of sampled cells larger than number of cells in Anndata, cells will be supersampled"
        )
        idx = np.random.choice(n_cells, size=sample_size, replace=True)

    # load only the subset into memory
    adata_subset = adata[idx, :].to_memory()

    # save the subsampled dataset for later use
    adata_subset.write(f"{path.removesuffix('.h5ad')}_{sample_size}.h5ad")


def create_triplets(distributions, labels):
    """Creates triplets for metric learning where i and j are from the same class and k is from a different class.

    :param distributions: distributions to create triplets from of shape (num_distributions, num_points, num_features)
    :type distributions: list of numpy.ndarray
    :param labels: labels corresponding to distributions
    :type labels: array-like
    :return: list of created triplets
    :rtype: list of tuples
    """
    triplets = []
    for i, _ in enumerate(distributions):
        for j, _ in enumerate(distributions):
            for k, _ in enumerate(distributions):
                if labels[i] == labels[j] and labels[j] != labels[k] and i != j:
                    triplets.append((i, j, k))
    return triplets


def create_t_triplets(distributions, labels, t=5, **kwargs):
    """Creates t triplets for each point for metric learning where i and j are from the same class and
    k is from a different class.

    :param distributions: distributions to create triplets from of shape (num_distributions, num_points, num_features)
    :type distributions: list of numpy.ndarray
    :param labels: labels corresponding to distributions
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
        # TODO optional skip self
        return np.random.choice(np.where(labels == class_)[0], size=t, replace=replace)

    for j, _ in enumerate(distributions):
        c_j = labels[j]
        for i in get_neighbors(c_j):
            for c_k in np.unique(labels):
                if c_k != c_j:
                    for k in get_neighbors(c_k):
                        triplets.append((i, j, k))
    return triplets
