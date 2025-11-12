import os
import requests
import scanpy as sc

from typing import Callable, TypeVar, ParamSpec


# Wrapping docs
P = ParamSpec("P")
T = TypeVar("T")


def wraps(wrapper: Callable[P, T]):
    """An implementation of functools.wraps."""

    def decorator(func: Callable) -> Callable[P, T]:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator


def load_cellxgene(
    dataset_id: str,
    url="https://datasets.cellxgene.cziscience.com",
    save_path=None,
    load=True,
):
    """Loads and caches Anndata object from CELLxGENE.

    :param dataset_id: the filename of the dataset to download
    :type dataset_id: str
    :param url: base URL of the CELLxGENE dataset repository, defaults to "https://datasets.cellxgene.cziscience.com"
    :type url: str, optional
    :param save_path: local path to save the downloaded dataset to, defaults to None
    :type save_path: str, optional
    :param load: whether to load and return the Anndata object, defaults to True
    :type load: bool, optional
    :return: Anndata object if `load=True`
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

    if load:
        return sc.read_h5ad(save_path)
