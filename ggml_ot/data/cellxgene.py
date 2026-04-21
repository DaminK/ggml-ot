import requests
import scanpy as sc
from pathlib import Path


def load_cellxgene(
    dataset_id: str,
    url="https://datasets.cellxgene.cziscience.com",
    path=None,
    load=True,
):
    """Loads and caches Anndata object from CELLxGENE.

    :param dataset_id: the filename of the dataset to download
    :type dataset_id: str
    :param url: base URL of the CELLxGENE dataset repository, defaults to "https://datasets.cellxgene.cziscience.com"
    :type url: str, optional
    :param path: local path for storing and loading datasets, defaults to None
    :type path: PosixPath, str, optional
    :param load: whether to load and return the Anndata object, defaults to True
    :type load: bool, optional
    :return: Anndata object if `load=True`
    :rtype: Anndata, optional
    """
    if ".h5ad" not in dataset_id:
        dataset_id = dataset_id + ".h5ad"
    if path is None:
        path = Path("data/")
    elif type(path) is str:
        path = Path(path)

    # check if dataset exists locally
    if not (path / dataset_id).is_file():
        # create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # download and save the file
        print(f"Downloading dataset {dataset_id} from CELLxGENE...")
        url = f"{url}/{dataset_id}"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path / dataset_id, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Dataset saved to: {str(path / dataset_id)}")

    if load:
        return sc.read_h5ad(path / dataset_id)
