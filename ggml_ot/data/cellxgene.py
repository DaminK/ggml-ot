import requests
import scanpy as sc
from pathlib import Path

from ggml_ot import settings


# Module-level flag so the fallback-cache hint is emitted at most once per
# process, regardless of how many datasets are loaded.
_cache_hint_emitted = False


def _resolve_cache_path(path):
    """Pick the cache directory and emit a one-time hint on fallback.

    Precedence: explicit ``path=`` argument → :data:`ggml_ot.settings.datadir`
    → ``Path("data/")`` relative to the current working directory.
    """
    if path is not None:
        return Path(path)
    if settings.datadir is not None:
        return Path(settings.datadir)

    global _cache_hint_emitted
    fallback = Path("data/")
    if not _cache_hint_emitted:
        print(
            f"[ggml_ot] Neither `path=` nor `ggml_ot.settings.datadir` is set; "
            f"caching dataset under {fallback.resolve()}. "
            f"Set `ggml_ot.settings.datadir = ...` or pass `path=` to override."
        )
        _cache_hint_emitted = True
    return fallback


def load_cellxgene(
    dataset_id: str,
    url="https://datasets.cellxgene.cziscience.com",
    path=None,
    load=True,
):
    """Download and cache an AnnData object from CELLxGENE.

    Parameters
    ----------
    dataset_id : str
        CELLxGENE dataset identifier — typically the UUID of the dataset
        version (the ``.h5ad`` suffix is added automatically if missing).
        You can find it as the filename of the download link on the
        dataset's CELLxGENE page (e.g. ``b8b5be07-...-f9ced877a068``).
    url : str, optional
        Base URL of the CELLxGENE dataset repository.
    path : str or pathlib.Path, optional
        Cache directory override. When ``None`` (default),
        :data:`ggml_ot.settings.datadir` is used; if that is also unset, the
        loader falls back to ``Path("data/")`` relative to the current
        working directory and prints a one-time hint.
    load : bool, optional
        Whether to read and return the AnnData object (default ``True``).
        Set to ``False`` to download-only.

    Returns
    -------
    anndata.AnnData, optional
        Returned only when ``load=True``.

    See Also
    --------
    ggml_ot.data.breastcancer, ggml_ot.data.covid,
    ggml_ot.data.myocardial_infarction, ggml_ot.data.kidney_injury :
        Ready-to-train shortcuts for curated CELLxGENE cohorts that wrap
        ``load_cellxgene`` together with :func:`ggml_ot.from_anndata`.
    """
    if ".h5ad" not in dataset_id:
        dataset_id = dataset_id + ".h5ad"
    path = _resolve_cache_path(path)

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
