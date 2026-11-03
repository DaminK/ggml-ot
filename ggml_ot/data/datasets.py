"""Curated CELLxGENE dataset shortcuts.

Each public function in this module wraps :func:`ggml_ot.data.load_cellxgene`
together with :func:`ggml_ot.from_anndata` to return a ready-to-train
:class:`~ggml_ot.data.AnnData_TripletDataset` for one of the cohorts we
benchmark against most often. Dataset metadata (CELLxGENE id, sample/label
columns, label type) lives in the private ``_DATASETS`` registry below; the
source publication for each cohort is recorded as a comment above its entry.

The triplet-construction kwargs (``n_cells``, ``n_triplets``, ``use_rep``,
``group_by``, ``gmm_key``, ...) are the ones you will typically want to tune
— each shortcut forwards them straight to :func:`ggml_ot.from_anndata`.
Set ``n_top_genes`` to keep only the most highly variable genes before
constructing the dataset, and ``normalize=True`` to normalize the returned
dataset supports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ggml_ot.data.cellxgene import load_cellxgene
from ggml_ot.data.interface import from_anndata

if TYPE_CHECKING:
    from ggml_ot.data import AnnData_TripletDataset


_DATASETS: dict[str, dict] = {
    # Pal et al., "A single-cell RNA expression atlas of normal, preneoplastic
    # and tumorigenic states in the human breast", The EMBO Journal (2021).
    "breastcancer": {
        "id": "b8b5be07-061b-4390-af0a-f9ced877a068.h5ad",
        "label_col": "reported_diseases",
        "patient_col": "donor_id",
        "label_type": "categorical",
    },
    # Stephenson et al., "Single-cell multi-omics analysis of the immune
    # response in COVID-19", Nature Medicine (2021).
    "covid": {
        "id": "e6b93598-7ae3-47d6-b971-d3c6d9162175.h5ad",
        "label_col": "disease",
        "patient_col": "donor_id",
        "label_type": "categorical",
    },
    # Kuppe et al., "Spatial multi-omic map of human myocardial infarction",
    # Nature (2022).
    "myocardial_infarction": {
        "id": "c1f6034b-7973-45e1-85e7-16933d0550bc.h5ad",
        "label_col": "patient_group",
        "patient_col": "sample",
        "label_type": "categorical",
    },
    # Lake et al., "An atlas of healthy and injured cell states and niches
    # in the human kidney", Nature (2023).
    "kidney_injury": {
        "id": "1c360b0b-eb2f-45a3-aba9-056026b39fa5.h5ad",
        "label_col": "disease",
        "patient_col": "donor_id",
        "label_type": "categorical",
    },
}

_CELLXGENE_URL = "https://datasets.cellxgene.cziscience.com"


def _load(
    name: str,
    *,
    path=None,
    url: str = _CELLXGENE_URL,
    n_top_genes: int | None = None,
    normalize: bool = False,
    **kwargs,
) -> "AnnData_TripletDataset":
    """Download a registered dataset and wrap it as an ``AnnData_TripletDataset``."""
    entry = _DATASETS[name]
    adata = load_cellxgene(entry["id"], url=url, path=path)
    if n_top_genes is not None:
        import scanpy as sc

        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)

    dataset = from_anndata(
        adata,
        patient_col=entry["patient_col"],
        label_col=entry["label_col"],
        label_type=entry["label_type"],
        **kwargs,
    )
    if normalize:
        dataset.normalize()
    return dataset


def breastcancer(
    *,
    path=None,
    url: str = _CELLXGENE_URL,
    n_top_genes: int | None = None,
    normalize: bool = False,
    **kwargs,
) -> "AnnData_TripletDataset":
    """Load the curated CELLxGENE breast-cancer cohort.

    Wires ``patient_col="donor_id"``, ``label_col="reported_diseases"``,
    and ``label_type="categorical"`` into :func:`ggml_ot.from_anndata`.

    Citation: Pal et al., "A single-cell RNA expression atlas of normal,
    preneoplastic and tumorigenic states in the human breast", The EMBO
    Journal (2021).

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Cache directory override; see :func:`load_cellxgene`.
    url : str, optional
        CELLxGENE base URL.
    n_top_genes : int, optional
        If provided, keep this many highly variable genes before dataset
        construction.
    normalize : bool, default ``False``
        If ``True``, normalize the returned dataset supports.
    **kwargs
        Forwarded to :func:`ggml_ot.from_anndata`, which is where the
        triplet-construction knobs (``n_cells``, ``n_triplets``,
        ``use_rep``, ``group_by``, ``gmm_key``, ...) live.

    Returns
    -------
    AnnData_TripletDataset

    See Also
    --------
    ggml_ot.from_anndata : Underlying dataset constructor; all triplet
        construction kwargs are documented there.
    ggml_ot.data.load_cellxgene : Underlying download/cache helper.
    """
    return _load(
        "breastcancer",
        path=path,
        url=url,
        n_top_genes=n_top_genes,
        normalize=normalize,
        **kwargs,
    )


def covid(
    *,
    path=None,
    url: str = _CELLXGENE_URL,
    n_top_genes: int | None = None,
    normalize: bool = False,
    **kwargs,
) -> "AnnData_TripletDataset":
    """Load the curated CELLxGENE COVID-19 cohort.

    Wires ``patient_col="donor_id"``, ``label_col="disease"``, and
    ``label_type="categorical"`` into :func:`ggml_ot.from_anndata`.

    Citation: Stephenson et al., "Single-cell multi-omics analysis of the
    immune response in COVID-19", Nature Medicine (2021).

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Cache directory override; see :func:`load_cellxgene`.
    url : str, optional
        CELLxGENE base URL.
    n_top_genes : int, optional
        If provided, keep this many highly variable genes before dataset
        construction.
    normalize : bool, default ``False``
        If ``True``, normalize the returned dataset supports.
    **kwargs
        Forwarded to :func:`ggml_ot.from_anndata`, which is where the
        triplet-construction knobs (``n_cells``, ``n_triplets``,
        ``use_rep``, ``group_by``, ``gmm_key``, ...) live.

    Returns
    -------
    AnnData_TripletDataset

    See Also
    --------
    ggml_ot.from_anndata : Underlying dataset constructor; all triplet
        construction kwargs are documented there.
    ggml_ot.data.load_cellxgene : Underlying download/cache helper.
    """
    return _load(
        "covid",
        path=path,
        url=url,
        n_top_genes=n_top_genes,
        normalize=normalize,
        **kwargs,
    )


def myocardial_infarction(
    *,
    path=None,
    url: str = _CELLXGENE_URL,
    n_top_genes: int | None = None,
    normalize: bool = False,
    **kwargs,
) -> "AnnData_TripletDataset":
    """Load the curated CELLxGENE myocardial-infarction cohort.

    Wires ``patient_col="sample"``, ``label_col="patient_group"``, and
    ``label_type="categorical"`` into :func:`ggml_ot.from_anndata`.

    Citation: Kuppe et al., "Spatial multi-omic map of human myocardial
    infarction", Nature (2022).

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Cache directory override; see :func:`load_cellxgene`.
    url : str, optional
        CELLxGENE base URL.
    n_top_genes : int, optional
        If provided, keep this many highly variable genes before dataset
        construction.
    normalize : bool, default ``False``
        If ``True``, normalize the returned dataset supports.
    **kwargs
        Forwarded to :func:`ggml_ot.from_anndata`, which is where the
        triplet-construction knobs (``n_cells``, ``n_triplets``,
        ``use_rep``, ``group_by``, ``gmm_key``, ...) live.

    Returns
    -------
    AnnData_TripletDataset

    See Also
    --------
    ggml_ot.from_anndata : Underlying dataset constructor; all triplet
        construction kwargs are documented there.
    ggml_ot.data.load_cellxgene : Underlying download/cache helper.
    """
    return _load(
        "myocardial_infarction",
        path=path,
        url=url,
        n_top_genes=n_top_genes,
        normalize=normalize,
        **kwargs,
    )


def kidney_injury(
    *,
    path=None,
    url: str = _CELLXGENE_URL,
    n_top_genes: int | None = None,
    normalize: bool = False,
    **kwargs,
) -> "AnnData_TripletDataset":
    """Load the curated CELLxGENE kidney-injury cohort.

    Wires ``patient_col="donor_id"``, ``label_col="disease"``, and
    ``label_type="categorical"`` into :func:`ggml_ot.from_anndata`.

    Citation: Lake et al., "An atlas of healthy and injured cell states and
    niches in the human kidney", Nature (2023).

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Cache directory override; see :func:`load_cellxgene`.
    url : str, optional
        CELLxGENE base URL.
    n_top_genes : int, optional
        If provided, keep this many highly variable genes before dataset
        construction.
    normalize : bool, default ``False``
        If ``True``, normalize the returned dataset supports.
    **kwargs
        Forwarded to :func:`ggml_ot.from_anndata`, which is where the
        triplet-construction knobs (``n_cells``, ``n_triplets``,
        ``use_rep``, ``group_by``, ``gmm_key``, ...) live.

    Returns
    -------
    AnnData_TripletDataset

    See Also
    --------
    ggml_ot.from_anndata : Underlying dataset constructor; all triplet
        construction kwargs are documented there.
    ggml_ot.data.load_cellxgene : Underlying download/cache helper.
    """
    return _load(
        "kidney_injury",
        path=path,
        url=url,
        n_top_genes=n_top_genes,
        normalize=normalize,
        **kwargs,
    )
