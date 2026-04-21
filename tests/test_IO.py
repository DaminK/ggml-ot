"""I/O roundtrip tests for AnnData-backed GGML datasets."""

import scanpy as sc
import os
import warnings
import pytest
import ggml_ot


@pytest.mark.anndata
def test_dataset_io(anndata_datasets, tmp_path):
    """Test read/write of trained AnnData and ggml core functions with settings.restore_adata_params=True.

    Uses the shared anndata_datasets fixture. Data source is controlled via
    ``--data-source=synthetic|network`` (synthetic by default).
    """
    ggml_ot.settings.restore_adata_params = True

    # Pick one dataset setup from the shared fixture
    datasets = anndata_datasets["datasets"]
    setup_name = next(iter(datasets))
    dataset = datasets[setup_name]

    # Dataset recreation from previous anndata
    new_adata = dataset.to_anndata()
    with pytest.warns(UserWarning, match=r"Restoring previously stored ggml_params"):
        new_dataset = ggml_ot.from_anndata(new_adata)
    assert new_dataset.use_rep == dataset.use_rep

    # Train dataset and store as anndata
    with warnings.catch_warnings():
        # Ignore warning about missing inverse transform (no .varm[X_pca] in AnnData dataset)
        warnings.filterwarnings(
            "ignore",
            message=r"Cannot project W_ggml back to gene space.*",
            category=UserWarning,
        )
        dataset.train(max_iter=1, plot_iter=0, verbose=False, return_dataset=True)
    assert dataset.map_A is not None
    temp_path = tmp_path / "temp.h5ad"
    dataset.to_anndata().write_h5ad(str(temp_path))

    # Dataset recreation from previously stored anndata
    rel_adata = sc.read_h5ad(str(temp_path))
    with pytest.warns(UserWarning, match=r"Restoring previously stored ggml_params"):
        rel_dataset = ggml_ot.from_anndata(rel_adata)

    if temp_path.exists():
        os.remove(temp_path)

    # Test previously trained ground metric
    scores = rel_dataset.test(n_splits=1, verbose=False, plot_split=False, plot_type=False, print_table=False)
    assert scores is not None
