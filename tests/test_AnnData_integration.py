"""AnnData integration tests.

Covers training, evaluation, train/test, and tuning flows on AnnData-backed datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import ggml_ot
import pytest
from anndata import AnnData
from .utils.config import get_kwargs


@pytest.mark.anndata
def test_AnnDataTripletDataset_init(anndata_datasets):
    """Tests initialization of AnnDataTripletDataset.

    Uses synthetic data by default, or network data with ``-m network``.
    """
    assert anndata_datasets["datasets"]


@pytest.mark.anndata
def test_AnnDataTripletDataset_train_and_test(anndata_datasets):
    """Tests training and testing of AnnDataTripletDataset on anndata dataset.

    Works with both network (real-world) and synthetic data sources.
    """
    for setup_name, dataset in anndata_datasets["datasets"].items():
        # Train/test on multiple dataset setups
        print(setup_name)
        # Suppress optional warning about gene space projection (occurs with use_rep but not always)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Cannot project W_ggml back to gene space", category=UserWarning)
            map_A = ggml_ot.train(dataset, **get_kwargs("smoke", "train"))
        assert map_A is not None
        plt.close("all")

        scores = ggml_ot.test(dataset, map_A, **get_kwargs("smoke", "test"))
        assert scores is not None
        plt.close("all")


@pytest.mark.anndata
def test_AnnDataTripletDataset_train_test(anndata_datasets):
    """Test train_test of AnnDataTripletDataset on anndata dataset.

    Works with both network (real-world) and synthetic data sources.
    """
    for setup_name, dataset in anndata_datasets["datasets"].items():
        print(setup_name)
        map_A, scores = ggml_ot.train_test(dataset, **get_kwargs("smoke", "train_test"))
        assert map_A is not None
        assert scores is not None
        plt.close("all")


@pytest.mark.anndata
def test_AnnDataTripletDataset_tune(anndata_datasets):
    """Test hyperparameter tuning of AnnDataTripletDataset on anndata dataset.

    Works with both network (real-world) and synthetic data sources.
    """
    for setup_name, dataset in anndata_datasets["datasets"].items():
        print(setup_name)
        # Check hyperparameter tuning
        w_thetas, scores = ggml_ot.tune(dataset, **get_kwargs("smoke", "tune"))
        assert w_thetas is not None
        assert scores is not None
        plt.close("all")


@pytest.mark.gmm
def test_AnnDataTripletDataset_layer_gmm_roundtrip():
    """A GMM fitted on a layer must be fitted in, and reloaded with, that layer instead of ``.X``."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"sample": np.repeat(["s1", "s2", "s3", "s4"], 3), "patient_group": np.repeat(["A", "B"], 6)})
    obs.index = obs.index.astype(str)
    adata = AnnData(X=rng.normal(size=(12, 4)), obs=obs)
    adata.layers["shifted"] = adata.X + 100.0

    ggml_ot.from_anndata(adata, n_cells=3, layer="shifted").fit_gmm(k_comps=2, component_sharing="global")
    mu = np.asarray(adata.uns["gmm_layer_shifted"]["model"]["mu"])
    assert mu.min() > adata.X.max()

    with pytest.warns(UserWarning, match=r"layer not provided"):
        reloaded = ggml_ot.from_anndata(adata, gmm_key="gmm_layer_shifted", gmm_weights_source="components")
    assert reloaded.layer == "shifted"
