"""AnnData integration tests.

Covers training, evaluation, train/test, and tuning flows on AnnData-backed datasets.
"""

import matplotlib.pyplot as plt
import warnings
import ggml_ot
import pytest
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
