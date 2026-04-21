"""Consolidated GGML-on-GMM integration tests."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ggml_ot


pytestmark = [pytest.mark.gmm, pytest.mark.anndata]
# Comment convention:
# - `GMM_TEST_KEEP`: intended long-term regression/maintenance coverage.
# - `GMM_TEMP_REFACTOR`: temporary implementation/refactor coverage; remove after GMM API freeze.


def _tiny_adata():
    x = np.array(
        [
            [0.1, 0.0, 0.2],
            [0.2, 0.1, 0.3],
            [1.0, 1.2, 1.1],
            [0.9, 1.1, 1.0],
            [2.0, 2.1, 1.9],
            [2.2, 2.0, 2.1],
        ],
        dtype=np.float64,
    )
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "patient_group": ["A", "A", "B", "B", "B", "B"],
        }
    )
    obs.index = obs.index.astype(str)
    return AnnData(X=x, obs=obs)


def _fit_anndata_via_dataset(
    adata,
    *,
    use_rep=None,
    distribution_col="sample",
    label_col="patient_group",
    **fit_kwargs,
):
    min_cells = int(adata.obs.groupby(distribution_col).size().min())
    dataset = ggml_ot.from_anndata(
        adata,
        patient_col=distribution_col,
        label_col=label_col,
        use_rep=use_rep,
        n_cells=min_cells,
    )
    dataset.fit_gmm(**fit_kwargs)
    return adata


def test_sample_specific_gmm_dataset_trains_with_ggml():
    # GMM_TEST_KEEP: end-to-end train() should work on sample-specific GMM datasets.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=2,
        component_sharing="sample_specific",
        gmm_key="gmm_X",
    )
    dataset = ggml_ot.from_anndata(
        adata,
        gmm_key="gmm_X",
    )
    map_A = ggml_ot.train(dataset, max_iter=1, plot_iter=0, verbose=False, return_dataset=False)
    assert map_A is not None


@pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    match=r"__array__ implementation doesn't accept a copy keyword, so passing copy=False failed.",
)
def test_gmm_ggml_on_anndata(anndata_datasets):
    # GMM_TEST_KEEP: end-to-end test() should run on AnnData-derived global GMM datasets.
    adata = anndata_datasets["adata"].copy()

    adata = _fit_anndata_via_dataset(
        adata,
        use_rep="X_pca",
        component_sharing="global",
        distribution_col="sample",
        k_comps=5,
        gmm_key="gmm_X_pca",
    )
    with pytest.warns(UserWarning, match=r"use_rep not provided"):
        dataset = ggml_ot.from_anndata(adata, gmm_key="gmm_X_pca")

    map_A = ggml_ot.train(dataset, max_iter=1, verbose=False, plot_iter=0, return_dataset=False)
    assert map_A is not None
    plt.close("all")

    scores = ggml_ot.test(
        dataset, map_A, n_splits=1, verbose=False, plot_split=False, plot_type=False, print_table=False
    )
    assert scores is not None
    plt.close("all")


@pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    match=r"__array__ implementation doesn't accept a copy keyword, so passing copy=False failed.",
)
def test_gmm_ggml_tune_on_anndata(anndata_datasets):
    # GMM_TEST_KEEP: tune() integration should work on GMM-backed datasets.
    adata = anndata_datasets["adata"].copy()

    adata = _fit_anndata_via_dataset(
        adata,
        use_rep="X_pca",
        component_sharing="global",
        distribution_col="sample",
        k_comps=3,
        gmm_key="gmm_X_pca",
    )
    with pytest.warns(UserWarning, match=r"use_rep not provided"):
        dataset = ggml_ot.from_anndata(adata, gmm_key="gmm_X_pca")

    ws, df_scores = ggml_ot.tune(
        dataset,
        alpha=[0.1],
        reg=[0.1],
        reg_type=[2],
        n_comps=[2],
        mi_reg=[0.1, 10],
        plot_contour=False,
        verbose=False,
        max_iter=1,
        n_splits=1,
        plot_split=False,
        plot_type=False,
        print_table=False,
    )

    assert ws is not None
    assert df_scores is not None
    assert "mi_reg" in list(df_scores.index.names)


def test_gmm_train_diag_bures_without_mi_reg_runs():
    # GMM_TEST_KEEP: training with diag_bures_approx and mi_reg=0 must not crash.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=2,
        component_sharing="sample_specific",
        gmm_key="gmm_diag_bures",
    )
    dataset = ggml_ot.from_anndata(adata, gmm_key="gmm_diag_bures")

    map_A = ggml_ot.train(
        dataset,
        max_iter=1,
        plot_iter=0,
        verbose=False,
        return_dataset=False,
        diag_bures_approx=True,
        mi_reg=0.0,
    )
    assert map_A is not None


def test_gmm_train_diag_bures_with_mi_reg_runs():
    # GMM_TEST_KEEP: training with diag_bures_approx + mi_reg should handle n_comps != dim.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=2,
        covariance_type="diag",
        component_sharing="sample_specific",
        gmm_key="gmm_diag_bures_mi",
    )
    dataset = ggml_ot.from_anndata(adata, gmm_key="gmm_diag_bures_mi")
    assert dataset.covariances.ndim == 4
    assert dataset.covariances.shape[-1] == dataset.covariances.shape[-2]

    map_A = ggml_ot.train(
        dataset,
        max_iter=1,
        plot_iter=0,
        verbose=False,
        return_dataset=False,
        n_comps=2,
        diag_bures_approx=True,
        mi_reg=1.0,
    )
    assert map_A is not None
