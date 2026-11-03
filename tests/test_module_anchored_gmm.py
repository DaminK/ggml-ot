"""Tests for cell-type-anchored GMM fitting (string ``k_comps``).

In anchored mode each component is a cell type with a fixed mean (the
per-distribution centroid) and an estimated covariance/weight, fit in closed
form. These tests cover the closed-form fitter math, the covariance modes
(``full``/``diag``/``adaptive``), label-based responsibilities, schema
provenance, and the public ``fit_gmm`` argument contract.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ggml_ot
from ggml_ot.gmm._anchored_fit import fit_anchored_sample_specific, storage_covariance_type


pytestmark = [pytest.mark.gmm, pytest.mark.anndata]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _anchored_adata():
    """Two distributions; s1 has cell types A & B, s2 only A (ragged K)."""
    rng = np.random.default_rng(0)
    s1_a = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(30, 2))
    s1_b = rng.normal(loc=[5.0, 5.0], scale=0.2, size=(25, 2))
    s2_a = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(40, 2))
    x = np.vstack([s1_a, s1_b, s2_a]).astype(np.float64)
    obs = pd.DataFrame(
        {
            "sample": ["s1"] * 55 + ["s2"] * 40,
            "patient_group": ["A"] * 95,
            "celltype": (["A"] * 30 + ["B"] * 25) + (["A"] * 40),
        }
    )
    obs.index = obs.index.astype(str)
    return AnnData(X=x, obs=obs)


def _fit_anchored_dataset(adata, *, covariance_type="full", **fit_kwargs):
    min_cells = int(adata.obs.groupby("sample").size().min())
    dataset = ggml_ot.from_anndata(
        adata,
        patient_col="sample",
        label_col="patient_group",
        n_cells=min_cells,
    )
    dataset.fit_gmm(k_comps="celltype", covariance_type=covariance_type, verbose=False, **fit_kwargs)
    return dataset


# --------------------------------------------------------------------------- #
# Closed-form fitter math
# --------------------------------------------------------------------------- #
def test_anchored_means_equal_centroids_and_full_covariance():
    # GMM_TEST_KEEP: anchored means must equal label-conditional centroids.
    adata = _anchored_adata()
    x = adata.X
    distribution_index = np.array([0] * 55 + [1] * 40)
    component_codes = np.array([0] * 30 + [1] * 25 + [0] * 40)  # A=0, B=1
    out = fit_anchored_sample_specific(
        x=x,
        distribution_index=distribution_index,
        distribution_ids=np.array(["s1", "s2"]),
        component_codes=component_codes,
        category_names=["A", "B"],
        covariance_mode="full",
        eps=1e-6,
        singularity_handling="guarded",
    )

    mu = out["model_payload"]["mu"]
    var = out["model_payload"]["var"]
    assert mu.shape == (2, 2, 2)  # (distributions, K_max, D)
    assert var.shape == (2, 2, 2, 2)

    # s1 component A (slot 0) and B (slot 1)
    np.testing.assert_allclose(mu[0, 0], x[:30].mean(0), rtol=1e-10)
    np.testing.assert_allclose(mu[0, 1], x[30:55].mean(0), rtol=1e-10)
    # s2 component A (slot 0); slot 1 is padding (zero)
    np.testing.assert_allclose(mu[1, 0], x[55:].mean(0), rtol=1e-10)
    np.testing.assert_array_equal(mu[1, 1], np.zeros(2))

    # Full covariance equals biased sample covariance about the centroid.
    cells = x[:30]
    expected = (cells - cells.mean(0)).T @ (cells - cells.mean(0)) / cells.shape[0]
    np.testing.assert_allclose(var[0, 0], expected, atol=1e-6)

    # Ragged K and weights.
    np.testing.assert_array_equal(out["distribution_n_components"], [2, 1])
    np.testing.assert_allclose(out["distribution_weights"].sum(1), [1.0, 1.0])
    assert out["distribution_weights"][1, 1] == 0.0  # padded slot has zero weight


def test_anchored_diag_mode_stores_per_feature_variance():
    # GMM_TEST_KEEP: diag mode stores (K, D) variances equal to per-feature var.
    adata = _anchored_adata()
    x = adata.X
    out = fit_anchored_sample_specific(
        x=x,
        distribution_index=np.array([0] * 55 + [1] * 40),
        distribution_ids=np.array(["s1", "s2"]),
        component_codes=np.array([0] * 30 + [1] * 25 + [0] * 40),
        category_names=["A", "B"],
        covariance_mode="diag",
        eps=1e-6,
        singularity_handling="guarded",
    )
    assert out["storage_covariance_type"] == "diag"
    var = out["model_payload"]["var"]
    assert var.shape == (2, 2, 2)  # (distributions, K_max, D)
    cells = x[:30]
    np.testing.assert_allclose(var[0, 0], cells.var(0), atol=1e-6)


def test_anchored_responsibilities_are_one_hot_labels():
    # GMM_TEST_KEEP: responsibilities/hard components equal the label slot.
    adata = _anchored_adata()
    out = fit_anchored_sample_specific(
        x=adata.X,
        distribution_index=np.array([0] * 55 + [1] * 40),
        distribution_ids=np.array(["s1", "s2"]),
        component_codes=np.array([0] * 30 + [1] * 25 + [0] * 40),
        category_names=["A", "B"],
        covariance_mode="full",
        eps=1e-6,
        singularity_handling="guarded",
    )
    resp = out["responsibilities"]
    hard = out["hard_components"]
    assert resp.shape == (95, 2)
    np.testing.assert_array_equal(resp.sum(1), np.ones(95))  # exactly one-hot
    np.testing.assert_array_equal(hard[:30], 0)  # s1 type A -> slot 0
    np.testing.assert_array_equal(hard[30:55], 1)  # s1 type B -> slot 1
    np.testing.assert_array_equal(hard[55:], 0)  # s2 type A -> slot 0


def test_adaptive_falls_back_to_diagonal_below_threshold():
    # GMM_TEST_KEEP: adaptive uses full when n_k >= D+1, diagonal otherwise.
    rng = np.random.default_rng(1)
    big = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(50, 2))  # n_k=50 >= D+1 -> full
    small = rng.normal(loc=[9.0, 9.0], scale=1.0, size=(2, 2))  # n_k=2 < D+1 -> diag
    x = np.vstack([big, small])
    out = fit_anchored_sample_specific(
        x=x,
        distribution_index=np.zeros(52, dtype=int),
        distribution_ids=np.array(["s1"]),
        component_codes=np.array([0] * 50 + [1] * 2),
        category_names=["big", "small"],
        covariance_mode="adaptive",
        eps=1e-6,
        singularity_handling="guarded",
    )
    assert out["storage_covariance_type"] == "full"  # materialized as full
    assert out["adaptive_diag_fallback_count"] == 1
    var = out["model_payload"]["var"]
    # small component: diagonal fallback -> off-diagonal is exactly zero
    assert var[0, 1][0, 1] == 0.0
    # big component: generic full covariance -> off-diagonal generally nonzero
    assert abs(var[0, 0][0, 1]) > 0.0


def test_storage_covariance_type_helper():
    assert storage_covariance_type("full") == "full"
    assert storage_covariance_type("adaptive") == "full"
    assert storage_covariance_type("diag") == "diag"
    with pytest.raises(NotImplementedError):
        storage_covariance_type("lowrank")


# --------------------------------------------------------------------------- #
# Public fit_gmm contract + schema
# --------------------------------------------------------------------------- #
def test_fit_gmm_anchored_persists_schema_and_provenance():
    # GMM_TEST_KEEP: anchored fit persists a sample_specific schema + provenance.
    adata = _anchored_adata()
    _fit_anchored_dataset(adata, covariance_type="full", gmm_key="gmm_anchored")
    schema = adata.uns["gmm_anchored"]
    assert schema["component_sharing"] == "sample_specific"
    assert schema["covariance_type"] == "full"
    assert schema["component_definition"] == "cell_type"
    assert schema["anchor_column"] == "celltype"
    assert schema["anchored_covariance_mode"] == "full"
    assert schema["component_cell_types"][0][:2] == ["A", "B"]
    assert schema["component_cell_types"][1][0] == "A"
    assert "anchored_diagnostics" in schema

    # Stored means equal the per-distribution centroids.
    mu = np.asarray(schema["model"]["mu"])
    np.testing.assert_allclose(mu[0, 0], adata.X[:30].mean(0), rtol=1e-6)
    np.testing.assert_allclose(mu[1, 0], adata.X[55:].mean(0), rtol=1e-6)


def test_fit_gmm_anchored_roundtrips_through_from_anndata():
    # GMM_TEST_KEEP: an anchored GMM reloads like any sample_specific GMM.
    adata = _anchored_adata()
    _fit_anchored_dataset(adata, covariance_type="diag", gmm_key="gmm_rt")
    reloaded = ggml_ot.from_anndata(
        adata,
        patient_col="sample",
        label_col="patient_group",
        gmm_key="gmm_rt",
        n_cells=40,
    )
    assert reloaded is not None


def test_fit_gmm_string_k_comps_rejected_for_generic_dataset():
    # GMM_TEST_KEEP: anchoring needs obs labels -> generic datasets are rejected.
    rng = np.random.default_rng(0)
    supports = rng.normal(size=(3, 8, 2)).astype(np.float64)
    dataset = ggml_ot.from_numpy(
        supports=supports,
        distribution_labels=np.array([0, 1, 1], dtype=int),
        identical_supports=False,
    )
    with pytest.raises(ValueError, match="AnnData"):
        dataset.fit_gmm(k_comps="celltype")


def test_fit_gmm_adaptive_rejected_without_anchoring():
    # GMM_TEST_KEEP: adaptive covariance only valid in anchored mode.
    adata = _anchored_adata()
    dataset = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group", n_cells=40)
    with pytest.raises(ValueError, match="adaptive"):
        dataset.fit_gmm(k_comps=2, covariance_type="adaptive")


def test_fit_gmm_anchored_missing_column_raises():
    adata = _anchored_adata()
    dataset = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group", n_cells=40)
    with pytest.raises(KeyError, match="not found in adata.obs"):
        dataset.fit_gmm(k_comps="does_not_exist")


def test_fit_gmm_anchored_warns_on_inert_parameters():
    # Rule 5: inert EM/selection params must warn, not silently no-op.
    adata = _anchored_adata()
    dataset = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group", n_cells=40)
    with pytest.warns(UserWarning, match="ignores EM/k-selection parameters"):
        dataset.fit_gmm(k_comps="celltype", k_selection_metric="heldout_nll", refit="full")


def test_anchored_unassigned_labels_excluded_with_warning():
    adata = _anchored_adata()
    celltype = adata.obs["celltype"].astype("object").to_numpy()
    celltype[:5] = np.nan  # unassigned labels
    adata.obs["celltype"] = celltype
    dataset = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group", n_cells=40)
    with pytest.warns(UserWarning, match="unassigned/NaN cell type"):
        dataset.fit_gmm(k_comps="celltype", gmm_key="gmm_nan", verbose=False)
    comp = adata.obs["gmm_nan_comp"].to_numpy()
    assert (comp[:5] == -1).all()  # unassigned cells flagged
    assert adata.uns["gmm_nan"]["anchored_diagnostics"]["n_unassigned_cells"] == 5
