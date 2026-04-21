"""Consolidated tests for GMM fitting, schema IO, dataset adapters, and model selection."""

import time

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

import ggml_ot


pytestmark = [pytest.mark.gmm, pytest.mark.anndata]
# Comment convention:
# - `GMM_TEST_KEEP`: intended long-term regression/maintenance coverage.
# - `GMM_TEMP_REFACTOR`: temporary implementation/refactor coverage. TODO: remove before v1.


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
            "celltype": ["c1", "c1", "c2", "c2", "c3", "c3"],
        }
    )
    obs.index = obs.index.astype(str)
    return AnnData(X=x, obs=obs)


def _separable_adata():
    rng = np.random.default_rng(0)
    x1 = rng.normal(loc=0.0, scale=0.15, size=(120, 3))
    x2 = rng.normal(loc=3.0, scale=0.15, size=(120, 3))
    x = np.vstack([x1, x2]).astype(np.float64)
    sample = np.array(["s1"] * 80 + ["s2"] * 80 + ["s3"] * 80)
    obs = pd.DataFrame({"sample": sample, "patient_group": ["A"] * 240})
    obs.index = obs.index.astype(str)
    return AnnData(X=x, obs=obs)


def _heterogeneous_k_adata():
    rng = np.random.default_rng(0)
    s1 = rng.normal(loc=0.0, scale=0.15, size=(220, 2))
    s2_a = rng.normal(loc=-4.0, scale=0.15, size=(110, 2))
    s2_b = rng.normal(loc=4.0, scale=0.15, size=(110, 2))
    s3_a = rng.normal(loc=-3.0, scale=0.15, size=(110, 2))
    s3_b = rng.normal(loc=3.0, scale=0.15, size=(110, 2))
    x = np.vstack([s1, s2_a, s2_b, s3_a, s3_b]).astype(np.float64)
    sample = np.array(["s1"] * 220 + ["s2"] * 220 + ["s3"] * 220)
    obs = pd.DataFrame({"sample": sample, "patient_group": ["A"] * x.shape[0]})
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
    if "refit" not in fit_kwargs:
        k_value = fit_kwargs.get("k_comps")
        if isinstance(k_value, (int, np.integer)):
            fit_kwargs["refit"] = "full"
    dataset.fit_gmm(**fit_kwargs)
    return adata


def _fit_gmm_schema_for_loading(*, component_sharing: str = "global", gmm_key: str) -> AnnData:
    adata = _tiny_adata()
    fit_kwargs = {
        "k_comps": 2,
        "component_sharing": component_sharing,
        "gmm_key": gmm_key,
    }
    _fit_anndata_via_dataset(adata, **fit_kwargs)
    return adata


def _mutate_weight_source_payload(
    adata: AnnData,
    *,
    gmm_key: str,
    drop_stored_weights: bool = False,
    drop_component_assignments: bool = False,
    drop_responsibilities: bool = False,
    invalidate_distribution_ids: bool = False,
) -> None:
    if drop_stored_weights:
        adata.uns[gmm_key]["distribution_weights"] = None
    if drop_component_assignments:
        adata.obs.drop(columns=[f"{gmm_key}_comp"], inplace=True)
    if drop_responsibilities:
        adata.obsm.pop(f"{gmm_key}_resp", None)
    if invalidate_distribution_ids:
        adata.uns[gmm_key]["weight_inference"]["distribution_ids"] = ["__missing_distribution__"]


def test_gaussian_mixture_to_from_dict_roundtrip():
    # GMM_TEST_KEEP: lightweight storage/load roundtrip for pretrained model parameters.
    from ggml_ot.gmm._GaussianMixture import GaussianMixture

    payload = {
        "mu": np.array([[0.0, 0.0], [3.0, 3.0]], dtype=np.float64),
        "var": np.array(
            [
                [[0.1, 0.0], [0.0, 0.1]],
                [[0.1, 0.0], [0.0, 0.1]],
            ],
            dtype=np.float64,
        ),
        "pi": np.array([0.5, 0.5], dtype=np.float64),
        "covariance_type": "full",
    }

    model = GaussianMixture.from_dict(payload)
    restored = GaussianMixture.from_dict(model.to_dict())

    x = np.array([[0.1, -0.1], [2.9, 3.2]], dtype=np.float64)
    pred = restored.predict_hard_components_numpy(x)
    assert pred.tolist() == [0, 1]


def test_gaussian_mixture_sample_single_component():
    # GMM_TEST_KEEP: sampling must work for K=1 models loaded from persisted params.
    from ggml_ot.gmm._GaussianMixture import GaussianMixture

    model = GaussianMixture.from_dict(
        {
            "mu": np.array([[0.0, 0.0]], dtype=np.float64),
            "var": np.array([[[0.1, 0.0], [0.0, 0.1]]], dtype=np.float64),
            "pi": np.array([1.0], dtype=np.float64),
            "covariance_type": "full",
        }
    )
    sampled, _ = model.sample(5)
    sampled = sampled.detach().cpu().float().numpy()
    assert sampled.shape == (5, 2)


def test_gmm_fit_scalar_k_writes_schema_and_assignments():
    # GMM_TEST_KEEP: scalar-k native fit must persist schema + assignments.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=2,
        component_sharing="global",
        gmm_key="gmm_scalar_k",
    )

    assert "gmm_scalar_k" in adata.uns
    cfg = adata.uns["gmm_scalar_k"]
    assert cfg["component_sharing"] == "global"
    assert int(cfg["n_components"]) == 2
    assert cfg["selection"] is None
    assert "model" in cfg and {"mu", "var", "pi"} <= set(cfg["model"].keys())
    assert "gmm_scalar_k_comp" in adata.obs.columns
    assert "gmm_scalar_k_resp" in adata.obsm


def test_fit_gmm_rejects_non_finite_representation_values():
    # GMM_TEST_KEEP: fail fast with explicit message when fit matrix contains NaN/Inf.
    adata = _tiny_adata()
    adata.X[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        _fit_anndata_via_dataset(
            adata,
            k_comps=1,
            component_sharing="global",
            gmm_key="gmm_non_finite",
        )


def test_from_anndata_reads_gmm_schema_with_stored_weights():
    # GMM_TEST_KEEP: from_anndata must reconstruct global GMM datasets.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=2,
        component_sharing="global",
        gmm_key="gmm_schema_global",
    )
    dataset = ggml_ot.from_anndata(adata, gmm_key="gmm_schema_global")
    assert dataset.covariances is not None
    assert dataset.weights is not None


def test_from_anndata_reads_sample_specific_schema():
    # GMM_TEST_KEEP: from_anndata must reconstruct sample-specific GMM datasets.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=2,
        component_sharing="sample_specific",
        gmm_key="gmm_schema_ss",
    )
    dataset = ggml_ot.from_anndata(adata, gmm_key="gmm_schema_ss")
    assert dataset.identical_supports is False
    assert dataset.covariances is not None
    assert dataset.weights is not None


def test_from_anndata_constructor_rejects_gmm_weights_source_without_gmm_key():
    # GMM_TEST_KEEP: explicit GMM weight-source toggle is only valid for gmm_key loading.
    adata = _tiny_adata()
    with pytest.raises(ValueError, match="only supported when gmm_key is provided"):
        ggml_ot.from_anndata(adata, gmm_weights_source="stored")


@pytest.mark.parametrize(
    ("component_sharing", "mutations", "gmm_weights_source"),
    [
        pytest.param(
            "global",
            {"drop_responsibilities": True},
            None,
            id="fallback-stored",
        ),
        pytest.param(
            "global",
            {
                "drop_stored_weights": True,
                "drop_component_assignments": True,
                "drop_responsibilities": True,
            },
            None,
            id="fallback-components-global",
        ),
        pytest.param(
            "sample_specific",
            {
                "drop_stored_weights": True,
                "drop_component_assignments": True,
                "drop_responsibilities": True,
            },
            None,
            id="fallback-components-sample-specific",
        ),
        pytest.param(
            "global",
            {"drop_responsibilities": True},
            "stored",
            id="explicit-stored",
        ),
        pytest.param(
            "global",
            {
                "drop_stored_weights": True,
                "drop_component_assignments": True,
                "drop_responsibilities": True,
            },
            "components",
            id="explicit-components",
        ),
    ],
)
def test_from_anndata_resolves_gmm_weight_source(component_sharing, mutations, gmm_weights_source):
    # GMM_TEST_KEEP: loader should resolve weight sources across fallback and explicit modes.
    gmm_key = f"gmm_weight_source_{component_sharing}_{gmm_weights_source or 'auto'}"
    adata = _fit_gmm_schema_for_loading(component_sharing=component_sharing, gmm_key=gmm_key)
    _mutate_weight_source_payload(adata, gmm_key=gmm_key, **mutations)

    load_kwargs = {"gmm_key": gmm_key}
    if gmm_weights_source is not None:
        load_kwargs["gmm_weights_source"] = gmm_weights_source
    dataset = ggml_ot.from_anndata(adata, **load_kwargs)

    assert dataset.weights is not None
    if component_sharing == "sample_specific":
        assert bool(dataset.identical_supports) is False


@pytest.mark.parametrize(
    ("mutations", "gmm_weights_source", "match"),
    [
        pytest.param(
            {"drop_stored_weights": True},
            "stored",
            "gmm_weights_source='stored' requested",
            id="missing-explicit-stored",
        ),
        pytest.param(
            {},
            "invalid",
            "Unsupported gmm_weights_source",
            id="invalid-source-invalid",
        ),
        pytest.param(
            {},
            "responsibilities",
            "Unsupported gmm_weights_source",
            id="invalid-source-responsibilities",
        ),
        pytest.param(
            {
                "drop_stored_weights": True,
                "drop_component_assignments": True,
                "invalidate_distribution_ids": True,
            },
            None,
            "Could not infer distribution weights",
            id="missing-all-sources",
        ),
    ],
)
def test_from_anndata_gmm_weight_source_errors(mutations, gmm_weights_source, match):
    # GMM_TEST_KEEP: loader should fail loudly when requested/inferred weight sources are unavailable.
    gmm_key = f"gmm_weight_source_error_{gmm_weights_source or 'auto'}"
    adata = _fit_gmm_schema_for_loading(component_sharing="global", gmm_key=gmm_key)
    _mutate_weight_source_payload(adata, gmm_key=gmm_key, **mutations)

    load_kwargs = {"gmm_key": gmm_key}
    if gmm_weights_source is not None:
        load_kwargs["gmm_weights_source"] = gmm_weights_source
    with pytest.raises(ValueError, match=match):
        ggml_ot.from_anndata(adata, **load_kwargs)


@pytest.mark.parametrize(
    ("from_anndata_kwargs", "component_sharing", "gmm_key", "expected_input_identical", "expected_output_identical"),
    [
        pytest.param(
            {"n_cells": 2},
            "auto",
            "gmm_auto_ds",
            False,
            False,
            id="auto->sample-specific",
        ),
        pytest.param(
            {"group_by": "celltype"},
            "auto",
            "gmm_auto_global",
            True,
            True,
            id="auto->global",
        ),
        pytest.param(
            {"group_by": "celltype"},
            "sample_specific",
            "gmm_override",
            True,
            False,
            id="explicit-override",
        ),
    ],
)
def test_fit_gmm_component_sharing_resolution(
    from_anndata_kwargs,
    component_sharing,
    gmm_key,
    expected_input_identical,
    expected_output_identical,
):
    # GMM_TEST_KEEP: auto/explicit component_sharing should resolve to the expected dataset form.
    adata = _tiny_adata()
    dataset = ggml_ot.from_anndata(adata, **from_anndata_kwargs)
    assert bool(dataset.identical_supports) is expected_input_identical

    transformed = dataset.fit_gmm(
        k_comps=1,
        component_sharing=component_sharing,
        refit="full",
        gmm_key=gmm_key,
    )

    assert bool(transformed.identical_supports) is expected_output_identical
    assert isinstance(transformed.weights, torch.Tensor) and transformed.weights.numel() > 0
    if not expected_output_identical:
        assert isinstance(transformed.covariances, torch.Tensor) and transformed.covariances.numel() > 0


def test_fit_gmm_generic_identical_supports_raises():
    # GMM_TEST_KEEP: generic identical supports are unsupported and must fail loudly.
    supports = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    weights = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float64)
    labels = np.array([0, 1], dtype=int)
    dataset = ggml_ot.from_numpy(
        supports=supports,
        distribution_labels=labels,
        weights=weights,
        identical_supports=True,
    )

    with pytest.raises(ValueError, match="identical_supports=True"):
        dataset.fit_gmm(k_comps=1, component_sharing="auto")


def test_fit_gmm_generic_non_identical_mutates_dataset():
    # GMM_TEST_KEEP: generic dataset fitting should mutate dataset into GMM form in-place.
    supports = np.array(
        [
            [[0.1, 0.2], [0.2, 0.1], [0.0, 0.1]],
            [[1.0, 1.1], [0.9, 1.0], [1.1, 0.9]],
            [[2.0, 2.1], [2.2, 2.0], [1.9, 2.0]],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 1], dtype=int)
    dataset = ggml_ot.from_numpy(
        supports=supports,
        distribution_labels=labels,
        identical_supports=False,
    )

    out = dataset.fit_gmm(
        k_comps=1,
        component_sharing="auto",
        refit="full",
        gmm_key="gmm_generic",
    )
    assert out is dataset
    assert out.identical_supports is False
    assert isinstance(out.covariances, torch.Tensor) and out.covariances.numel() > 0
    assert isinstance(out.weights, torch.Tensor) and out.weights.numel() > 0


def test_fit_gmm_generic_preserves_distribution_labels():
    # GMM_TEST_KEEP: fitting must preserve existing distribution label mapping.
    supports = np.array(
        [
            [[0.1, 0.2], [0.2, 0.1], [0.0, 0.1]],
            [[1.0, 1.1], [0.9, 1.0], [1.1, 0.9]],
            [[2.0, 2.1], [2.2, 2.0], [1.9, 2.0]],
        ],
        dtype=np.float64,
    )
    labels = np.array([5, 5, 10], dtype=int)
    dataset = ggml_ot.from_numpy(
        supports=supports,
        distribution_labels=labels,
        identical_supports=False,
    )

    transformed = dataset.fit_gmm(
        k_comps=1,
        component_sharing="auto",
        refit="full",
    )

    assert np.array_equal(np.asarray(transformed.distribution_labels), labels)


def test_roundtrip_anndata_first_vs_dataset_first_global_shapes():
    # GMM_TEMP_REFACTOR: TODO: remove before v1. Parity check for annadata-first vs dataset-first global conversion.
    adata_a = _tiny_adata()
    adata_b = _tiny_adata()

    _fit_anndata_via_dataset(
        adata_a,
        k_comps=1,
        component_sharing="global",
        distribution_col="sample",
        gmm_key="gmm_roundtrip_a",
    )
    anndata_first = ggml_ot.from_anndata(
        adata_a,
        patient_col="sample",
        label_col="patient_group",
        gmm_key="gmm_roundtrip_a",
    )

    empirical = ggml_ot.from_anndata(
        adata_b,
        patient_col="sample",
        label_col="patient_group",
        n_cells=2,
    )
    dataset_first = ggml_ot.gmm.fit_gmm(
        empirical,
        k_comps=1,
        component_sharing="global",
        refit="full",
        gmm_key="gmm_roundtrip_b",
    )

    assert anndata_first.identical_supports is True
    assert dataset_first.identical_supports is True
    assert tuple(anndata_first.supports.shape) == tuple(dataset_first.supports.shape)
    assert tuple(anndata_first.covariances.shape) == tuple(dataset_first.covariances.shape)
    assert tuple(anndata_first.weights.shape) == tuple(dataset_first.weights.shape)
    assert torch.allclose(dataset_first.weights.sum(dim=1), torch.ones(dataset_first.weights.shape[0]), atol=1e-5)


def test_k_comps_iterable_global_refit_full_uses_all_cells():
    # GMM_TEMP_REFACTOR: TODO: remove before v1. Internal metadata check for full refit behavior.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        component_sharing="global",
        refit="full",
        gmm_key="gmm_sel",
    )

    cfg = adata.uns["gmm_sel"]
    selection = cfg["selection"]
    assert selection["best_k"] in {1, 2}
    assert len(selection["scores"]["k"]) == 2
    assert selection["refit"] == "full"
    assert cfg["backend_metadata"]["fit_indices_count"] == adata.n_obs


def test_k_comps_iterable_global_refit_none_reuses_selection_subset():
    # GMM_TEMP_REFACTOR: TODO: remove before v1. Internal metadata check for refit='none' behavior.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        component_sharing="global",
        train_size=0.5,
        refit="none",
        gmm_key="gmm_sel_none",
    )

    cfg = adata.uns["gmm_sel_none"]
    selection = cfg["selection"]
    assert selection["refit"] == "none"
    assert cfg["backend_metadata"]["fit_indices_count"] == selection["selection_cells_count"]
    assert cfg["backend_metadata"]["fit_indices_count"] < adata.n_obs


def test_k_comps_iterable_sample_specific_grouped_selection_keeps_all_distributions():
    # GMM_TEMP_REFACTOR: TODO: remove before v1. Grouped selection smoke for sample-specific branch.
    adata = _separable_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        component_sharing="sample_specific",
        refit="none",
        gmm_key="gmm_sel_ss",
    )

    cfg = adata.uns["gmm_sel_ss"]
    assert cfg["selection"]["selection_cells_count"] >= 3
    assert cfg["selection"]["refit"] == "none"
    assert "best_k_by_distribution" in cfg["selection"]
    assert len(cfg["selection"]["best_k_by_distribution"]) == 3


def test_fixed_k_coerces_refit_none_to_full():
    # GMM_TEST_KEEP: fixed-k fits ignore refit='none' and use the fixed-k path.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=2,
        component_sharing="global",
        refit="none",
        gmm_key="gmm_fixed_refit",
    )

    cfg = adata.uns["gmm_fixed_refit"]
    assert cfg["selection"] is None
    assert cfg["backend_metadata"]["refit"] == "full"


def test_k_comps_candidate_list_is_respected():
    # GMM_TEMP_REFACTOR: TODO: remove before v1. Candidate list traceability during model-selection development.
    adata = _separable_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 3],
        component_sharing="global",
        refit="none",
        gmm_key="gmm_k_comps_values",
    )

    cfg = adata.uns["gmm_k_comps_values"]
    selection = cfg["selection"]
    observed_ks = [int(k) for k in selection["scores"]["k"]]
    assert observed_ks == [1, 3]
    assert selection["k_comps"] == [1, 3]


def test_selection_metadata_roundtrips_in_schema():
    # GMM_TEST_KEEP: selection metadata must persist in schema.
    adata = _tiny_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        component_sharing="global",
        gmm_key="gmm_sel_roundtrip",
    )

    cfg = adata.uns["gmm_sel_roundtrip"]
    selection = cfg["selection"]
    assert selection is not None
    assert selection["best_k"] in {1, 2}
    assert selection["k_comps"] == [1, 2]
    assert len(selection["scores"]["k"]) == 2


def test_heldout_nll_selection_metadata_stores_train_validation_diagnostics():
    # GMM_TEST_KEEP: heldout_nll selection must persist the fit-time train/validation diagnostics.
    adata = _separable_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        k_selection_metric="heldout_nll",
        component_sharing="sample_specific",
        train_size=0.8,
        refit="none",
        gmm_key="gmm_holdout_schema",
    )

    selection = adata.uns["gmm_holdout_schema"]["selection"]
    assert selection["metric"] == "heldout_nll"
    assert selection["train_frac"] == pytest.approx(0.8)
    assert selection["val_frac"] == pytest.approx(0.2)
    assert selection["score_name"] == "validation_nll"
    assert set(selection["scores"].keys()) >= {
        "k",
        "score",
        "train_nll",
        "validation_nll",
        "nll_gap",
        "n_train_cells",
        "n_val_cells",
    }
    assert set(selection["scores_by_distribution"]["s1"].keys()) >= {
        "k",
        "score",
        "train_nll",
        "validation_nll",
        "nll_gap",
        "n_train_cells",
        "n_val_cells",
    }
    assert selection["scores_by_distribution"]["s1"]["n_train_cells"] == [64, 64]
    assert selection["scores_by_distribution"]["s1"]["n_val_cells"] == [16, 16]


def test_evaluate_holdout_nll_returns_patient_level_diagnostics():
    # GMM_TEST_KEEP: hold-out NLL diagnostics should return one finite row per distribution.
    adata = _separable_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        k_selection_metric="heldout_nll",
        component_sharing="sample_specific",
        train_size=0.8,
        refit="none",
        gmm_key="gmm_holdout_eval",
    )

    results = ggml_ot.gmm.evaluate_holdout_nll(
        adata,
        gmm_key="gmm_holdout_eval",
        patient_col="sample",
        label_col="patient_group",
    )

    assert results["distribution_id"].tolist() == ["s1", "s2", "s3"]
    assert results["label"].tolist() == ["A", "A", "A"]
    assert results["n_train_cells"].tolist() == [64, 64, 64]
    assert results["n_val_cells"].tolist() == [16, 16, 16]
    assert np.isfinite(results["train_nll"]).all()
    assert np.isfinite(results["validation_nll"]).all()
    assert np.isfinite(results["nll_gap"]).all()
    assert np.allclose(results["train_frac"].to_numpy(dtype=float), 0.8)
    assert np.allclose(results["val_frac"].to_numpy(dtype=float), 0.2)


def test_summarize_holdout_nll_groups_rows():
    # GMM_TEST_KEEP: grouped NLL summaries should produce flat user-facing columns.
    results = pd.DataFrame(
        {
            "dataset": ["A", "A", "B"],
            "train_nll": [1.0, 2.0, 3.0],
            "validation_nll": [1.2, 2.3, 3.4],
            "nll_gap": [0.2, 0.3, 0.4],
        }
    )

    summary = ggml_ot.gmm.summarize_holdout_nll(results, groupby_cols=["dataset"])

    assert summary.index.tolist() == ["A", "B"]
    assert summary.loc["A", "count"] == 2
    assert summary.loc["A", "mean_train_nll"] == pytest.approx(1.5)
    assert summary.loc["A", "sd_validation_nll"] == pytest.approx(np.std([1.2, 2.3], ddof=1))


def test_validate_gmm_dataset_wrapper_returns_report(monkeypatch):
    # GMM_TEST_KEEP: dataset.validate_gmm should bundle results, summary, and plot handle.
    adata = _separable_adata()
    dataset = ggml_ot.from_anndata(
        adata,
        patient_col="sample",
        label_col="patient_group",
        n_cells=80,
    )
    dataset.fit_gmm(
        k_comps=[1, 2],
        k_selection_metric="heldout_nll",
        component_sharing="sample_specific",
        train_size=0.8,
        refit="none",
        gmm_key="gmm_holdout_validate",
    )

    from ggml_ot.plot import eval as eval_mod

    calls = {}

    def _fake_boxplot(results, **kwargs):
        calls["results"] = pd.DataFrame(results)
        calls["kwargs"] = dict(kwargs)
        return "AX"

    monkeypatch.setattr(eval_mod, "gmm_fit_validation_boxplot", _fake_boxplot)

    report = dataset.validate_gmm(
        gmm_key="gmm_holdout_validate",
        plot_kwargs={"show": False},
    )

    assert report.ax == "AX"
    assert report.results["distribution_id"].tolist() == ["s1", "s2", "s3"]
    assert report.summary.loc["All", "count"] == 3
    assert calls["results"]["distribution_id"].tolist() == ["s1", "s2", "s3"]
    assert calls["kwargs"]["show"] is False


def test_validate_gmm_public_api_accepts_raw_anndata():
    # GMM_TEST_KEEP: ggml_ot.gmm.validate_gmm should work directly on fitted AnnData objects.
    adata = _separable_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        k_selection_metric="heldout_nll",
        component_sharing="sample_specific",
        train_size=0.8,
        refit="none",
        gmm_key="gmm_holdout_alias",
    )

    report = ggml_ot.gmm.validate_gmm(
        adata,
        gmm_key="gmm_holdout_alias",
        patient_col="sample",
        label_col="patient_group",
        plot=False,
    )

    assert report.ax is None
    assert report.results["distribution_id"].tolist() == ["s1", "s2", "s3"]
    assert report.summary.loc["All", "count"] == 3


def test_split_train_val_indices_raises_when_train_split_is_too_small():
    # GMM_TEST_KEEP: invalid held-out splits must fail loudly instead of reusing the same cells.
    from ggml_ot._utils._splits import split_train_val_indices

    with pytest.raises(ValueError, match="Training split is too small"):
        split_train_val_indices(
            4,
            n_components=4,
            train_frac=0.75,
            rng=np.random.default_rng(0),
        )


def test_heldout_nll_defaults_to_balanced_split():
    # GMM_TEST_KEEP: heldout_nll should default to an explicit balanced split.
    adata = _separable_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        k_selection_metric="heldout_nll",
        component_sharing="sample_specific",
        refit="none",
        gmm_key="gmm_holdout_default_train_frac",
    )

    selection = adata.uns["gmm_holdout_default_train_frac"]["selection"]
    assert selection["train_frac"] == pytest.approx(0.5)
    assert selection["val_frac"] == pytest.approx(0.5)


def test_sample_specific_iterable_k_persists_distribution_n_components_and_padding():
    # GMM_TEST_KEEP: sample-specific iterable-K should select per-distribution K and pad to K_max.
    adata = _heterogeneous_k_adata()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2],
        k_selection_metric="bic",
        component_sharing="sample_specific",
        refit="none",
        train_size=0.5,
        gmm_key="gmm_hetero_k",
    )

    cfg = adata.uns["gmm_hetero_k"]
    dist_ks = np.asarray(cfg["distribution_n_components"], dtype=int)
    assert dist_ks.shape == (3,)
    assert set(dist_ks.tolist()) <= {1, 2}
    assert np.any(dist_ks == 1)
    assert np.any(dist_ks == 2)
    assert int(cfg["n_components"]) == int(np.max(dist_ks))

    weights = np.asarray(cfg["distribution_weights"], dtype=np.float64)
    assert weights.shape[1] == int(np.max(dist_ks))
    for i, k_i in enumerate(dist_ks.tolist()):
        if k_i < weights.shape[1]:
            assert np.allclose(weights[i, k_i:], 0.0)

    dataset = ggml_ot.from_anndata(adata, gmm_key="gmm_hetero_k")
    assert dataset.weights.shape[1] == int(np.max(dist_ks))


def test_k_selection_runtime_smoke_native():
    # GMM_TEMP_REFACTOR: TODO: remove before v1. Runtime smoke guard for native selection implementation.
    rng = np.random.default_rng(0)
    x1 = rng.normal(loc=0.0, scale=0.2, size=(300, 5))
    x2 = rng.normal(loc=3.0, scale=0.2, size=(300, 5))
    x = np.vstack([x1, x2]).astype(np.float64)
    obs = pd.DataFrame(
        {
            "sample": np.array(["s1"] * 200 + ["s2"] * 200 + ["s3"] * 200),
            "patient_group": np.array(["A"] * 600),
        }
    )
    obs.index = obs.index.astype(str)
    adata = AnnData(X=x, obs=obs)

    t0 = time.perf_counter()
    _fit_anndata_via_dataset(
        adata,
        k_comps=[1, 2, 3, 4],
        component_sharing="global",
        refit="full",
        gmm_key="gmm_runtime",
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0


@pytest.mark.parametrize("k_selection_metric", ["bic", "aic", "heldout_nll"])
def test_from_synth_sample_specific_selects_k3_for_all_metrics(k_selection_metric):
    # GMM_TEMP_REFACTOR: TODO: remove before v1. Synthetic benchmark expectation used to tune model-selection behavior.
    old_seed = ggml_ot.settings.random_seed
    try:
        ggml_ot.settings.random_seed = 0
        np.random.seed(0)
        dataset = ggml_ot.data.from_synth(
            distribution_size=120,
            class_means=[5, 10, 15],
            offsets=[1.5, 4.5, 7.5],
            shared_means_x=[-40, 40],
            shared_means_y=[-40, 40],
            varying_size=False,
            noise_scale=0.5,
            noise_dims=1,
            show=False,
            t=1,
        )

        transformed = dataset.fit_gmm(
            component_sharing="sample_specific",
            k_comps=[1, 2, 3, 4, 5],
            k_selection_metric=k_selection_metric,
        )
        selected_k = int(transformed.weights.shape[1])
        if k_selection_metric in {"aic", "heldout_nll"}:
            # With per-distribution selection, K_max can drift upward slightly.
            assert selected_k in {3, 4, 5}
        else:
            assert selected_k in {3, 4}
    finally:
        ggml_ot.settings.random_seed = old_seed


# ---------------------------------------------------------------------------
# Singularity handling
# ---------------------------------------------------------------------------


def _make_near_singular_var(*, n_components: int = 2, n_features: int = 3, eps: float = 1e-4) -> np.ndarray:
    """Build a (1, K, D, D) covariance block where one component is near-singular."""
    rng = np.random.default_rng(42)
    mats = []
    for k in range(n_components):
        a = rng.standard_normal((n_features, n_features))
        cov = a @ a.T / n_features + np.eye(n_features)
        if k == 0:
            # Force one eigenvalue well below eps to guarantee clamping.
            cov[0, :] = 0.0
            cov[:, 0] = 0.0
            cov[0, 0] = eps * 1e-3
        mats.append(cov)
    return np.stack(mats, axis=0).reshape(1, n_components, n_features, n_features)


@pytest.mark.parametrize("mode", ["guarded", "robust"])
def test_apply_singularity_handling_clamp_modes_fix_silently(mode):
    # GMM_TEST_KEEP: guarded and robust must silently clamp near-singular eigenvalues when
    # clamping succeeds (no warning, no raise). Warning/raise only fires on residual failure.
    import warnings

    from ggml_ot._utils._covariance import apply_singularity_handling

    eps = 1e-4
    var = _make_near_singular_var(eps=eps)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sanitized = apply_singularity_handling(
            var,
            covariance_type="full",
            eps=eps,
            singularity_handling=mode,
            n_components=2,
        )

    # All eigenvalues must be >= eps after clamping.
    mats = sanitized.reshape(-1, var.shape[-2], var.shape[-1])
    for cov in mats:
        assert float(np.linalg.eigvalsh(cov).min()) >= eps * 0.99


def test_apply_singularity_handling_strict_raises_on_near_singular():
    # GMM_TEST_KEEP: strict mode must raise ValueError immediately on near-singular input.
    from ggml_ot._utils._covariance import apply_singularity_handling

    eps = 1e-4
    var = _make_near_singular_var(eps=eps)

    with pytest.raises(ValueError, match="near-singular"):
        apply_singularity_handling(
            var,
            covariance_type="full",
            eps=eps,
            singularity_handling="strict",
            n_components=2,
        )


@pytest.mark.parametrize("mode", ["guarded", "robust", "strict"])
def test_apply_singularity_handling_no_op_on_spd(mode):
    # GMM_TEST_KEEP: all modes must be silent and return unchanged covariances for well-conditioned input.
    import warnings

    from ggml_ot._utils._covariance import apply_singularity_handling

    eps = 1e-4
    # Build clearly SPD covariances (identity scaled up, well above eps).
    var = (np.eye(3) * 0.5).reshape(1, 1, 3, 3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sanitized = apply_singularity_handling(
            var,
            covariance_type="full",
            eps=eps,
            singularity_handling=mode,
            n_components=1,
        )

    assert np.allclose(sanitized, var)


def test_apply_singularity_handling_diag_passthrough():
    # GMM_TEST_KEEP: diagonal covariances must be returned unchanged.
    from ggml_ot._utils._covariance import apply_singularity_handling

    var = np.ones((1, 2, 3))  # diag storage: (1, K, D)
    result = apply_singularity_handling(
        var,
        covariance_type="diag",
        eps=1e-4,
        singularity_handling="robust",
        n_components=2,
    )
    assert result is var


def test_gmmfitconfig_singularity_handling_field():
    # GMM_TEST_KEEP: GMMFitConfig must expose singularity_handling with all three modes and default 'guarded'.
    from ggml_ot.gmm._fit_core import GMMFitConfig

    assert GMMFitConfig(n_components=3).singularity_handling == "guarded"
    assert GMMFitConfig(n_components=3, singularity_handling="robust").singularity_handling == "robust"
    assert GMMFitConfig(n_components=3, singularity_handling="strict").singularity_handling == "strict"
