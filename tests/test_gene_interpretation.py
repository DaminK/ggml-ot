"""Tests for gene interpretation: axis ranking, GMM summaries, and enrichment gene scores.

Covers Phases 1-3 of the gene enrichment plan.
Does NOT test decoupler integration (requires network access + optional dep).
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import ggml_ot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adata_with_ggml():
    """Minimal AnnData with GGML loadings plus structured cell annotations."""
    rng = np.random.default_rng(42)
    n_patients = 5
    # n_cells_per_patient = 18
    # n_cells = n_patients * n_cells_per_patient
    n_genes = 50
    n_axes = 3
    cell_type_names = np.array(["T_cell", "B_cell", "Monocyte"], dtype=object)

    cluster_means = np.zeros((3, n_genes), dtype=np.float32)
    cluster_means[0, :10] = 2.0
    cluster_means[1, 10:20] = -2.0
    cluster_means[2, 20:30] = 1.5
    cluster_means[2, 30:40] = -1.5

    xs = []
    patient_ids = []
    patient_groups = []
    cell_types = []
    patient_group_labels = np.array(["control", "control", "disease", "disease", "disease"], dtype=object)

    for patient_idx in range(n_patients):
        patient_id = f"patient_{patient_idx}"
        patient_offset = np.zeros(n_genes, dtype=np.float32)
        patient_offset[40:] = np.float32((patient_idx - 2) * 0.15)
        for cluster_idx, cell_type in enumerate(cell_type_names):
            x_block = cluster_means[cluster_idx] + patient_offset
            x_block = x_block + 0.30 * rng.normal(size=(6, n_genes)).astype(np.float32)
            xs.append(x_block)
            patient_ids.extend([patient_id] * x_block.shape[0])
            patient_groups.extend([patient_group_labels[patient_idx]] * x_block.shape[0])
            cell_types.extend([cell_type] * x_block.shape[0])

    adata = AnnData(X=np.vstack(xs).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.var["feature_name"] = [f"Gene{i}" for i in range(n_genes)]
    adata.obs["sample"] = np.asarray(patient_ids, dtype=object)
    adata.obs["patient_group"] = np.asarray(patient_groups, dtype=object)
    adata.obs["cell_type"] = np.asarray(cell_types, dtype=object)

    W = rng.normal(size=(n_genes, n_axes)).astype(np.float64)
    adata.varm["W_ggml"] = W
    adata.obsm["X_ggml"] = adata.X @ W

    return adata


def _fit_gene_test_gmm(*, component_sharing: str) -> tuple[AnnData, str]:
    adata = _make_adata_with_ggml()
    min_cells = int(adata.obs.groupby("sample").size().min())
    dataset = ggml_ot.from_anndata(
        adata,
        patient_col="sample",
        label_col="patient_group",
        use_rep=None,
        n_cells=min_cells,
    )
    dataset.fit_gmm(
        component_sharing=component_sharing,
        k_comps=3,
        covariance_type="diag",
        max_iter=40,
        n_init=1,
        gmm_key="gmm_X",
        verbose=False,
    )
    return dataset.adata, "gmm_X"


@pytest.fixture
def adata_with_ggml():
    return _make_adata_with_ggml()


@pytest.fixture(scope="module")
def _global_gmm_payload():
    return _fit_gene_test_gmm(component_sharing="global")


@pytest.fixture
def adata_with_global_gmm(_global_gmm_payload):
    adata, gmm_key = _global_gmm_payload
    return adata.copy(), gmm_key


# ---------------------------------------------------------------------------
# Phase 2: Axis ranking tests
# ---------------------------------------------------------------------------


class TestRankLatentAxes:
    def test_returns_dataframe(self, adata_with_ggml):
        from ggml_ot.gene._axis import rank_latent_axes

        result = rank_latent_axes(adata_with_ggml)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, adata_with_ggml):
        from ggml_ot.gene._axis import rank_latent_axes

        result = rank_latent_axes(adata_with_ggml)
        assert set(result.columns) == {"axis", "gene", "score", "abs_score", "sign", "rank"}

    def test_all_axes_ranked(self, adata_with_ggml):
        from ggml_ot.gene._axis import rank_latent_axes

        result = rank_latent_axes(adata_with_ggml)
        n_axes = adata_with_ggml.varm["W_ggml"].shape[1]
        n_genes = adata_with_ggml.n_vars
        assert len(result) == n_axes * n_genes

    def test_subset_axes(self, adata_with_ggml):
        from ggml_ot.gene._axis import rank_latent_axes

        result = rank_latent_axes(adata_with_ggml, axes=[0, 2])
        assert set(result["axis"].unique()) == {0, 2}

    def test_ranked_by_abs_score(self, adata_with_ggml):
        from ggml_ot.gene._axis import rank_latent_axes

        result = rank_latent_axes(adata_with_ggml)
        for ax in result["axis"].unique():
            ax_df = result[result["axis"] == ax]
            assert ax_df["abs_score"].is_monotonic_decreasing

    def test_signed_scores_preserved(self, adata_with_ggml):
        from ggml_ot.gene._axis import rank_latent_axes

        result = rank_latent_axes(adata_with_ggml)
        # Should have both positive and negative scores
        assert (result["sign"] == 1).any()
        assert (result["sign"] == -1).any()

    def test_gene_symbols(self, adata_with_ggml):
        from ggml_ot.gene._axis import rank_latent_axes

        result = rank_latent_axes(adata_with_ggml, gene_symbols="feature_name")
        assert result["gene"].iloc[0].startswith("Gene")

    def test_missing_W_ggml_raises(self):
        from ggml_ot.gene._axis import rank_latent_axes

        adata = AnnData(X=np.zeros((3, 4)))
        with pytest.raises(ValueError, match="W_ggml"):
            rank_latent_axes(adata)

    def test_invalid_axis_raises(self, adata_with_ggml):
        from ggml_ot.gene._axis import rank_latent_axes

        with pytest.raises(ValueError, match="out of range"):
            rank_latent_axes(adata_with_ggml, axes=[99])


# ---------------------------------------------------------------------------
# Phase 3: GMM summary tests
# ---------------------------------------------------------------------------


class TestResolveGmmKey:
    def test_auto_resolve(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import resolve_gmm_key

        adata, expected_key = adata_with_global_gmm
        assert resolve_gmm_key(adata, None, None) == expected_key

    def test_explicit_key(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import resolve_gmm_key

        adata, gmm_key = adata_with_global_gmm
        assert resolve_gmm_key(adata, gmm_key, None) == gmm_key

    def test_missing_key_raises(self, adata_with_ggml):
        from ggml_ot.gene._gmm_summary import resolve_gmm_key

        with pytest.raises(KeyError, match="not found"):
            resolve_gmm_key(adata_with_ggml, "nonexistent", None)


class TestRequireGlobalOrGrouped:
    def test_global_passes(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import require_global_or_grouped

        adata, gmm_key = adata_with_global_gmm
        require_global_or_grouped(adata, gmm_key, None)  # should not raise

    def test_sample_specific_no_aggregate_raises(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import require_global_or_grouped

        adata, gmm_key = adata_with_global_gmm
        adata.uns[gmm_key]["component_sharing"] = "sample_specific"
        with pytest.raises(ValueError, match="sample-specific"):
            require_global_or_grouped(adata, gmm_key, None)

    def test_sample_specific_with_aggregate_passes(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import require_global_or_grouped

        adata, gmm_key = adata_with_global_gmm
        adata.uns[gmm_key]["component_sharing"] = "sample_specific"
        require_global_or_grouped(adata, gmm_key, "mean")  # should not raise


class TestSummarizeGmmComponents:
    def test_returns_dict_with_expected_keys(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_global_gmm
        result = summarize_gmm_components(adata, gmm_key, groupby="cell_type")
        assert set(result.keys()) == {"celltype_table", "patient_weights", "purity"}

    def test_celltype_table_shape(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_global_gmm
        result = summarize_gmm_components(adata, gmm_key, groupby="cell_type")
        ct = result["celltype_table"]
        n_components = adata.uns[gmm_key]["n_components"]
        n_types = len(adata.obs["cell_type"].unique())
        assert ct.shape == (n_components, n_types)

    def test_component_normalization(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_global_gmm
        result = summarize_gmm_components(adata, gmm_key, groupby="cell_type", normalize="component")
        row_sums = result["celltype_table"].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_hard_weighting(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_global_gmm
        result = summarize_gmm_components(adata, gmm_key, groupby="cell_type", weighting="hard")
        assert isinstance(result["celltype_table"], pd.DataFrame)

    def test_patient_weights_shape(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_global_gmm
        result = summarize_gmm_components(adata, gmm_key, groupby="cell_type")
        pw = result["patient_weights"]
        n_patients = len(adata.uns[gmm_key]["weight_inference"]["distribution_ids"])
        n_components = adata.uns[gmm_key]["n_components"]
        assert pw.shape == (n_patients, n_components)

    def test_purity_has_expected_columns(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_global_gmm
        result = summarize_gmm_components(adata, gmm_key, groupby="cell_type")
        assert set(result["purity"].columns) == {"dominant_type", "purity", "entropy"}

    def test_missing_groupby_raises(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_global_gmm
        with pytest.raises(KeyError, match="nonexistent"):
            summarize_gmm_components(adata, gmm_key, groupby="nonexistent")

    def test_sample_specific_raises(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_global_gmm
        adata.uns[gmm_key]["component_sharing"] = "sample_specific"
        with pytest.raises(ValueError, match="sample-specific"):
            summarize_gmm_components(adata, gmm_key, groupby="cell_type")


# ---------------------------------------------------------------------------
# Phase 3: Component gene scores
# ---------------------------------------------------------------------------


class TestComponentGeneScores:
    def test_returns_dataframe(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_global_gmm
        result = component_gene_scores(adata, gmm_key)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_global_gmm
        result = component_gene_scores(adata, gmm_key)
        assert set(result.columns) == {"component", "gene", "score", "abs_score", "rank"}

    def test_all_components_present(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_global_gmm
        result = component_gene_scores(adata, gmm_key)
        n_components = adata.uns[gmm_key]["n_components"]
        assert set(result["component"].unique()) == set(range(n_components))

    def test_correct_total_rows(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_global_gmm
        result = component_gene_scores(adata, gmm_key)
        n_components = adata.uns[gmm_key]["n_components"]
        n_genes = adata.n_vars
        assert len(result) == n_components * n_genes

    def test_reference_rest_vs_global_mean_differ(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_global_gmm
        rest = component_gene_scores(adata, gmm_key, reference="rest")
        gm = component_gene_scores(adata, gmm_key, reference="global_mean")
        # Scores should differ (different reference baselines)
        assert not np.allclose(rest["score"].values, gm["score"].values)

    def test_unknown_contrast_raises(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_global_gmm
        with pytest.raises(ValueError, match="Unknown contrast"):
            component_gene_scores(adata, gmm_key, contrast="unknown")

    def test_sample_specific_raises(self, adata_with_global_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_global_gmm
        adata.uns[gmm_key]["component_sharing"] = "sample_specific"
        with pytest.raises(ValueError, match="sample-specific"):
            component_gene_scores(adata, gmm_key)

    def test_scores_have_both_signs(self, adata_with_global_gmm):
        """Gene scores can be negative (sign is arbitrary in latent space)."""
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_global_gmm
        result = component_gene_scores(adata, gmm_key)
        assert (result["score"] > 0).any()
        assert (result["score"] < 0).any()


# ---------------------------------------------------------------------------
# Phase 4: Component grouping for sample-specific GMMs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _sample_specific_gmm_payload():
    return _fit_gene_test_gmm(component_sharing="sample_specific")


@pytest.fixture
def adata_with_sample_specific_gmm(_sample_specific_gmm_payload):
    adata, gmm_key = _sample_specific_gmm_payload
    return adata.copy(), gmm_key


class TestGroupComponents:
    def test_mean_grouping_returns_expected_keys(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = group_components(adata, gmm_key, grouping_method="mean")
        assert set(result.keys()) == {
            "label_matrix",
            "distribution_ids",
            "grouped_mu",
            "grouped_var",
            "grouped_weights",
            "grouping_method",
            "group_representative",
            "barycenter_weighting",
            "n_groups",
            "source_gmm_key",
            "source_checksum",
            "grouping_key",
        }

    def test_label_matrix_shape(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = group_components(adata, gmm_key, grouping_method="mean")
        assert result["label_matrix"].shape == tuple(np.asarray(adata.uns[gmm_key]["model"]["mu"]).shape[:2])
        assert (result["label_matrix"] >= 0).all()
        assert result["grouped_var"] is None

    def test_grouped_weights_shape(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = group_components(adata, gmm_key, grouping_method="mean", n_groups=3)
        assert result["grouped_weights"].shape == (5, 3)

    def test_grouped_mu_shape(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = group_components(adata, gmm_key, grouping_method="mean", n_groups=3)
        assert result["grouped_mu"].shape == (3, adata.n_vars)

    def test_bures_grouping(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = group_components(
            adata,
            gmm_key,
            grouping_method="bures-wasserstein",
            group_representative="gaussian",
            n_groups=3,
        )
        assert result["grouping_method"] == "bures-wasserstein"
        assert result["grouped_var"].shape == (3, adata.n_vars, adata.n_vars)

    def test_global_gmm_raises(self, adata_with_global_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_global_gmm
        with pytest.raises(ValueError, match="sample_specific"):
            group_components(adata, gmm_key, grouping_method="mean")

    def test_accepts_dataset_input(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        dataset = ggml_ot.from_anndata(adata.copy(), gmm_key=gmm_key)
        result = group_components(dataset, gmm_key, grouping_method="mean", n_groups=3)
        assert result["grouped_weights"].shape == (len(result["distribution_ids"]), 3)

    def test_grouping_key_override_persists_result(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = group_components(
            adata,
            gmm_key,
            grouping_method="mean",
            n_groups=3,
            grouping_key="custom_grouping_key",
        )
        assert result["grouping_key"] == "custom_grouping_key"
        assert "custom_grouping_key" in adata.uns

    def test_inactive_padded_components_marked_as_minus_one(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        adata.uns[gmm_key]["distribution_n_components"] = [3, 2, 1, 3, 2]
        weights = np.asarray(adata.uns[gmm_key]["distribution_weights"], dtype=np.float64)
        for dist_idx, k_i in enumerate(adata.uns[gmm_key]["distribution_n_components"]):
            if k_i < weights.shape[1]:
                weights[dist_idx, k_i:] = 0.0
        adata.uns[gmm_key]["distribution_weights"] = weights

        result = group_components(adata, gmm_key, grouping_method="mean")
        label_matrix = result["label_matrix"]
        assert np.all(label_matrix[1, 2:] == -1)
        assert np.all(label_matrix[2, 1:] == -1)
        assert np.all(label_matrix[4, 2:] == -1)

    def test_grouped_weights_rows_sum_to_one(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = group_components(adata, gmm_key, grouping_method="mean", n_groups=3)
        np.testing.assert_allclose(result["grouped_weights"].sum(axis=1), 1.0, atol=1e-10)

    def test_uniform_weighting_affects_mean_representative(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result_cw = group_components(
            adata.copy(),
            gmm_key,
            grouping_method="mean",
            n_groups=2,
            barycenter_weighting="component_weights",
        )
        result_uni = group_components(
            adata.copy(),
            gmm_key,
            grouping_method="mean",
            n_groups=2,
            barycenter_weighting="uniform",
        )
        # With heterogeneous component weights the two schemes should differ
        # (unless all weights within each group happen to be equal, which is
        # unlikely for a fitted GMM).  Allow exact equality only if the GMM
        # happens to have perfectly uniform weights.
        weights = np.asarray(adata.uns[gmm_key]["distribution_weights"], dtype=np.float64)
        if not np.allclose(weights, weights.mean()):
            assert not np.allclose(result_cw["grouped_mu"], result_uni["grouped_mu"])

    def test_canonicalize_group_order_tiebreaker(self):
        """Equal-mass groups are ordered by ascending mean coordinate (dim 0 first)."""
        from ggml_ot.gene._grouping import _canonicalize_group_order

        # Two groups with identical total mass but different mean positions.
        label_matrix = np.array([[0, 1]])  # 1 distribution, 2 components
        grouped_mu = np.array([[5.0, 0.0], [1.0, 0.0]])  # group 0 at 5, group 1 at 1
        grouped_weights = np.array([[0.5, 0.5]])  # equal mass

        new_labels, new_mu, _, new_weights = _canonicalize_group_order(
            label_matrix=label_matrix,
            grouped_mu=grouped_mu,
            grouped_var=None,
            grouped_weights=grouped_weights,
        )
        # After reordering: group with smaller mean[0] should come first.
        assert new_mu[0, 0] < new_mu[1, 0]
        # label_matrix should be remapped consistently.
        np.testing.assert_array_equal(new_weights.sum(axis=1), grouped_weights.sum(axis=1))


class TestSampleSpecificSummary:
    def test_no_grouping_raises(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_sample_specific_gmm
        with pytest.raises(ValueError, match="sample-specific"):
            summarize_gmm_components(adata, gmm_key, groupby="cell_type")

    def test_grouping_mean_works(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = summarize_gmm_components(adata, gmm_key, groupby="cell_type", grouping_method="mean")
        assert "celltype_table" in result
        assert "grouping" in result

    def test_grouping_bures_works(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = summarize_gmm_components(
            adata,
            gmm_key,
            groupby="cell_type",
            grouping_method="bures-wasserstein",
            n_groups=3,
        )
        assert result["celltype_table"].shape[0] == 3

    def test_grouped_patient_weights_sum_to_one(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_sample_specific_gmm
        result = summarize_gmm_components(adata, gmm_key, groupby="cell_type", grouping_method="mean")
        row_sums = result["patient_weights"].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_stale_grouping_warns_without_overwriting(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._grouping import group_components
        from ggml_ot.gene._gmm_summary import summarize_gmm_components

        adata, gmm_key = adata_with_sample_specific_gmm
        grouping = group_components(
            adata,
            gmm_key,
            grouping_method="mean",
            n_groups=3,
        )
        original_checksum = str(adata.uns[grouping["grouping_key"]]["source_checksum"])
        adata.uns[gmm_key]["model"]["mu"] = np.asarray(adata.uns[gmm_key]["model"]["mu"]) + 1e-3

        with pytest.warns(UserWarning, match="outdated"):
            summarize_gmm_components(
                adata,
                gmm_key,
                groupby="cell_type",
                grouping_method="mean",
                n_groups=3,
                grouping_key=grouping["grouping_key"],
            )

        assert str(adata.uns[grouping["grouping_key"]]["source_checksum"]) == original_checksum


class TestSampleSpecificGeneScores:
    def test_no_grouping_raises(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_sample_specific_gmm
        with pytest.raises(ValueError, match="sample-specific"):
            component_gene_scores(adata, gmm_key)

    def test_grouping_mean_returns_scores(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_sample_specific_gmm
        result = component_gene_scores(adata, gmm_key, grouping_method="mean", n_groups=3)
        assert isinstance(result, pd.DataFrame)
        assert set(result["component"].unique()) == {0, 1, 2}

    def test_grouping_bures_returns_scores(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_sample_specific_gmm
        result = component_gene_scores(adata, gmm_key, grouping_method="bures-wasserstein", n_groups=3)
        assert len(result) == 3 * adata.n_vars

    def test_accepts_dataset_input(self, adata_with_sample_specific_gmm):
        from ggml_ot.gene._gmm_summary import component_gene_scores

        adata, gmm_key = adata_with_sample_specific_gmm
        dataset = ggml_ot.from_anndata(adata.copy(), gmm_key=gmm_key)
        result = component_gene_scores(
            dataset,
            gmm_key=gmm_key,
            grouping_method="mean",
            n_groups=3,
        )
        assert isinstance(result, pd.DataFrame)
