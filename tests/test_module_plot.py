"""Plotting smoke tests.

Scope
-----
These tests assert that plotting entry points run without raising exceptions in
headless test environments. They do not validate plot aesthetics.

Notes
-----
- We force a non-interactive matplotlib backend.
- We patch ``plt.show`` to a no-op.
"""

from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytest

import ggml_ot
from ggml_ot.data import from_synth_gmm
from ggml_ot.plot.eval import gmm_fit_validation_boxplot
from ggml_ot.plot import subspace as subspace_mod


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    """Prevent plt.show from blocking in all tests."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture()
def figdir(tmp_path):
    """Temporary figure directory for save tests."""
    d = tmp_path / "figures"
    ggml_ot.settings.figdir = str(d)
    yield d
    ggml_ot.settings.figdir = None


# -------------------------------------------------------------------
# Clustermap / embedding
# -------------------------------------------------------------------

_D = np.array(
    [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.5, 2.5],
        [2.0, 1.5, 0.0, 1.2],
        [3.0, 2.5, 1.2, 0.0],
    ],
    dtype=float,
)
_LABELS = ["a", "a", "b", "b"]


def test_clustermap_embedding_smoke():
    """Clustermap/embedding entrypoint should run without raising."""
    ggml_ot.pl.clustermap_embedding(_D, _LABELS, plot="embedding", method="mds", show=False)
    ggml_ot.pl.clustermap_embedding(_D, _LABELS, plot="clustermap", show=False)


@pytest.mark.parametrize(
    ("plot_name", "expected_prefix"),
    [
        ("clustermap", "clustermap"),
        ("embedding", "embedding"),
    ],
)
def test_plot_save(figdir, plot_name, expected_prefix):
    """Core plot entrypoints should honor save=True."""
    if plot_name == "clustermap":
        ggml_ot.pl.clustermap(_D, _LABELS, show=False, save=True)
    else:
        ggml_ot.pl.embedding(_D, _LABELS, method="mds", show=False, save=True)

    files = list(figdir.iterdir())
    assert any(f.name.startswith(expected_prefix) for f in files), f"Expected {expected_prefix} file in {files}"


# -------------------------------------------------------------------
# Distribution
# -------------------------------------------------------------------


def test_distribution_plot_smoke():
    """Distribution plotting entrypoint should run without raising."""
    rng = np.random.default_rng(0)
    distributions = [rng.normal(size=(20, 3)).astype(np.float32) for _ in range(6)]
    labels = [0, 0, 0, 1, 1, 1]

    ax = ggml_ot.pl.distribution(distributions=distributions, labels=labels, dim_red="pca", legend=False, show=False)
    assert ax is not None


# -------------------------------------------------------------------
# Evaluation plots
# -------------------------------------------------------------------


def test_gmm_fit_validation_boxplot_smoke(figdir):
    """GMM fit validation box plot should render and honor save=True."""
    ax = gmm_fit_validation_boxplot(
        {
            "train_nll": [1.0, 1.1, 1.2, 1.3],
            "validation_nll": [1.05, 1.15, 1.25, 1.35],
        },
        show=False,
        save=True,
    )

    assert ax is not None
    files = list(figdir.iterdir())
    assert any(f.name.startswith("gmm_fit_validation_boxplot") for f in files), files


# -------------------------------------------------------------------
# Embedding
# -------------------------------------------------------------------


def test_embedding_return_embedding():
    """embedding() with return_embedding=True should return (ax, emb)."""
    result = ggml_ot.pl.embedding(_D, _LABELS, method="mds", return_embedding=True, show=False)
    assert isinstance(result, tuple) and len(result) == 2
    ax, emb = result
    assert hasattr(ax, "set_title")
    assert emb.shape == (4, 2)


# -------------------------------------------------------------------
# Dataset integration
# -------------------------------------------------------------------


def test_dataset_compute_ot_save_forwards_plot_options(figdir):
    """TripletDataset.compute_OT should forward save/show to plotting API."""
    rng = np.random.default_rng(1)
    supports = [rng.normal(size=(6, 3)).astype(np.float32) for _ in range(4)]
    labels = [0, 0, 1, 1]
    dataset = ggml_ot.from_numpy(supports=supports, distribution_labels=labels)

    _ = dataset.compute_OT(ground_metric="euclidean", plot_type="clustermap", show=False, save=True)

    files = list(figdir.iterdir())
    assert len(files) > 0, f"Expected saved plot file in {figdir}, got {files}"


@pytest.fixture(scope="module")
def _latent_gmm_payloads():
    from tests.test_gene_interpretation import _fit_gene_test_gmm

    global_payload = _fit_gene_test_gmm(component_sharing="global")
    sample_payload = _fit_gene_test_gmm(component_sharing="sample_specific")
    return {
        "global": global_payload,
        "sample_specific": sample_payload,
    }


def test_latent_gmm_raw_global_smoke(figdir, _latent_gmm_payloads):
    """latent_gmm should plot a global fitted GMM in GGML space."""
    adata, gmm_key = _latent_gmm_payloads["global"]

    ax = ggml_ot.pl.latent_gmm(
        adata.copy(),
        gmm_key=gmm_key,
        component_view="raw",
        show=False,
        save=True,
    )

    assert ax is not None
    files = list(figdir.iterdir())
    assert any(f.name.startswith("latent_gmm") for f in files), files


def test_latent_gmm_grouped_sample_specific_smoke(_latent_gmm_payloads):
    """latent_gmm should plot grouped sample-specific GMMs from a dataset input."""
    adata, gmm_key = _latent_gmm_payloads["sample_specific"]
    dataset = ggml_ot.from_anndata(adata.copy(), gmm_key=gmm_key)

    ax = ggml_ot.pl.latent_gmm(
        dataset,
        gmm_key=gmm_key,
        component_view="grouped",
        grouping_method="mean",
        n_groups=3,
        show=False,
    )
    assert ax is not None


def test_latent_gmm_grouped_rejects_global_gmm(_latent_gmm_payloads):
    """Grouped latent view should reject globally shared GMMs for now."""
    adata, gmm_key = _latent_gmm_payloads["global"]

    with pytest.raises(ValueError, match="component_view='grouped'"):
        ggml_ot.pl.latent_gmm(
            adata.copy(),
            gmm_key=gmm_key,
            component_view="grouped",
            show=False,
        )


def test_latent_gmm_selected_groups_only_filters_points_and_ellipses(_latent_gmm_payloads):
    """group_display='only' should subset both visible cells and plotted grouped ellipses."""
    from ggml_ot.gene._gmm_summary import _resolve_grouping_for_consumer

    adata, gmm_key = _latent_gmm_payloads["sample_specific"]
    grouping = _resolve_grouping_for_consumer(
        adata,
        gmm_key,
        grouping_method="mean",
        n_groups=3,
        group_representative="gaussian",
        barycenter_weighting="component_weights",
        grouping_key=None,
    )
    group_ids = subspace_mod._cell_group_assignments(
        adata,
        gmm_key=gmm_key,
        label_matrix=np.asarray(grouping["label_matrix"], dtype=int),
        distribution_ids=[str(x) for x in grouping["distribution_ids"]],
        patient_col="sample",
    )
    selected_group = int(np.unique(group_ids[group_ids >= 0])[0])
    expected_cells = int(np.sum(group_ids == selected_group))

    ax = ggml_ot.pl.latent_gmm(
        adata.copy(),
        gmm_key=gmm_key,
        component_view="grouped",
        grouping_method="mean",
        n_groups=3,
        selected_groups=[selected_group],
        group_display="only",
        show=False,
    )

    assert len(ax.patches) == 1
    plotted_points = sum(collection.get_offsets().shape[0] for collection in ax.collections)
    assert plotted_points == expected_cells


def _make_mock_panel_synth_dataset(*, refit_modes: bool) -> SimpleNamespace:
    """Build a minimal synthetic-GMM-like dataset object for panel tests."""
    rng = np.random.default_rng(7)

    n_patients = 4
    n_samples_per_patient = 6
    n_dim = 4
    n_noise = 1
    n_components = 3  # 2 signal + 1 noise

    patient_ids = np.repeat(np.arange(n_patients), n_samples_per_patient)
    dist_labels = np.array([0, 0, 1, 1], dtype=int)
    point_labels = np.repeat(dist_labels, n_samples_per_patient)
    x_high = rng.normal(size=(patient_ids.shape[0], n_dim))
    clean = [rng.normal(size=(n_samples_per_patient, 2)) for _ in range(n_patients)]

    patient_gmms = []
    for pid, lbl in enumerate(dist_labels):
        means = rng.normal(size=(n_components, n_dim)) + pid
        covs = np.stack([np.diag(rng.uniform(0.3, 1.2, size=n_dim)) for _ in range(n_components)], axis=0)
        patient_gmms.append(
            {
                "patient_id": int(pid),
                "label": int(lbl),
                "means": means,
                "covs": covs,
                "weights": np.array([0.45, 0.35, 0.20], dtype=float),
            }
        )

    supports = np.stack([pg["means"] for pg in patient_gmms], axis=0)
    covariances = np.stack([pg["covs"] for pg in patient_gmms], axis=0)
    if refit_modes:
        # Simulate a refit where component ordering changed.
        supports = supports.copy()
        covariances = covariances.copy()
        supports[:, [0, 1], :] = supports[:, [1, 0], :]
        covariances[:, [0, 1], :, :] = covariances[:, [1, 0], :, :]

    synth_data = {
        "samples": {
            "clean": clean,
            "labels": point_labels.tolist(),
            "patient_ids": patient_ids.tolist(),
        },
        "Q_mixing": np.eye(n_dim, dtype=float),
        "R_rotation": np.eye(2, dtype=float),
        "X_high_all": x_high,
        "n_noise": n_noise,
        "agg_means_2d": rng.normal(size=(5, 2)),
        "agg_covs_2d": np.stack([np.eye(2) * 0.5 for _ in range(5)], axis=0),
        "patient_gmms": patient_gmms,
        "fitted_gmm_provenance": "fit_gmm" if refit_modes else "synthetic_ground_truth",
    }

    return SimpleNamespace(
        synth_data=synth_data,
        supports=supports,
        covariances=covariances,
        distribution_labels=dist_labels,
        identical_supports=False,
        _map_A=None,
    )


# TODO: remove before v1; temporary synthetic-GMM plotting semantics.
def test_panel_synth_dataset_refit_disables_fitted_mode_colors(monkeypatch):
    """Refitted per-patient GMM overlays should not use synthetic mode colours."""
    dataset = _make_mock_panel_synth_dataset(refit_modes=True)
    calls = []

    def _capture_ellipse_overlay(means, covs, ax, **kwargs):
        calls.append(
            {
                "means": np.asarray(means),
                "labels": kwargs.get("labels"),
                "palette": kwargs.get("palette"),
                "color_mode": kwargs.get("color_mode"),
                "n_signal": kwargs.get("n_signal"),
            }
        )
        return ax

    monkeypatch.setattr(subspace_mod, "ellipse_overlay", _capture_ellipse_overlay)

    fig = subspace_mod.panel_synth_dataset(
        dataset,
        fitted_gmm_view="selected_patients",
        selected_patient_ids=[0, 2],
        show_learned_panel=False,
        show=False,
    )
    if fig is not None:
        plt.close(fig)

    assert len(calls) == 2  # GT panel + fitted panel
    fitted_call = calls[1]
    assert fitted_call["labels"] is None
    assert fitted_call["palette"] is None
    assert fitted_call["color_mode"] == "none"
    assert fitted_call["n_signal"] == 0


def test_panel_synth_dataset_selected_patients_only_plots_selected_ellipses(monkeypatch):
    """selected_patients view should draw only selected patients' components in both panels."""
    dataset = _make_mock_panel_synth_dataset(refit_modes=False)
    calls = []

    def _capture_ellipse_overlay(means, covs, ax, **kwargs):
        calls.append(
            {
                "means": np.asarray(means),
                "labels": kwargs.get("labels"),
                "palette": kwargs.get("palette"),
                "color_mode": kwargs.get("color_mode"),
                "n_signal": kwargs.get("n_signal"),
            }
        )
        return ax

    monkeypatch.setattr(subspace_mod, "ellipse_overlay", _capture_ellipse_overlay)

    selected = [1]
    fig = subspace_mod.panel_synth_dataset(
        dataset,
        fitted_gmm_view="selected_patients",
        selected_patient_ids=selected,
        show_learned_panel=False,
        show=False,
    )
    if fig is not None:
        plt.close(fig)

    assert len(calls) == 2
    expected_components = len(selected) * dataset.supports.shape[1]
    panel1_means = calls[0]["means"]
    fitted_means = calls[1]["means"]
    assert panel1_means.shape[0] == expected_components
    assert fitted_means.shape[0] == expected_components
    assert calls[1]["n_signal"] == len(selected) * (dataset.supports.shape[1] - dataset.synth_data["n_noise"])


def test_panel_synth_dataset_class_average_keeps_global_clean_overlay(monkeypatch):
    """class_average view should keep the aggregate pre-jitter clean-space overlay in panel 1."""
    dataset = _make_mock_panel_synth_dataset(refit_modes=False)
    calls = []

    def _capture_ellipse_overlay(means, covs, ax, **kwargs):
        calls.append(np.asarray(means))
        return ax

    monkeypatch.setattr(subspace_mod, "ellipse_overlay", _capture_ellipse_overlay)

    fig = subspace_mod.panel_synth_dataset(
        dataset,
        fitted_gmm_view="class_average",
        show_learned_panel=False,
        show=False,
    )
    if fig is not None:
        plt.close(fig)

    assert len(calls) == 2
    assert calls[0].shape[0] == dataset.synth_data["agg_means_2d"].shape[0]


def test_panel_synth_dataset_patient_coloring_uses_selected_patient_ids(monkeypatch):
    """patient mode_coloring should colour every selected fitted component by patient id."""
    dataset = _make_mock_panel_synth_dataset(refit_modes=False)
    calls = []

    def _capture_ellipse_overlay(means, covs, ax, **kwargs):
        calls.append(
            {
                "means": np.asarray(means),
                "labels": kwargs.get("labels"),
                "palette": kwargs.get("palette"),
                "color_mode": kwargs.get("color_mode"),
                "n_signal": kwargs.get("n_signal"),
            }
        )
        return ax

    monkeypatch.setattr(subspace_mod, "ellipse_overlay", _capture_ellipse_overlay)

    selected = [1, 3]
    fig = subspace_mod.panel_synth_dataset(
        dataset,
        fitted_gmm_view="selected_patients",
        selected_patient_ids=selected,
        mode_coloring="patients",
        show_learned_panel=False,
        show=False,
    )
    if fig is not None:
        plt.close(fig)

    assert len(calls) == 2
    fitted_call = calls[1]
    assert fitted_call["color_mode"] == "all"
    assert fitted_call["palette"] is not None
    expected_labels = np.array(
        [1, 1, 3, 3] + [1, 3],
        dtype=int,
    )
    assert np.array_equal(fitted_call["labels"], expected_labels)
    assert fitted_call["n_signal"] == 4


# TODO: remove before v1; temporary synthetic-GMM plotting semantics.
def test_panel_synth_dataset_patient_coloring_uses_class_aligned_base_palette(monkeypatch):
    """Patient-coloured fitted modes should derive from the same class palette as the points."""
    dataset = _make_mock_panel_synth_dataset(refit_modes=False)
    ellipse_calls = []
    scatter_calls = []

    def _capture_ellipse_overlay(means, covs, ax, **kwargs):
        ellipse_calls.append(kwargs)
        return ax

    def _capture_scatter(*args, **kwargs):
        scatter_calls.append(kwargs)
        return kwargs["ax"]

    monkeypatch.setattr(subspace_mod, "ellipse_overlay", _capture_ellipse_overlay)
    monkeypatch.setattr(subspace_mod, "_scatter_subspace_with_patient_focus", _capture_scatter)

    fig = subspace_mod.panel_synth_dataset(
        dataset,
        fitted_gmm_view="selected_patients",
        selected_patient_ids=[0, 2],
        mode_coloring="patients",
        show_learned_panel=False,
        show=False,
    )
    if fig is not None:
        plt.close(fig)

    point_palette = scatter_calls[1]["palette"]
    fitted_palette = ellipse_calls[1]["palette"]
    assert fitted_palette[0] == point_palette[0]
    assert fitted_palette[2] == point_palette[1]


def test_panel_synth_dataset_signal_panel_undoes_global_rotation(monkeypatch):
    """Panel 2 should return to the clean signal axes, not the rotated mixed axes."""
    dataset = from_synth_gmm(
        representation="gmm",
        adata=False,
        n_dim=8,
        n_patients=4,
        n_samples=60,
        signal_mass_ratio=0.30,
        n_modes=4,
        signal_means_offset=12.0,
        signal_means_jitter=0.3,
        noise_means_offset=4.0 / 3.0,
        noise_means_jitter=0.3,
        signal_mean_shift=0.2,
        signal_cov_scale=1.0,
        signal_anisotropy=6.0,
        cov_rotation_jitter=0.0,
        cov_scale_jitter=0.0,
        global_rotation=30.0,
        random_seed=0,
    )
    calls = []

    def _capture_ellipse_overlay(means, covs, ax, **kwargs):
        calls.append(np.asarray(means))
        return ax

    monkeypatch.setattr(subspace_mod, "ellipse_overlay", _capture_ellipse_overlay)

    fig = subspace_mod.panel_synth_dataset(
        dataset,
        fitted_gmm_view="selected_patients",
        selected_patient_ids=[0],
        show_learned_panel=False,
        show=False,
    )
    if fig is not None:
        plt.close(fig)

    assert len(calls) == 2
    fitted_means = calls[1]
    signal_means = fitted_means[:2]
    off_axis = np.abs(np.array([signal_means[0, 1], signal_means[1, 0]]))
    on_axis = np.abs(np.array([signal_means[0, 0], signal_means[1, 1]]))
    assert np.all(off_axis < 1e-6)
    assert np.all(on_axis > 1.0)


# TODO: remove before v1; temporary synthetic-GMM plotting semantics.
def test_ellipse_overlay_without_labels_uses_neutral_signal_color():
    """When labels are unknown, signal ellipses should be neutral (not class-coloured)."""
    fig, ax = plt.subplots()
    means = np.array([[0.0, 0.0], [1.0, 1.0]])
    covs = np.stack([np.eye(2), np.eye(2)], axis=0)

    subspace_mod.ellipse_overlay(
        means,
        covs,
        ax,
        labels=None,
        palette={0: "#E24A33", 1: "#348ABD"},
        n_signal=2,
        diagonal_approx=False,
        annotate=False,
    )

    expected = mcolors.to_rgba("dimgray", 1.0)
    got = ax.patches[0].get_edgecolor()
    assert np.allclose(got, expected)
    plt.close(fig)


# TODO: remove before v1; temporary synthetic-GMM plotting semantics.
def test_ellipse_overlay_none_mode_keeps_all_components_grey():
    """color_mode='none' should disable all component colouring."""
    fig, ax = plt.subplots()
    means = np.array([[0.0, 0.0], [1.0, 1.0]])
    covs = np.stack([np.eye(2), np.eye(2)], axis=0)

    subspace_mod.ellipse_overlay(
        means,
        covs,
        ax,
        labels=np.array([10, 11]),
        palette={10: "#ff0000", 11: "#0000ff"},
        color_mode="none",
        n_signal=1,
        diagonal_approx=False,
        annotate=False,
    )

    expected = mcolors.to_rgba("dimgray", 1.0)
    assert np.allclose(ax.patches[0].get_edgecolor()[:3], expected[:3])
    assert np.allclose(ax.patches[1].get_edgecolor()[:3], expected[:3])
    plt.close(fig)
