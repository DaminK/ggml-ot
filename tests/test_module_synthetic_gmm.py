"""Synthetic GMM generator tests."""

from __future__ import annotations

import numpy as np
import pytest

import ggml_ot
from ggml_ot.data import from_synth_gmm
from ggml_ot.data.synthetic_gmm import (
    _build_spiked_subspace_covariances,
    _jitter_covariance,
    synth_gmm,
    synth_gmm_anndata,
)


def _small_synth_kwargs() -> dict:
    return {
        "n_dim": 4,
        "n_patients": 2,
        "n_samples": 40,
        "n_modes": 4,
        "random_seed": 123,
    }


def _clean_unmix_projection(data: dict) -> np.ndarray:
    q = np.asarray(data["Q_mixing"], dtype=float)
    r = np.asarray(data["R_rotation"], dtype=float)
    projection = np.eye(q.shape[0], dtype=float)
    projection[:2, :2] = r
    return q @ projection


def test_synth_gmm_new_parameterization_produces_expected_shapes():
    data = synth_gmm(
        signal_mass_ratio=0.35,
        signal_means_offset=9.35,
        signal_means_jitter=0.4,
        noise_means_offset=3.3,
        noise_means_jitter=0.25,
        signal_mean_shift=0.8,
        signal_cov_scale=1.1,
        signal_anisotropy=8.0,
        cov_rotation_jitter=7.0,
        cov_scale_jitter=0.2,
        global_rotation=25.0,
        **_small_synth_kwargs(),
    )

    assert data["X_high_all"].shape[1] == 4
    assert data["n_modes"] == 4
    assert data["n_noise"] == 2
    assert data["agg_means_2d"].shape == (6, 2)
    assert data["agg_covs_2d"].shape == (6, 2, 2)
    assert len(data["patient_gmms"]) == 2
    assert data["patient_gmms"][0]["means"].shape == (4, 4)


def test_signal_mean_shift_zero_gives_covariance_only_signal_means():
    data = synth_gmm(signal_mean_shift=0.0, **_small_synth_kwargs())
    signal_means = data["agg_means_2d"][:4]
    assert np.allclose(signal_means[0], signal_means[1])
    assert np.allclose(signal_means[2], signal_means[3])


def test_signal_mean_jitter_is_independent_of_signal_and_noise_offsets():
    kwargs = {
        **_small_synth_kwargs(),
        "n_dim": 6,
        "n_patients": 2,
        "n_samples": 60,
        "n_modes": 4,
        "signal_means_jitter": 0.4,
        "noise_means_jitter": 0.3,
        "signal_mean_shift": 0.6,
        "global_rotation": 25.0,
        "random_seed": 7,
    }
    data_lo = synth_gmm(signal_means_offset=7.5, noise_means_offset=2.0, **kwargs)
    data_hi = synth_gmm(signal_means_offset=17.5, noise_means_offset=6.0, **kwargs)

    proj_lo = _clean_unmix_projection(data_lo)
    proj_hi = _clean_unmix_projection(data_hi)
    means_lo = np.asarray(data_lo["patient_gmms"][0]["means"]) @ proj_lo
    means_hi = np.asarray(data_hi["patient_gmms"][0]["means"]) @ proj_hi

    signal_delta_lo = means_lo[:2, :2] - data_lo["agg_means_2d"][[0, 2]]
    signal_delta_hi = means_hi[:2, :2] - data_hi["agg_means_2d"][[0, 2]]
    assert np.allclose(signal_delta_lo, signal_delta_hi)
    assert np.allclose(means_lo[:2, 2:], means_hi[:2, 2:])


def test_signal_and_noise_mean_offsets_control_their_respective_subspaces():
    kwargs = {
        **_small_synth_kwargs(),
        "n_dim": 6,
        "n_patients": 2,
        "n_samples": 60,
        "n_modes": 4,
        "signal_means_offset": 10.0,
        "signal_means_jitter": 0.0,
        "noise_means_jitter": 0.0,
        "signal_mean_shift": 0.5,
        "global_rotation": 25.0,
        "random_seed": 17,
    }
    data_lo = synth_gmm(noise_means_offset=1.5, **kwargs)
    data_hi = synth_gmm(noise_means_offset=5.0, **kwargs)

    assert np.allclose(data_lo["agg_means_2d"][4:], 0.0)
    assert np.allclose(data_hi["agg_means_2d"][4:], 0.0)
    assert np.linalg.norm(data_lo["agg_means_2d"][:4], axis=1).min() > 0.0

    proj_lo = _clean_unmix_projection(data_lo)
    proj_hi = _clean_unmix_projection(data_hi)
    means_lo = np.asarray(data_lo["patient_gmms"][0]["means"]) @ proj_lo
    means_hi = np.asarray(data_hi["patient_gmms"][0]["means"]) @ proj_hi

    noise_plane_lo = means_lo[2:, :2]
    noise_plane_hi = means_hi[2:, :2]
    assert np.allclose(noise_plane_lo, 0.0)
    assert np.allclose(noise_plane_hi, 0.0)

    noise_subspace_lo = np.linalg.norm(means_lo[2:, 2:], axis=1)
    noise_subspace_hi = np.linalg.norm(means_hi[2:, 2:], axis=1)
    assert noise_subspace_hi.mean() > noise_subspace_lo.mean()


def test_signal_modes_have_patient_specific_nuisance_mean_jitter():
    data = synth_gmm(
        n_dim=6,
        n_patients=4,
        n_samples=60,
        n_modes=4,
        signal_means_jitter=0.5,
        noise_means_jitter=0.0,
        random_seed=11,
    )
    projection = _clean_unmix_projection(data)
    patient_means = [np.asarray(entry["means"]) @ projection for entry in data["patient_gmms"]]

    signal_means = np.stack([means[:2] for means in patient_means], axis=0)
    assert np.std(signal_means[:, :, 2:]) > 0.0
    assert np.allclose(signal_means[:, 0, 1], 0.0)
    assert np.allclose(signal_means[:, 1, 0], 0.0)


def test_signal_weights_can_stay_near_uniform_while_noise_weights_vary():
    data = synth_gmm(
        n_dim=6,
        n_patients=12,
        n_samples=120,
        n_modes=6,
        signal_mass_ratio=0.30,
        signal_weight_concentration=25.0,
        noise_weight_concentration=0.5,
        random_seed=19,
    )
    weights = np.stack([entry["weights"] for entry in data["patient_gmms"]], axis=0)

    signal_weight_gaps = np.abs(weights[:, 0] - weights[:, 1])
    noise_weight_spread = np.ptp(weights[:, 2:], axis=1)

    assert np.all(weights > 0.0)
    assert noise_weight_spread.mean() > signal_weight_gaps.mean()


def test_cov_rotation_jitter_rotates_high_dimensional_nuisance_covariances():
    rng = np.random.RandomState(23)
    cov = np.diag([1.0, 0.7, 0.3, 0.1])
    jittered = _jitter_covariance(
        cov,
        rng=rng,
        rotation_jitter=25.0,
        scale_jitter=0.0,
    )

    off_diag = jittered - np.diag(np.diag(jittered))
    assert np.linalg.norm(off_diag) > 0.0


def test_noise_subspace_rank_builds_low_rank_spiked_archetypes():
    rng = np.random.RandomState(29)
    cov = _build_spiked_subspace_covariances(
        n_modes=1,
        n_dims=6,
        rng=rng,
        cov_scale=0.8,
        rank=2,
    )[0]
    eigvals = np.linalg.eigvalsh(cov)
    tail = eigvals[:-2]
    head = eigvals[-2:]

    assert np.max(tail) - np.min(tail) < 1.0e-8
    assert np.all(head > tail[-1])


def test_synth_gmm_supports_signal_only_debug_configuration():
    data = synth_gmm(
        n_modes=2, signal_mass_ratio=0.1, **{k: v for k, v in _small_synth_kwargs().items() if k != "n_modes"}
    )
    assert data["n_noise"] == 0
    assert data["n_modes"] == 2
    assert all(entry["means"].shape[0] == 2 for entry in data["patient_gmms"])


def test_from_synth_gmm_uses_new_parameters_and_sets_provenance():
    dataset_cells = from_synth_gmm(
        representation="cells",
        adata=False,
        signal_mass_ratio=0.35,
        signal_means_offset=9.35,
        signal_means_jitter=0.4,
        noise_means_offset=3.3,
        noise_means_jitter=0.25,
        signal_mean_shift=0.8,
        signal_cov_scale=1.1,
        signal_anisotropy=8.0,
        cov_rotation_jitter=7.0,
        cov_scale_jitter=0.2,
        global_rotation=25.0,
        **_small_synth_kwargs(),
    )
    dataset_gmm = from_synth_gmm(representation="gmm", adata=False, **_small_synth_kwargs())

    assert len(dataset_cells) == 2
    assert dataset_cells.synth_data["fitted_gmm_provenance"] == "none"
    assert dataset_gmm.synth_data["fitted_gmm_provenance"] == "synthetic_ground_truth"


def test_synth_gmm_anndata_can_persist_ground_truth_gmm_schema():
    adata = synth_gmm_anndata(gmm_key="gmm_synth_raw", **_small_synth_kwargs())

    assert "gmm_synth_raw" in adata.uns
    cfg = adata.uns["gmm_synth_raw"]
    assert cfg["backend"] == "synthetic_ground_truth"
    assert cfg["component_sharing"] == "sample_specific"
    assert cfg["use_rep"] is None

    dataset = ggml_ot.from_anndata(adata, gmm_key="gmm_synth_raw")
    assert dataset.identical_supports is False
    assert dataset.weights is not None
    assert dataset.covariances is not None


def test_from_synth_gmm_adata_gmm_key_persists_schema_and_supports_reload():
    dataset = from_synth_gmm(
        representation="gmm",
        adata=True,
        gmm_key="gmm_synth_dataset",
        **_small_synth_kwargs(),
    )

    assert "gmm_synth_dataset" in dataset.adata.uns
    reloaded = ggml_ot.from_anndata(dataset.adata, gmm_key="gmm_synth_dataset")
    assert reloaded.identical_supports is False
    assert reloaded.weights is not None
    assert reloaded.covariances is not None


def test_from_synth_gmm_rejects_gmm_key_without_anndata():
    with pytest.raises(ValueError, match="gmm_key"):
        from_synth_gmm(representation="cells", adata=False, gmm_key="gmm_invalid", **_small_synth_kwargs())
