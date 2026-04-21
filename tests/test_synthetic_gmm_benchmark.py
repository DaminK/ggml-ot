"""Synthetic GMM benchmark regressions."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warnings

import ggml_ot
from ggml_ot.data import from_synth_gmm


pytestmark = [pytest.mark.gmm, pytest.mark.dev]


SMALL_SPLIT_SYNTH_KWARGS = dict(
    n_dim=10,
    n_patients=6,
    n_samples=80,
    signal_mass_ratio=0.25,
    n_modes=6,
    signal_means_offset=12.0,
    signal_means_jitter=0.75,
    noise_means_offset=3.0,
    noise_means_jitter=0.75,
    signal_mean_shift=1.0,
    signal_cov_scale=1.2,
    signal_anisotropy=12.0,
    cov_rotation_jitter=10.0,
    cov_scale_jitter=0.15,
    global_rotation=30.0,
    random_seed=42,
)


def test_synthetic_gmm_train_beats_identity_gaussian_ot():
    """Learned diag-Bures + MI should beat the identity Gaussian baseline."""
    synth_kwargs = dict(
        n_dim=16,
        n_patients=40,
        n_samples=500,
        signal_mass_ratio=0.08,
        n_modes=12,
        signal_means_offset=22.5,
        signal_means_jitter=1.8,
        noise_means_offset=1.62,
        noise_means_jitter=1.8,
        signal_mean_shift=0.10,
        signal_cov_scale=1.2,
        signal_anisotropy=12.0,
        cov_rotation_jitter=10.0,
        cov_scale_jitter=0.15,
        global_rotation=30.0,
        random_seed=42,
    )
    train_kwargs = dict(
        alpha=10,
        reg=0.001,
        reg_type="fro",
        n_comps=2,
        train_size=0.7,
        verbose=False,
        n_splits=1,
        lr=0.1,
        max_iter=30,
        diag_bures_approx=True,
        mi_reg=1.0,
        entropic_reg=0.0,
        plot_iter=0,
        plot_split=False,
        plot_type=False,
        print_table=False,
        return_dataset=False,
    )
    test_kwargs = dict(
        train_size=train_kwargs["train_size"],
        n_splits=train_kwargs["n_splits"],
        verbose=False,
        plot_split=False,
        plot_type=False,
        print_table=False,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"\[Errno 13\] Permission denied\.  joblib will operate in serial mode"
        )
        warnings.filterwarnings("ignore", message=r"CUDA initialization.*")

        np.random.seed(0)
        torch.manual_seed(0)
        dataset = from_synth_gmm(representation="gmm", adata=False, **synth_kwargs)

        np.random.seed(0)
        torch.manual_seed(0)
        _, learned_scores = ggml_ot.train_test(dataset, **train_kwargs)
        baseline_scores = ggml_ot.test(dataset, ground_metric="euclidean", **test_kwargs)

    learned_acc = float(learned_scores.iloc[0][("knn", "mean")])
    baseline_acc = float(baseline_scores.iloc[0][("knn", "mean")])

    # On GMM datasets, ground_metric="euclidean" is the identity Gaussian OT
    # baseline (Euclidean mean term + Bures covariance term).
    assert learned_acc >= baseline_acc + 0.25


def test_synthetic_gmm_test_accepts_knn_k_on_small_splits():
    """Small synthetic splits should support an explicit benchmark KNN setting."""
    dataset = from_synth_gmm(representation="gmm", adata=False, **SMALL_SPLIT_SYNTH_KWARGS)

    scores = ggml_ot.test(
        dataset,
        ground_metric="euclidean",
        train_size=0.8,
        n_splits=1,
        knn_k=3,
        verbose=False,
        plot_split=False,
        plot_type=False,
        print_table=False,
    )

    assert ("knn", "mean") in scores.columns


def test_synthetic_gmm_test_rejects_knn_k_larger_than_train_split():
    """The benchmark API should raise a clear error for invalid KNN settings."""
    dataset = from_synth_gmm(representation="gmm", adata=False, **SMALL_SPLIT_SYNTH_KWARGS)

    with pytest.raises(ValueError, match=r"`n_neighbors` \(5\) exceeds the train split size \(4\)"):
        ggml_ot.test(
            dataset,
            ground_metric="euclidean",
            train_size=0.8,
            n_splits=1,
            knn_k=5,
            verbose=False,
            plot_split=False,
            plot_type=False,
            print_table=False,
        )
