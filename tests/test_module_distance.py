"""Distance computation unit tests.

Scope
-----
- `compute_OT` basic invariants (symmetry, diagonal)
- error handling for invalid combinations
"""

import numpy as np
import ot
import pytest
import torch

from ggml_ot.distances.ot import compute_OT
from ggml_ot.distances.bures_wasserstein import (
    bures_covariance_distance,
    cross_gaussian_distance,
    pairwise_gaussian_distance,
)


def test_compute_ot_symmetric_and_zero_diag_empirical():
    """Empirical OT matrix should be symmetric with zero diagonal."""
    rng = np.random.default_rng(0)
    supports = np.stack([rng.normal(size=(5, 3)).astype(np.float32) for _ in range(4)], axis=0)

    D = compute_OT(supports, weights=None, identical_supports=False, ground_metric="euclidean")

    assert D.shape == (4, 4)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0)


def test_compute_ot_identical_supports_requires_weights():
    """Identical-support OT without weights should fail fast."""
    rng = np.random.default_rng(0)
    shared_supports = rng.normal(size=(6, 2)).astype(np.float32)

    with pytest.raises(ValueError, match=r"weights is None"):
        compute_OT(shared_supports, weights=None, identical_supports=True, ground_metric=None)


def test_compute_ot_accepts_condensed_precomputed_distances():
    """When precomputed_distances is 1D, compute_OT should squareform it."""

    rng = np.random.default_rng(0)
    supports = np.stack([rng.normal(size=(3, 2)).astype(np.float32) for _ in range(3)], axis=0)

    # Precompute full point-to-point distances between concatenated supports.
    points = np.concatenate(supports, axis=0)
    full = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    # Convert to condensed form like scipy.spatial.distance.squareform uses.
    # We re-derive it to avoid importing scipy in tests.
    iu = np.triu_indices(full.shape[0], k=1)
    condensed = full[iu]

    D1 = compute_OT(supports, precomputed_distances=condensed, ground_metric=None)
    D2 = compute_OT(supports, precomputed_distances=full, ground_metric=None)

    assert np.allclose(D1, D2)


def test_compute_ot_nonidentical_gmm_uses_covariances():
    """Non-identical GMM OT should depend on covariance tensors."""
    rng = np.random.default_rng(0)
    supports = np.stack([rng.normal(size=(2, 2)).astype(np.float32) for _ in range(3)], axis=0)
    weights = np.full((3, 2), 0.5, dtype=np.float32)

    covariances_a = np.stack(
        [
            np.stack(
                [
                    np.diag(np.array([1.0 + 0.1 * i, 1.2 + 0.2 * i], dtype=np.float32)),
                    np.diag(np.array([1.1 + 0.1 * i, 1.4 + 0.2 * i], dtype=np.float32)),
                ],
                axis=0,
            )
            for i in range(3)
        ],
        axis=0,
    )
    covariances_b = covariances_a * 5.0

    D_a = compute_OT(
        supports,
        covariances=covariances_a,
        weights=weights,
        identical_supports=False,
        ground_metric="euclidean",
    )
    D_b = compute_OT(
        supports,
        covariances=covariances_b,
        weights=weights,
        identical_supports=False,
        ground_metric="euclidean",
    )

    assert not np.allclose(D_a, D_b)


def test_compute_ot_nonidentical_gmm_skips_zero_weight_padded_components():
    """Zero-weight components should be excluded from Gaussian ground-cost computation."""
    supports = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [99.0, 99.0]],
            [[0.0, 1.0], [1.0, 1.0], [98.0, 98.0]],
        ],
        dtype=np.float32,
    )
    covariances = np.array(
        [
            [np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32), np.full((2, 2), np.nan, dtype=np.float32)],
            [np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32), np.full((2, 2), np.nan, dtype=np.float32)],
        ],
        dtype=np.float32,
    )
    weights = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    D = compute_OT(
        supports=supports,
        covariances=covariances,
        weights=weights,
        identical_supports=False,
        ground_metric="euclidean",
    )

    assert D.shape == (2, 2)
    assert np.all(np.isfinite(D))


def test_compute_ot_identical_gmm_skips_zero_weight_padded_components():
    """Identical-support Gaussian OT should avoid full precompute over zero-weight padded components."""
    supports = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [99.0, 99.0],
        ],
        dtype=np.float32,
    )
    covariances = np.array(
        [
            np.eye(2, dtype=np.float32),
            np.eye(2, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    weights = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    D = compute_OT(
        supports=supports,
        covariances=covariances,
        weights=weights,
        identical_supports=True,
        ground_metric="euclidean",
    )

    assert D.shape == (2, 2)
    assert np.all(np.isfinite(D))


def test_compute_ot_handles_small_mass_drift():
    """Compute OT should be robust to tiny weight-sum mismatch."""
    rng = np.random.default_rng(0)
    supports = rng.normal(size=(3, 2)).astype(np.float32)
    weights = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
        ],
        dtype=np.float32,
    )
    # Reproduce small sum drift seen in notebook runs after float32 pipelines.
    weights[0] *= np.float32(1.000001)
    weights[1] *= np.float32(1.000004)

    D = compute_OT(
        supports,
        weights=weights,
        identical_supports=True,
        ground_metric="euclidean",
    )

    assert D.shape == (2, 2)
    assert np.all(np.isfinite(D))
    assert np.allclose(D, D.T)


def test_compute_ot_sinkhorn_matches_pot_reference():
    """compute_OT should dispatch to POT Sinkhorn when entropic_reg > 0."""
    supports = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    weights = np.array(
        [
            [0.75, 0.25],
            [0.40, 0.60],
        ],
        dtype=np.float32,
    )
    entropic_reg = 0.5

    D_sink = compute_OT(
        supports,
        weights=weights,
        identical_supports=False,
        ground_metric="euclidean",
        entropic_reg=entropic_reg,
    )
    D_exact = compute_OT(
        supports,
        weights=weights,
        identical_supports=False,
        ground_metric="euclidean",
        entropic_reg=0.0,
    )

    M = np.linalg.norm(supports[0][:, None, :] - supports[1][None, :, :], axis=-1)
    M_scale = max(float(M.mean()), 1e-12)
    expected = ot.sinkhorn2(
        weights[0],
        weights[1],
        M / M_scale,
        reg=entropic_reg,
        numItermax=100,
        stopThr=0.0,
        warn=False,
    )
    expected = float(expected) * M_scale

    assert D_sink.shape == (2, 2)
    assert np.all(np.isfinite(D_sink))
    assert np.allclose(D_sink, D_sink.T)
    assert np.allclose(np.diag(D_sink), 0.0)
    assert np.isclose(float(D_sink[0, 1]), expected, rtol=1e-5, atol=1e-5)
    assert not np.isclose(float(D_sink[0, 1]), float(D_exact[0, 1]))


@pytest.mark.parametrize("diag_bures_approx", [False, True])
@pytest.mark.parametrize("squared_ground_cost", [True, False])
def test_pairwise_and_cross_gaussian_distance_match_on_same_inputs(diag_bures_approx, squared_ground_cost):
    """pairwise_gaussian_distance should match cross_gaussian_distance(X, X) for same inputs."""
    rng = np.random.default_rng(0)
    n, d = 4, 3
    means = rng.normal(size=(n, d)).astype(np.float32)
    a = rng.normal(size=(n, d, d)).astype(np.float32)
    covs = np.matmul(a, np.transpose(a, (0, 2, 1))) + np.eye(d, dtype=np.float32)[None] * 0.1
    linear_map = np.eye(d, dtype=np.float32)

    pairwise = pairwise_gaussian_distance(
        means,
        covs,
        linear_map,
        diag_bures_approx=diag_bures_approx,
        as_numpy=True,
        squared_ground_cost=squared_ground_cost,
    )
    cross = cross_gaussian_distance(
        means,
        covs,
        means,
        covs,
        linear_map,
        diag_bures_approx=diag_bures_approx,
        as_numpy=True,
        squared_ground_cost=squared_ground_cost,
    )

    assert np.allclose(pairwise, cross, atol=1e-5, rtol=1e-5)


def test_pairwise_gaussian_distance_respects_squared_ground_cost_flag():
    """Squared/non-squared mean terms should produce distinct Gaussian distances."""
    means = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    covs = np.stack([np.eye(2, dtype=np.float32) for _ in range(3)], axis=0)
    linear_map = np.eye(2, dtype=np.float32)

    squared = pairwise_gaussian_distance(means, covs, linear_map, as_numpy=True, squared_ground_cost=True)
    unsquared = pairwise_gaussian_distance(means, covs, linear_map, as_numpy=True, squared_ground_cost=False)

    assert squared.shape == unsquared.shape == (3, 3)
    assert np.allclose(np.diag(squared), 0.0)
    assert np.allclose(np.diag(unsquared), 0.0)
    assert not np.allclose(squared, unsquared)


def test_bures_covariance_distance_supports_batched_inputs():
    """Batched Bures helper should return finite, symmetric pairwise matrices."""
    rng = np.random.default_rng(1)
    batch, n, d = 2, 3, 2
    a = rng.normal(size=(batch, n, d, d)).astype(np.float32)
    covs = np.matmul(a, np.transpose(a, (0, 1, 3, 2))) + np.eye(d, dtype=np.float32)[None, None] * 0.1
    covs_t = torch.tensor(covs)

    full = bures_covariance_distance(covs_t, covs_t, diag_approx=False, symmetric=True)
    diag = bures_covariance_distance(covs_t, covs_t, diag_approx=True, symmetric=True)

    assert full.shape == (batch, n, n)
    assert diag.shape == (batch, n, n)
    assert torch.all(torch.isfinite(full))
    assert torch.all(torch.isfinite(diag))
    assert torch.allclose(full, full.transpose(-1, -2), atol=1e-6, rtol=1e-6)
    assert torch.allclose(diag, diag.transpose(-1, -2), atol=1e-6, rtol=1e-6)
