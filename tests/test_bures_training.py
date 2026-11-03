"""Regression tests for the differentiable full-Bures training path."""

import pytest
import torch

from ggml_ot.distances.bures_wasserstein import (
    bures_covariance_distance,
    cross_gaussian_distance,
    pairwise_gaussian_distance,
)
from ggml_ot.optimization.loss import batch_triplet_loss_emd2, batch_triplet_loss_sinkhorn
from ggml_ot.optimization.params import WassersteinBuresParams


def _random_spd_set(n_matrices: int, dimension: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    factors = torch.randn(n_matrices, dimension, dimension, dtype=torch.float64, generator=generator)
    identity = torch.eye(dimension, dtype=torch.float64)
    return factors @ factors.transpose(-1, -2) + 0.5 * identity


def test_differentiable_bures_matches_reference_counterexample():
    """The Cholesky branch must compute the same Bures value as the symmetric-root branch."""
    cov_a = torch.tensor([[[4.0, 1.0], [1.0, 2.0]]], dtype=torch.float64, requires_grad=True)
    cov_b = torch.tensor([[[2.0, 0.7], [0.7, 3.0]]], dtype=torch.float64, requires_grad=True)

    differentiable = bures_covariance_distance(cov_a, cov_b, eps=0, symmetric=False)
    reference = bures_covariance_distance(cov_a.detach(), cov_b.detach(), eps=0, symmetric=False)

    assert torch.allclose(reference, torch.tensor([[0.4508862847764519]], dtype=torch.float64))
    assert torch.allclose(differentiable, reference, atol=1e-8, rtol=1e-7)


def test_differentiable_bures_is_exchange_symmetric_with_matching_gradients():
    """Exchanging covariance sets must transpose values and exchange their gradients."""
    cov_a = _random_spd_set(3, 3, seed=1).requires_grad_()
    cov_b = _random_spd_set(4, 3, seed=2).requires_grad_()

    distance_ab = bures_covariance_distance(cov_a, cov_b, eps=0, symmetric=False)
    grad_a_ab, grad_b_ab = torch.autograd.grad(distance_ab.sum(), (cov_a, cov_b))

    cov_a_reverse = cov_a.detach().clone().requires_grad_()
    cov_b_reverse = cov_b.detach().clone().requires_grad_()
    distance_ba = bures_covariance_distance(cov_b_reverse, cov_a_reverse, eps=0, symmetric=False)
    grad_b_ba, grad_a_ba = torch.autograd.grad(distance_ba.sum(), (cov_b_reverse, cov_a_reverse))

    assert distance_ab.shape == (3, 4)
    assert distance_ba.shape == (4, 3)
    assert torch.allclose(distance_ab, distance_ba.transpose(-1, -2), atol=1e-8, rtol=1e-7)
    assert torch.allclose(grad_a_ab, grad_a_ba, atol=1e-8, rtol=1e-7)
    assert torch.allclose(grad_b_ab, grad_b_ba, atol=1e-8, rtol=1e-7)


def test_differentiable_bures_passes_gradcheck():
    """The corrected Cholesky formulation must retain valid autograd derivatives."""
    factor_a = _random_spd_set(2, 3, seed=3).requires_grad_()
    factor_b = _random_spd_set(3, 3, seed=4).requires_grad_()
    identity = torch.eye(3, dtype=torch.float64)

    def distance_from_factors(a, b):
        cov_a = a @ a.transpose(-1, -2) + 0.5 * identity
        cov_b = b @ b.transpose(-1, -2) + 0.5 * identity
        return bures_covariance_distance(cov_a, cov_b, eps=0, symmetric=False)

    assert torch.autograd.gradcheck(
        distance_from_factors,
        (factor_a, factor_b),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_differentiable_bures_is_finite_near_jitter_floor():
    """The corrected path must retain finite values and gradients for near-singular inputs."""
    cov_a = torch.tensor([[[1e-12, 0.0], [0.0, 1.0]]], dtype=torch.float64, requires_grad=True)
    cov_b = torch.tensor([[[0.5, 0.1], [0.1, 2.0]]], dtype=torch.float64, requires_grad=True)

    distance = bures_covariance_distance(cov_a, cov_b, eps=1e-4, symmetric=False)
    gradients = torch.autograd.grad(distance.sum(), (cov_a, cov_b))

    assert torch.all(torch.isfinite(distance))
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)


def test_cross_gaussian_distance_with_trainable_map_matches_pairwise_reference():
    """A trainable-map cross call must agree with the unaffected pairwise evaluation path."""
    means = torch.tensor(
        [[0.0, 0.0], [1.0, -0.5], [-0.25, 0.75]],
        dtype=torch.float32,
    )
    covariances = _random_spd_set(3, 2, seed=5).float()
    linear_map = torch.tensor([[1.0, 0.2], [0.1, 0.8]], dtype=torch.float32, requires_grad=True)

    pairwise = pairwise_gaussian_distance(means, covariances, linear_map, eps=0)
    cross = cross_gaussian_distance(
        means,
        covariances,
        means,
        covariances,
        linear_map,
        eps=0,
        squared_ground_cost=True,
    )

    assert torch.allclose(cross, pairwise, atol=1e-5, rtol=1e-5)


def _triplet_loss_inputs():
    factors = torch.tensor(
        [
            [[1.4, 0.2], [0.3, 0.9]],
            [[0.8, 0.5], [0.1, 1.3]],
            [[1.2, -0.4], [0.2, 0.7]],
            [[0.7, 0.1], [-0.5, 1.1]],
        ],
        dtype=torch.float32,
    )
    covariances = (factors @ factors.transpose(-1, -2) + 0.2 * torch.eye(2)).unsqueeze(1)
    points = torch.zeros((4, 1, 2), dtype=torch.float32)
    weights = torch.ones((4, 1), dtype=torch.float32)
    triplets = torch.tensor([[0, 1, 2]], dtype=torch.long)
    linear_map = torch.tensor([[1.0, 0.2], [0.1, 0.8]], dtype=torch.float32)
    return points, covariances, weights, triplets, linear_map


@pytest.mark.parametrize("loss_fn", [batch_triplet_loss_emd2, batch_triplet_loss_sinkhorn])
def test_exact_triplet_loss_is_invariant_to_batch_permutation(loss_fn):
    """Batch-local pair orientation must not change Exact loss or learned-map gradients."""
    points, covariances, weights, triplets, initial_map = _triplet_loss_inputs()
    permutation = torch.tensor([1, 2, 0, 3])
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(len(permutation))
    permuted_triplets = inverse_permutation[triplets]
    ot_params = WassersteinBuresParams(
        identical_supports=False,
        eps=0,
        squared_ground_cost=True,
        entropic_reg=0.5,
        sinkhorn_max_iter=100,
        sinkhorn_stop_thr=1e-8,
    )

    results = []
    for batch_points, batch_covariances, batch_weights, batch_triplets in (
        (points, covariances, weights, triplets),
        (points[permutation], covariances[permutation], weights[permutation], permuted_triplets),
    ):
        linear_map = initial_map.clone().requires_grad_()
        loss = loss_fn(
            torch.tensor(0.0),
            batch_points,
            batch_weights,
            batch_covariances,
            batch_triplets,
            linear_map,
            10.0,
            ot_params=ot_params,
        )
        (gradient,) = torch.autograd.grad(loss, (linear_map,))
        results.append((loss.detach(), gradient.detach()))

    assert torch.allclose(results[0][0], results[1][0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(results[0][1], results[1][1], atol=1e-5, rtol=1e-5)
