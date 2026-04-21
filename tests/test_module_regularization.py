"""Regularization loss unit tests for norm and mutual-information terms."""

import pytest
import torch

from ggml_ot.optimization.params import MIRegularizationParams, RegularizationParams
from ggml_ot.optimization.regularization import (
    mutual_information_loss,
    regularizer_loss,
    regularizer_loss_mutual_information,
)


def test_regularizer_loss_l1():
    """reg_type=1 should use elementwise L1 norm, not matrix induced norm."""
    w = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    params = RegularizationParams(reg=1.0, reg_type=1)

    reg = regularizer_loss(w, params)

    assert torch.isclose(reg, w.abs().sum())


@pytest.mark.parametrize("reg_type", ["fro", 2], ids=["fro", "2"])
def test_regularizer_loss_l2(reg_type):
    """Frobenius aliases should match torch.linalg.norm(..., ord='fro')."""
    w = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    params = RegularizationParams(reg=1.0, reg_type=reg_type)

    reg = regularizer_loss(w, params)

    assert torch.isclose(reg, torch.linalg.norm(w, ord="fro"))


def test_regularizer_loss_nuc():
    """Nuclear regularization should match torch.linalg.norm(..., ord='nuc')."""
    w = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    params = RegularizationParams(reg=1.0, reg_type="nuc")

    reg = regularizer_loss(w, params)

    assert torch.isclose(reg, torch.linalg.norm(w, ord="nuc"))


def test_regularizer_loss_mutual_information():
    """Mutual-information loss should support batched covariances with weights."""
    sigma = torch.eye(3).repeat(2, 4, 1, 1)  # (B=2, C=4, D=3, D=3)
    w = torch.randn(2, 3)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

    loss = mutual_information_loss(sigma, w, weights=weights)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("weighting", ["components", "uniform", "projection"])
def test_regularizer_loss_mutual_information_backward(weighting):
    """Mutual-information regularizer should backpropagate gradients to W for all weighting strategies."""
    w = torch.randn(2, 3, requires_grad=True)
    mu = torch.randn(5, 4, 3)
    sigma = torch.eye(3).repeat(5, 4, 1, 1)
    weights = torch.rand(5, 4)

    params = MIRegularizationParams(
        reg=0.1,
        reg_type="fro",
        mi_reg=1.0,
        mi_reg_weighting=weighting,
    )

    loss = regularizer_loss_mutual_information(w, mu, sigma, weights, params)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert w.grad is not None
