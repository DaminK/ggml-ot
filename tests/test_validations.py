"""Direct validation tests for shared transformation utilities."""

from __future__ import annotations

import pytest
import torch

from ggml_ot._utils._array import slice_matrix
from ggml_ot._utils._covariance import (
    validate_cross_mean_covariance_shapes,
    validate_mean_covariance_shapes,
)
from ggml_ot._utils._transformation import project_covariances, project_means


def test_project_means_rejects_non_2d_linear_map():
    means = torch.randn(4, 3)
    linear_map = torch.randn(2, 3, 1)
    with pytest.raises(ValueError, match="linear_map must be 2D"):
        project_means(means, linear_map)


def test_project_means_rejects_feature_mismatch():
    means = torch.randn(4, 3)
    linear_map = torch.randn(2, 5)
    with pytest.raises(ValueError, match="means.shape\\[-1\\]=3 vs linear_map.shape\\[-1\\]=5"):
        project_means(means, linear_map)


def test_project_means_preserves_leading_dims():
    means = torch.randn(2, 4, 3)
    linear_map = torch.randn(5, 3)
    out = project_means(means, linear_map)
    assert out.shape == (2, 4, 5)


def test_project_covariances_rejects_non_2d_linear_map():
    covariances = torch.eye(3).reshape(1, 3, 3)
    linear_map = torch.randn(2, 3, 1)
    with pytest.raises(ValueError, match="linear_map must be 2D"):
        project_covariances(covariances, linear_map)


def test_project_covariances_rejects_non_square_covariances():
    covariances = torch.randn(4, 3, 2)
    linear_map = torch.randn(2, 2)
    with pytest.raises(ValueError, match="trailing dims must be square"):
        project_covariances(covariances, linear_map)


def test_project_covariances_rejects_feature_mismatch():
    covariances = torch.eye(3).reshape(1, 3, 3)
    linear_map = torch.randn(2, 4)
    with pytest.raises(ValueError, match="covariances.shape\\[-1\\]=3 vs linear_map.shape\\[-1\\]=4"):
        project_covariances(covariances, linear_map)


def test_project_covariances_preserves_leading_dims():
    covariances = torch.stack([torch.eye(3), 2.0 * torch.eye(3)], dim=0).unsqueeze(0)  # (1, 2, 3, 3)
    linear_map = torch.randn(5, 3)
    out = project_covariances(covariances, linear_map)
    assert out.shape == (1, 2, 5, 5)


def test_validate_mean_covariance_shapes_rejects_mismatch():
    means = torch.randn(4, 3)
    covariances = torch.randn(5, 3, 3)
    with pytest.raises(ValueError, match="Mean/covariance shape mismatch"):
        validate_mean_covariance_shapes(means, covariances, means_name="mu", covariances_name="sigma")


def test_validate_cross_mean_covariance_shapes_rejects_non_square():
    means_i = torch.randn(4, 3)
    cov_i = torch.randn(4, 3, 2)
    means_j = torch.randn(6, 3)
    cov_j = torch.randn(6, 3, 3)
    with pytest.raises(ValueError, match="Expected square covariances"):
        validate_cross_mean_covariance_shapes(means_i, cov_i, means_j, cov_j)


def test_slice_matrix_numpy_bool_masks():
    matrix = torch.arange(16, dtype=torch.float32).reshape(4, 4).cpu().numpy()
    rows = [True, False, True, False]
    cols = [False, True, True, False]
    out = slice_matrix(matrix, rows, cols)
    assert out.shape == (2, 2)
