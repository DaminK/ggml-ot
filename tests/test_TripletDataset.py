"""TripletDataset unit tests.

This file tests core functionalities of the dataset class that GGML builds on, including:
 - indexing
 - subsetting
 - handling of untrained states.
"""

import numpy as np
import pytest

import ggml_ot
from ggml_ot.data.generic import TripletDataset
from .utils.config import get_synth_config


def test_TripletDataset_GGML_interface():
    """Tests core functions of GGML when using methods of TripletDataset class"""

    # Setup small version of synthetic 2D dataset (from AISTATS25 paper)
    dataset = ggml_ot.data.from_synth(distribution_size=25, show=False)

    # Check training
    map_A = dataset.train(max_iter=1, plot_iter=0, verbose=False, return_dataset=False)
    assert map_A is not None

    # Check test
    scores = dataset.test(map_A, n_splits=1, verbose=False)
    assert scores is not None


def test_map_A_warns_if_untrained():
    """Accessing map_A before training should emit a warning."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    with pytest.warns(UserWarning, match=r"not been trained"):
        _ = ds.map_A


def test_w_theta_deprecated():
    """Accessing w_theta on an untrained dataset emits a DeprecationWarning (and a UserWarning since untrained)."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    with pytest.warns((DeprecationWarning, UserWarning)) as record:
        _ = ds.w_theta
    assert any(issubclass(w.category, DeprecationWarning) and "w_theta is deprecated" in str(w.message) for w in record)


""" DEPRECATED with current triplet implementation
def test_subset_preserves_alignment_and_regenerates_triplets():
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    sub = ds._subset([0, 2, 4, 5])

    assert len(sub.supports) == 4
    assert len(sub.distribution_labels) == 4
    assert sub.dim == ds.dim

    # Triplets must be valid indices in the subset.
    assert len(sub.triplets) == len(sub)
    assert all(0 <= i < 4 and 0 <= j < 4 and 0 <= k < 4 for (i, j, k) in sub.triplets)
"""


def test_getitem_empirical_contract():
    """Empirical datasets should return empty covariance/weight placeholders."""
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    support, covariances, weights, label = ds[0]

    assert support.shape[-1] == ds.dim
    assert covariances.numel() == 0
    assert weights.numel() == 0  # weights=None returns empty tensor by design
    assert np.isscalar(label)


def test_getitem_identical_supports_contract():
    """Identical-support datasets should return weights and empty support/covariance tensors."""
    rng = np.random.default_rng(0)

    supports = rng.normal(size=(6, 2)).astype(np.float32)  # shared support points
    weights = [rng.random(size=(6,)).astype(np.float32) for _ in range(4)]
    weights = [w / w.sum() for w in weights]
    labels = np.array([0, 0, 1, 1])

    ds = TripletDataset(
        supports=supports,
        distribution_labels=labels,
        weights=weights,
        identical_supports=True,
    )

    supports, covariances, weights_one, label_one = ds[0]
    assert supports.numel() == 0
    assert covariances.numel() == 0
    assert weights_one.shape == (6,)
    assert np.isscalar(label_one)
