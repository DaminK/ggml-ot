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


# ---------------------------------------------------------------------------
# Ordinal label support
# ---------------------------------------------------------------------------


def test_distribution_labels_plot_categorical_returns_strings():
    ds = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    plot_labels = ds.distribution_labels_plot
    assert plot_labels.dtype.kind in {"U", "O"}


# ---------------------------------------------------------------------------
# distribution_labels_str storage + subsetting (label streamline)
# ---------------------------------------------------------------------------


def _generic_dataset(labels, distribution_labels_str=None, seed=0):
    """Minimal generic TripletDataset with one support per label."""
    rng = np.random.default_rng(seed)
    supports = [rng.normal(size=(5, 2)).astype(np.float32) for _ in labels]
    return TripletDataset(
        supports=supports,
        distribution_labels=labels,
        distribution_labels_str=distribution_labels_str,
    )


def test_generic_categorical_stores_stringified_labels():
    """Without explicit strings, distribution_labels_str is the stringified labels."""
    ds = _generic_dataset([0, 1, 0, 1])
    assert list(ds.distribution_labels_str) == ["0", "1", "0", "1"]
    assert len(ds.distribution_labels_str) == len(ds.distribution_labels)


def test_generic_explicit_string_labels_preserved():
    """Explicit distribution_labels_str is stored verbatim."""
    ds = _generic_dataset([0, 1, 2], distribution_labels_str=["mild", "severe", "healthy"])
    assert list(ds.distribution_labels_str) == ["mild", "severe", "healthy"]


def test_distribution_labels_str_length_mismatch_raises():
    with pytest.raises(ValueError, match="distribution_labels_str"):
        _generic_dataset([0, 1, 2], distribution_labels_str=["a", "b"])


def test_subset_preserves_label_array_alignment():
    """_subset keeps distribution_labels / _str / _plot aligned with supports."""
    ds = _generic_dataset(
        [0, 1, 2, 0, 1, 2],
        distribution_labels_str=["a", "b", "c", "a", "b", "c"],
    )
    sub = ds._subset([0, 2, 4])
    assert len(sub.supports) == 3
    assert len(sub.distribution_labels) == 3
    assert len(sub.distribution_labels_str) == 3
    assert len(sub.distribution_labels_plot) == 3
    assert list(sub.distribution_labels_str) == ["a", "c", "b"]


def test_anndata_categorical_subset_label_alignment():
    """Categorical AnnData subset keeps every per-distribution label array aligned.

    Regression test for the subset/plot length mismatch: distribution_labels_str
    was recomputed from the full AnnData and diverged from a subset's supports.
    """
    adata = ggml_ot.data.synth_anndata(distribution_size=20)
    ds = ggml_ot.from_anndata(adata, n_cells=10)
    assert ds.label_type == "categorical"

    sub = ds._subset(list(range(len(ds.distribution_labels) - 1)))  # drop one distribution
    n = len(sub.supports)
    assert n == len(sub.distribution_labels) == len(sub.distribution_labels_str) == len(sub.distribution_labels_plot)


def test_anndata_categorical_labels_int_ids_str_keeps_original():
    """Categorical distribution_labels are int IDs; distribution_labels_str keeps obs values."""
    adata = ggml_ot.data.synth_anndata(distribution_size=20)
    ds = ggml_ot.from_anndata(adata, n_cells=10)
    assert np.issubdtype(np.asarray(ds.distribution_labels).dtype, np.integer)
    obs_values = set(adata.obs["patient_group"].astype(str))
    assert set(ds.distribution_labels_str).issubset(obs_values)


@pytest.mark.parametrize("n_samples", [7, 10, 23])
def test_split_complementary_fractions_cover_all_samples(n_samples):
    """train_size + test_size == 1 must partition every sample (no dropped/duplicated index).

    The fractions collapse to a single one so sklearn derives the integer
    complement; independently rounded counts could otherwise drop a sample.
    """
    from ggml_ot.data._split import _split_shuffle

    labels = np.zeros(n_samples)
    train_idx, test_idx = _split_shuffle(labels, n_splits=1, train_size=0.7, test_size=0.3)[0]
    assert set(train_idx) | set(test_idx) == set(range(n_samples))
    assert set(train_idx).isdisjoint(test_idx)


def test_split_partial_fractions_leave_holdout():
    """train_size + test_size < 1 keeps a genuine holdout instead of being normalized away."""
    from ggml_ot.data._split import _split_shuffle

    labels = np.zeros(20)
    train_idx, test_idx = _split_shuffle(labels, n_splits=1, train_size=0.5, test_size=0.3)[0]
    assert len(train_idx) + len(test_idx) < 20
    assert set(train_idx).isdisjoint(test_idx)
