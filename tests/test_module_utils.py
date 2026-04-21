"""Unit tests for internal utility helpers.

Covers:
- ``StratifiedBatchSampler`` — proportional class representation per batch.
- ``warn_empty_triplets``    — empty-triplet guard used in the training loop.
- ``_create_triplets``       — contrastive triplet index generation.
- ``_setup_dataloader``      — DataLoader factory with stratified batching.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

import ggml_ot
from ggml_ot._utils._batch import StratifiedBatchSampler, _setup_dataloader
from ggml_ot.optimization._triplets import warn_empty_triplets
from ggml_ot.optimization.triplets import _create_triplets
from ggml_ot.optimization.params import DataLoaderParams

from .utils.config import get_synth_config


# ---------------------------------------------------------------------------
# Shared fixture — reuse the existing synthetic smoke dataset.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_dataset():
    """Synthetic smoke dataset from ``dataset_params.yml``."""
    return ggml_ot.data.from_synth(**get_synth_config(), show=False)


# ===================================================================
# StratifiedBatchSampler
# ===================================================================


class TestStratifiedBatchSampler:
    """Tests for ``StratifiedBatchSampler``."""

    @staticmethod
    def _labels(ds):
        return ds.distribution_labels

    def test_all_indices_used_once(self, smoke_dataset):
        """Every sample index appears exactly once across all batches."""
        labels = self._labels(smoke_dataset)
        sampler = StratifiedBatchSampler(labels, batch_size=4, shuffle=False)
        all_indices = sorted(idx for batch in sampler for idx in batch)
        assert all_indices == list(range(len(labels)))

    def test_class_proportions_preserved(self, smoke_dataset):
        """Each batch should roughly preserve the global class frequencies."""
        labels = self._labels(smoke_dataset)
        sampler = StratifiedBatchSampler(labels, batch_size=4, shuffle=False)
        for batch in sampler:
            batch_labels = [labels[i] for i in batch]
            counts = Counter(batch_labels)
            assert len(counts) >= 2, f"Batch has < 2 classes: {counts}"

    def test_single_batch_fallthrough(self, smoke_dataset):
        """When batch_size >= N, sampler yields one batch with all indices."""
        labels = self._labels(smoke_dataset)
        sampler = StratifiedBatchSampler(labels, batch_size=9999, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 1
        assert sorted(batches[0]) == list(range(len(labels)))

    def test_shuffle_varies_order(self, smoke_dataset):
        """Shuffled iteration produces different intra-batch orderings."""
        import numpy as np

        labels = self._labels(smoke_dataset)
        sampler = StratifiedBatchSampler(labels, batch_size=4, shuffle=True)
        np.random.seed(0)
        run_a = [list(b) for b in sampler]
        np.random.seed(1)
        run_b = [list(b) for b in sampler]
        assert any(a != b for a, b in zip(run_a, run_b))

    def test_len_estimate(self, smoke_dataset):
        """__len__ returns a reasonable batch-count estimate."""
        labels = self._labels(smoke_dataset)
        sampler = StratifiedBatchSampler(labels, batch_size=4, shuffle=False)
        assert len(sampler) >= len(list(sampler))

    def test_imbalanced_minority_class(self):
        """A tiny minority class still gets quota >= 1 per batch."""
        labels = [0] * 50 + [1] * 2
        sampler = StratifiedBatchSampler(labels, batch_size=10, shuffle=False)
        first_batch = next(iter(sampler))
        batch_labels = [labels[i] for i in first_batch]
        assert 1 in Counter(batch_labels), "Minority class missing from first batch"


# ===================================================================
# _create_triplets
# ===================================================================


class TestCreateTriplets:
    """Tests for ``_create_triplets``."""

    def test_basic_two_classes(self):
        """Two classes with 2 samples each produces the expected number of triplets."""
        labels = [0, 0, 1, 1]
        triplets = _create_triplets(labels)
        assert triplets.ndim == 2
        assert triplets.shape[1] == 3
        # Same-class pairs: (0,0),(0,1),(1,0),(1,1) for class 0 and class 1
        # each pair has 2 different-class k options → 4*2 + 4*2 = 16
        assert triplets.shape[0] == 16

    def test_single_class_returns_empty(self):
        """A single class cannot form contrastive triplets."""
        triplets = _create_triplets([0, 0, 0])
        assert triplets.shape[0] == 0

    def test_empty_input(self):
        """Empty labels produce empty triplets."""
        triplets = _create_triplets([])
        assert triplets.shape[0] == 0

    def test_triplet_class_invariant(self):
        """For every (i,j,k): labels[i]==labels[j] and labels[j]!=labels[k]."""
        labels = torch.tensor([0, 0, 1, 1, 2])
        triplets = _create_triplets(labels)
        for row in triplets:
            i, j, k = row.tolist()
            assert labels[i] == labels[j]
            assert labels[j] != labels[k]

    def test_on_smoke_dataset(self, smoke_dataset):
        """Smoke dataset labels produce valid triplets."""
        labels = smoke_dataset.distribution_labels
        triplets = _create_triplets(labels)
        assert triplets.shape[0] > 0
        assert triplets.shape[1] == 3


# ===================================================================
# warn_empty_triplets
# ===================================================================


class TestWarnEmptyTriplets:
    """Tests for ``warn_empty_triplets``."""

    def test_returns_false_for_nonempty(self):
        triplets = torch.tensor([[0, 1, 2]])
        labels = torch.tensor([0, 0, 1])
        assert warn_empty_triplets(triplets, labels) is False

    def test_returns_true_and_warns_for_empty(self):
        triplets = torch.zeros((0, 3), dtype=torch.long)
        labels = torch.tensor([0, 0, 0])
        with pytest.warns(UserWarning, match="no valid triplets"):
            result = warn_empty_triplets(triplets, labels, epoch=1, batch_idx=2)
        assert result is True

    def test_warning_message_content(self):
        triplets = torch.zeros((0, 3), dtype=torch.long)
        labels = torch.tensor([0, 0])
        with pytest.warns(UserWarning, match=r"Epoch 5, batch 3.*2 sample.*1 class"):
            warn_empty_triplets(triplets, labels, epoch=5, batch_idx=3)

    def test_accepts_list_labels(self):
        """Should also work with plain Python lists as labels."""
        triplets = torch.zeros((0, 3), dtype=torch.long)
        labels = ["A", "A", "A"]
        with pytest.warns(UserWarning, match="no valid triplets"):
            assert warn_empty_triplets(triplets, labels) is True


# ===================================================================
# _setup_dataloader
# ===================================================================


class TestSetupDataloader:
    """Tests for ``_setup_dataloader`` factory."""

    def test_single_batch_uses_plain_dataloader(self, smoke_dataset):
        """batch_size >= dataset length → standard DataLoader (no batch_sampler)."""
        params = DataLoaderParams(dataset=smoke_dataset, batch_size=9999)
        dl = _setup_dataloader(params)
        assert not isinstance(dl.batch_sampler, StratifiedBatchSampler)

    def test_mini_batch_uses_stratified_sampler(self, smoke_dataset):
        """batch_size < dataset length → StratifiedBatchSampler."""
        params = DataLoaderParams(dataset=smoke_dataset, batch_size=4)
        dl = _setup_dataloader(params)
        assert isinstance(dl.batch_sampler, StratifiedBatchSampler)

    def test_mini_batch_yields_multi_class_batches(self, smoke_dataset):
        """Every mini-batch from stratified loader has >= 2 classes."""
        params = DataLoaderParams(dataset=smoke_dataset, batch_size=4)
        dl = _setup_dataloader(params)
        for _points, _covs, _weights, batch_labels in dl:
            if isinstance(batch_labels, torch.Tensor):
                unique_count = len(batch_labels.unique())
            else:
                unique_count = len(set(batch_labels))
            assert unique_count >= 2, f"Batch has only {unique_count} class(es)"

    def test_train_size_split(self, smoke_dataset):
        """train_size < 1.0 should produce a smaller dataset."""
        params = DataLoaderParams(dataset=smoke_dataset, batch_size=9999, train_size=0.5)
        dl = _setup_dataloader(params)
        total_samples = sum(len(batch_labels) for _p, _c, _w, batch_labels in dl)
        assert total_samples < len(smoke_dataset)
