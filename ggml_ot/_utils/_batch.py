"""Batch-level helpers: device transfer and stratified batch sampling.

Recursively moves nested structures (tensors, arrays, lists, dicts) to a
target ``torch.device`` so that any consumer can transfer a full dataloader
batch in one call.
"""

from __future__ import annotations

import numpy as np
import torch

from ggml_ot import settings


# ---------------------------------------------------------------------------
# Stratified batch sampler
# ---------------------------------------------------------------------------


class StratifiedBatchSampler:
    """Batch sampler with proportional class representation.

    For contrastive / triplet-loss training every mini-batch should contain
    samples from multiple classes so that valid *(same-class, different-class)*
    triplets can be formed.  This sampler draws a **proportional quota** from
    each class into every batch, closely preserving the global class
    frequencies.

    Notes
    -----
    * Actual batch sizes may differ from *batch_size* by a small rounding
      amount.
    * Once a minority-class pool is exhausted, later batches will lack that
      class.  Combine with an empty-triplet guard in the training loop to
      safely skip those batches.

    Parameters
    ----------
    labels : array-like
        Class labels, one per sample (numeric or string).
    batch_size : int
        Target number of samples per batch.
    shuffle : bool
        Shuffle within each class at the start of every iteration.
    """

    def __init__(self, labels, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        labels_arr = np.asarray(labels)
        self._total = len(labels_arr)

        # Group sample indices by class label
        self._class_to_indices: dict = {}
        for idx, lbl in enumerate(labels_arr):
            key = lbl.item() if hasattr(lbl, "item") else lbl
            self._class_to_indices.setdefault(key, []).append(idx)

        # Per-class quota: proportional to class frequency, at least 1
        self._quotas: dict = {}
        for cls, indices in self._class_to_indices.items():
            self._quotas[cls] = max(1, round(batch_size * len(indices) / self._total))

    # ------------------------------------------------------------------

    def __iter__(self):
        # Fresh shuffled copy of each class pool for this epoch
        pools: dict = {}
        for cls, indices in self._class_to_indices.items():
            arr = np.array(indices)
            if self.shuffle:
                settings.numpy_generator.shuffle(arr)
            pools[cls] = arr.tolist()

        classes = sorted(pools)

        while any(pools[c] for c in classes):
            batch: list[int] = []
            for cls in classes:
                if not pools[cls]:
                    continue
                n = min(self._quotas[cls], len(pools[cls]))
                batch.extend(pools[cls][:n])
                pools[cls] = pools[cls][n:]

            if self.shuffle:
                settings.numpy_generator.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        # The actual batch size is the sum of per-class quotas, so we
        # estimate the number of batches from the largest class pool.
        # This is an *estimate* — the true batch count depends on
        # per-class pool exhaustion order and may differ by ±1.  Currently
        # only ``tqdm`` consumes this value (progress-bar length).  If a
        # consumer that relies on an exact count is added (e.g. LR
        # schedulers, pre-allocated result buffers), revisit and compute the
        # exact value by simulating the yield loop or tracking actual yields.
        effective_batch = sum(self._quotas.values())
        if effective_batch <= 0:
            return 1
        return max(1, (self._total + effective_batch - 1) // effective_batch)


def move_batch_to_device(
    obj,
    device: torch.device | None = None,
    non_blocking: bool | None = None,
    dtype: torch.dtype | None = None,
):
    """Recursively move a batch to the selected device.

    Parameters
    ----------
    obj
        A tensor, numpy array, or nested list / tuple / dict of them.
    device
        Target device.  Falls back to :data:`ggml_ot.settings.device`.
    non_blocking
        Whether to use non-blocking transfers.  Defaults to ``True``
        when the target device is CUDA.
    dtype
        Optional dtype cast applied together with the device transfer.

    Returns
    -------
    The same structure with every leaf tensor on *device*.
    """
    device = settings.device if device is None else device
    non_blocking = (device.type == "cuda") if non_blocking is None else non_blocking

    if isinstance(obj, torch.Tensor):
        if obj.device == device and (dtype is None or obj.dtype == dtype):
            return obj
        kwargs = {"device": device, "non_blocking": non_blocking}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return obj.to(**kwargs)

    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj, device=device, dtype=dtype)

    if isinstance(obj, (list, tuple)):
        if all(np.isscalar(x) for x in obj):
            return torch.tensor(obj, device=device, dtype=dtype)
        if all(isinstance(x, (torch.Tensor, np.ndarray)) for x in obj):
            shapes = [tuple(x.shape) for x in obj]
            if len(set(shapes)) == 1:
                return torch.stack([torch.as_tensor(x, dtype=dtype) for x in obj]).to(
                    device=device, non_blocking=non_blocking
                )
        return type(obj)(move_batch_to_device(x, device=device, non_blocking=non_blocking, dtype=dtype) for x in obj)

    if isinstance(obj, dict):
        return {
            k: move_batch_to_device(v, device=device, non_blocking=non_blocking, dtype=dtype) for k, v in obj.items()
        }

    raise TypeError(f"Unsupported batch element type: {type(obj).__name__}")


# ---------------------------------------------------------------------------
# DataLoader factory (moved from optimization/_ggml to reduce bloat)
# ---------------------------------------------------------------------------


def _setup_dataloader(params, device: torch.device | None = None):
    """Build a training DataLoader, using stratified batching when needed.

    When ``batch_size < len(dataset)`` a :class:`StratifiedBatchSampler` is
    used so that every mini-batch preserves the global class frequencies,
    giving each batch a good chance of containing >= 2 classes (required for
    contrastive triplet formation).

    Parameters
    ----------
    params : DataLoaderParams
        Carries *dataset*, *batch_size*, and optional *train_size*.
    device : torch.device, optional
        Target device; used only for ``pin_memory`` heuristic.
    """
    pin_memory = (device.type == "cuda") if device is not None else (settings.device.type == "cuda")

    dataset = params.dataset
    if params.train_size is not None:
        split_indices = dataset._train_test_split(n_splits=1, train_size=params.train_size)
        dataset = dataset._subset(split_indices[0][0])

    # Single batch -> plain DataLoader (stratification is moot)
    if params.batch_size >= len(dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            generator=settings.torch_generator,
        )

    # Multi-batch -> stratified sampling to keep class proportions per batch
    sampler = StratifiedBatchSampler(
        labels=dataset.distribution_labels,
        batch_size=params.batch_size,
        shuffle=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        pin_memory=pin_memory,
    )


# OPTIMIZE: prefer using condensed-distance indexing in hot paths instead of
# materializing square matrices (e.g. via squareform) to reduce peak VRAM usage.
def get_condensed_pair_distance(dists: torch.Tensor, i: int, j: int, n: int) -> torch.Tensor:
    """Index one pair distance from a batched condensed-distance tensor.

    Parameters
    ----------
    dists
        Condensed distances with shape ``(B, n*(n-1)/2)``.
    i, j
        Pair indices in ``[0, n)``.
    n
        Number of points used to build the condensed representation.
    """
    if dists.ndim != 2:
        raise ValueError(f"Expected condensed distances with shape (B, L), got {tuple(dists.shape)}.")
    if i < 0 or j < 0 or i >= n or j >= n:
        raise IndexError(f"Pair indices out of bounds: i={i}, j={j}, n={n}.")
    if i == j:
        return torch.zeros(dists.shape[0], device=dists.device, dtype=dists.dtype)

    if i > j:
        i, j = j, i

    # Condensed upper-triangle index formula.
    k = int(n * i - (i * (i + 1) // 2) + j - i - 1)
    if k < 0 or k >= dists.shape[1]:
        raise IndexError(f"Computed condensed index {k} is out of bounds for shape {tuple(dists.shape)} and n={n}.")
    return dists[:, k]
