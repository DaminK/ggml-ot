"""Data splitting and subsampling helpers."""

from __future__ import annotations

import numpy as np


def split_by_group(
    x: np.ndarray,
    distribution_index: np.ndarray,
    distribution_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split a matrix into per-distribution blocks."""
    x_by_distribution: dict[str, np.ndarray] = {}
    for d_idx, dist in enumerate(distribution_ids):
        x_d = x[distribution_index == d_idx]
        if x_d.shape[0] == 0:
            raise ValueError(
                f"Distribution {dist!r} has no cells after subsampling. Increase subsample fraction or use full refit."
            )
        x_by_distribution[str(dist)] = x_d
    return x_by_distribution


def subsample_global(n: int, frac: float | None, rng: np.random.Generator) -> np.ndarray:
    """Return a sorted global subset of ``range(n)``."""
    if frac is None:
        return np.arange(n, dtype=int)
    if not (0 < frac <= 1):
        raise ValueError("subsample_frac must satisfy 0 < subsample_frac <= 1.")
    m = max(1, int(round(frac * n)))
    return np.sort(rng.choice(n, size=m, replace=False))


def subsample_stratified(
    distribution_index: np.ndarray,
    frac: float | None,
    rng: np.random.Generator,
    min_count_per_group: int = 1,
) -> np.ndarray:
    """Subsample indices per distribution group."""
    if frac is None:
        return np.arange(distribution_index.shape[0], dtype=int)
    if not (0 < frac <= 1):
        raise ValueError("subsample_frac must satisfy 0 < subsample_frac <= 1.")

    selected: list[np.ndarray] = []
    for d_idx in np.unique(distribution_index):
        group_idx = np.where(distribution_index == d_idx)[0]
        requested = int(round(frac * len(group_idx)))
        if requested < min_count_per_group:
            raise ValueError(
                f"subsample_frac={frac} selects too few cells for distribution "
                f"index {int(d_idx)}: {requested} < required {min_count_per_group}. "
                "Increase subsample_frac."
            )
        m = min(requested, len(group_idx))
        if m == len(group_idx):
            selected.append(group_idx)
        else:
            selected.append(rng.choice(group_idx, size=m, replace=False))

    return np.sort(np.concatenate(selected))


def split_train_val_indices(
    n: int,
    *,
    n_components: int,
    train_frac: float | None = None,
    val_frac: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create strict train/validation split indices with component-count safety."""
    if n < 2:
        raise ValueError("At least two samples are required to create a train/validation split.")

    if rng is None:
        rng = np.random.default_rng()

    if train_frac is None:
        if not (0 < val_frac < 1):
            raise ValueError("val_frac must satisfy 0 < val_frac < 1.")
        n_val = max(1, int(round(val_frac * n)))
        n_val = min(n_val, n - 1)
        n_train = n - n_val
    else:
        if not (0 < train_frac <= 1):
            raise ValueError("train_frac must satisfy 0 < train_frac <= 1.")
        requested_train = max(1, int(round(train_frac * n)))
        n_train = min(requested_train, n - 1)
        n_val = n - n_train

    if n_train < n_components:
        raise ValueError(
            "Training split is too small for the requested number of mixture components: "
            f"n_train={n_train} < n_components={n_components}. Increase the training split "
            "or reduce k_comps."
        )
    if n_val < 1:
        raise ValueError("Validation split must contain at least one sample.")

    perm = rng.permutation(n)
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    return train_idx, val_idx
