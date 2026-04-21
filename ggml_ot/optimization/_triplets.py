"""Triplet-index helpers for optimization losses."""

from __future__ import annotations

import warnings

import torch


def warn_empty_triplets(
    triplets_idx: torch.Tensor,
    labels,
    *,
    epoch: int | None = None,
    batch_idx: int | None = None,
) -> bool:
    """Return *True* and emit a warning when *triplets_idx* is empty.

    A batch may contain only a single class (especially the last mini-batch)
    which makes triplet formation impossible.  This helper centralises the
    check so that calling code stays concise.
    """
    if triplets_idx.shape[0] > 0:
        return False

    n_unique = len(labels.unique()) if isinstance(labels, torch.Tensor) else len(set(labels))
    loc = ""
    if epoch is not None:
        loc += f"Epoch {epoch}"
    if batch_idx is not None:
        loc += f", batch {batch_idx}" if loc else f"Batch {batch_idx}"
    if loc:
        loc += ": "

    warnings.warn(
        f"{loc}no valid triplets (batch has {len(labels)} sample(s) across {n_unique} class(es)). Skipping batch.",
        stacklevel=3,
    )
    return True


def get_unique_triplet_pairs(triplets_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique sorted `(i, j)` pairs from `(i, j, k)` triplets."""
    pair_ij = triplets_idx[:, [0, 1]]
    pair_jk = triplets_idx[:, [1, 2]]
    all_pairs = torch.cat([pair_ij, pair_jk], dim=0)
    all_pairs_sorted, _ = torch.sort(all_pairs, dim=1)
    return torch.unique(all_pairs_sorted, dim=0, return_inverse=True)


def map_unique_pair_costs_to_triplets(
    unique_costs: torch.Tensor,
    inverse_indices: torch.Tensor,
    *,
    n_triplets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map unique pair costs back to per-triplet `(i,j)` and `(j,k)` costs."""
    idx_ij_mapped = inverse_indices[:n_triplets]
    idx_jk_mapped = inverse_indices[n_triplets:]
    return unique_costs[idx_ij_mapped], unique_costs[idx_jk_mapped]
