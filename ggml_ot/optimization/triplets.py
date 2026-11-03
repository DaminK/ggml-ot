"""Triplet construction helpers for GGML.

Categorical distribution-level triplets: same-class anchor/positive,
different-class negative.

The returned tensor follows the existing loss contract ``(i, j, k)`` where
pair ``(i, j)`` is the close pair, pair ``(j, k)`` is the far pair, and ``j``
is the shared anchor (see
:func:`ggml_ot.optimization.loss.batch_triplet_loss_emd2`).
"""

from __future__ import annotations

from typing import Literal

import torch


def create_triplets(labels, *, label_type: Literal["categorical"] = "categorical") -> torch.Tensor:
    """Construct categorical distribution-level triplets.

    Parameters
    ----------
    labels
        Per-distribution labels.
    label_type
        Only ``"categorical"`` is supported in this stable release; other
        values raise :class:`NotImplementedError`.

    Returns
    -------
    torch.Tensor
        ``(M, 3)`` long tensor of ``(i, j, k)`` index triples.
    """
    if label_type != "categorical":
        raise NotImplementedError(
            f"label_type={label_type!r} is not supported in this stable release; "
            "only label_type='categorical' is available."
        )
    return _create_categorical_triplets(labels)


def _create_categorical_triplets(labels) -> torch.Tensor:
    """Categorical triplets where ``c(i) == c(j)`` and ``c(j) != c(k)``."""
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    l_i = labels.view(-1, 1, 1)
    l_j = labels.view(1, -1, 1)
    l_k = labels.view(1, 1, -1)

    mask = (l_i == l_j) & (l_j != l_k)
    return mask.nonzero(as_tuple=False)
