"""Label helpers for GGML datasets.

Central home for per-distribution label handling: resolving display/string
labels and any related label bookkeeping. Keep label-specific logic here
rather than scattering it across the dataset and AnnData modules.
"""

from __future__ import annotations

import numpy as np


def resolve_label_strings(distribution_labels_str, distribution_labels):
    """Resolve per-distribution string labels aligned to *distribution_labels*.

    Returns *distribution_labels_str* coerced to a string array when provided,
    falling back to stringified *distribution_labels* otherwise. Raises
    ``ValueError`` if an explicit array has a mismatched length.
    """
    if distribution_labels_str is None:
        return np.asarray([str(label) for label in distribution_labels], dtype=str)
    labels_str = np.asarray(distribution_labels_str, dtype=str)
    if len(labels_str) != len(distribution_labels):
        raise ValueError(
            f"distribution_labels_str length ({len(labels_str)}) must match "
            f"distribution_labels length ({len(distribution_labels)})."
        )
    return labels_str
