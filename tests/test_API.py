"""API surface tests.

Validates that the documented/imported public API exists and remains importable.
"""

import ggml_ot
import ggml_ot.optimization


def test_top_level():
    """Check top level user-facing exports in __all__ are present."""

    expected = [
        "train",
        "train_emd2",
        "train_sinkhorn",
        "train_gmm",
        "from_anndata",
        "from_numpy",
        "test",
        "train_test",
        "tune",
        "data",
        "pl",
        "gene",
        "gmm",
        "settings",
    ]

    for name in expected:
        assert hasattr(ggml_ot, name), f"Missing top-level export: {name}"
        assert name in ggml_ot.__all__, f"Missing from __all__: {name}"
