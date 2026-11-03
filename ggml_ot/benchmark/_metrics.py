"""Metric functions and scoring metadata for benchmark evaluation."""

import numpy as np

SCORING_LOWER_IS_BETTER: dict[str, bool] = {
    "knn": False,
    "ari": False,
    "mi": False,
    "vi": True,
    "silhouette": False,
}
"""Mapping from supported scoring metric name to whether lower values are better."""


def _resolve_default_scoring(dataset, scoring):
    """Return ``scoring`` if given, otherwise the default scoring metrics."""
    if scoring is not None:
        return scoring
    return ["knn", "ari"]


def _scoring_metadata(scoring):
    """Return the automatic selection metric and whether lower values are better."""
    best_score = scoring[0]
    return best_score, SCORING_LOWER_IS_BETTER[best_score]


TIE_BREAKS = ("first", "reg", "mi_reg", "both")
"""Rules for choosing among hyperparameter combinations that tie on the scoring metrics."""


def _index_level(index, name: str) -> np.ndarray:
    """Values of an index level as floats; zeros if the index has no such level."""
    if name not in (index.names or []):
        return np.zeros(len(index))
    return np.asarray(index.get_level_values(name), dtype=float)


def _tie_break_key(index, tie_break: str) -> list:
    """Sort key per index entry for ``tie_break`` (larger wins). A level the index does not have
    counts as zero, so ``"mi_reg"`` and ``"both"`` reduce to ``"reg"`` without an mi_reg level."""
    reg, mi_reg = _index_level(index, "reg"), _index_level(index, "mi_reg")
    if tie_break == "reg":
        return list(zip(reg, mi_reg))
    if tie_break == "mi_reg":
        return list(zip(mi_reg, reg))
    # a weight of 0 counts as the smallest on the log scale
    return list(np.log10(np.clip(reg, 1e-12, None)) + np.log10(np.clip(mi_reg, 1e-12, None)))


def _best_index_by_scoring(df, scoring, *, display_precision: int | None = None, tie_break: str = "first"):
    """Return the best row index, using the second scoring metric as a tie-breaker (if available).

    Combinations that still tie are resolved by ``tie_break``:

    - ``"first"``: the first in grid order (the smallest weights for ascending grids).
    - ``"reg"``: the largest ``reg``, then the largest ``mi_reg``.
    - ``"mi_reg"``: the largest ``mi_reg``, then the largest ``reg``.
    - ``"both"``: the largest log10 ``reg`` + log10 ``mi_reg``, i.e. the most regularized overall
      on the grid's log scale.

    Remaining ties go to the first in grid order.
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"Unknown tie_break {tie_break!r}, expected one of {TIE_BREAKS}.")
    candidates = df.index

    for metric in scoring[:2]:
        values = df.loc[candidates, (metric, "mean")].dropna()
        if values.empty:
            break

        comparable = values.round(display_precision) if display_precision is not None else values
        best_value = comparable.min() if SCORING_LOWER_IS_BETTER[metric] else comparable.max()
        candidates = comparable.index[comparable == best_value]
        if len(candidates) <= 1:
            break

    if tie_break != "first" and len(candidates) > 1:
        keys = _tie_break_key(candidates, tie_break)
        # max() returns the first of equal keys, so remaining ties keep grid order
        return candidates[max(range(len(keys)), key=keys.__getitem__)]
    return candidates[0]
