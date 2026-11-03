"""Tests for benchmark evaluation metrics."""

import numpy as np
import pytest
from sklearn.metrics import silhouette_score

import ggml_ot
from .utils.config import get_kwargs, get_synth_config


def test_silhouette_all_patients():
    """Reported silhouette must match a calculation on all patients, as for ARI."""
    dataset = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    metrics, scores = ggml_ot.train_test(dataset, **get_kwargs("smoke", "train_test", scoring=["silhouette"]))
    for metric, score in zip(metrics["Ws"], scores.attrs["per_split"]["silhouette"]):
        distances = dataset.compute_OT(ground_metric=metric, plot_type=False)
        expected = silhouette_score(distances.cpu().numpy(), dataset.distribution_labels, metric="precomputed")
        np.testing.assert_allclose(score, expected)


def test_silhouette_linkage_independent():
    """Changing hierarchical linkage must not change the silhouette score."""
    dataset = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    kwargs = get_kwargs("smoke", "test", scoring=["silhouette"], ground_metric="euclidean")
    complete = ggml_ot.test(dataset, cluster_linkage="complete", **kwargs)
    average = ggml_ot.test(dataset, cluster_linkage="average", **kwargs)
    np.testing.assert_allclose(complete.attrs["per_split"]["silhouette"], average.attrs["per_split"]["silhouette"])


def test_linkage_list_matches_single_linkages():
    """A list of linkages must reproduce the separate single-linkage scores, with the
    first linkage unsuffixed and further linkages suffixed."""
    dataset = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    kwargs = get_kwargs("smoke", "test", scoring=["ari", "mi"], ground_metric="euclidean")
    both = ggml_ot.test(dataset, cluster_linkage=["average", "complete"], **kwargs).attrs["per_split"]
    average = ggml_ot.test(dataset, cluster_linkage="average", **kwargs).attrs["per_split"]
    complete = ggml_ot.test(dataset, cluster_linkage="complete", **kwargs).attrs["per_split"]
    for metric in ("ari", "mi"):
        np.testing.assert_allclose(both[metric], average[metric])
        np.testing.assert_allclose(both[f"{metric}_complete"], complete[metric])


@pytest.mark.parametrize("metric_type", ["array", "callable"])
def test_train_test_external_matches_train_test(metric_type):
    """With a fixed ground metric, train_test_external must reproduce train_test on the
    same splits, and train_fn must only receive the training distributions of a split."""
    dataset = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    if metric_type == "array":
        ground_metric = np.eye(dataset.supports.shape[-1])
    else:

        def ground_metric(supports):
            return np.asarray(supports)

    def train_fixed(train_dataset):
        assert len(train_dataset.distribution_labels) < len(dataset.distribution_labels)
        return ground_metric, 0.0  # (ground metric, epoch time in seconds)

    kwargs = get_kwargs("smoke", "test", n_splits=2)
    _, expected = ggml_ot.train_test(dataset, ground_metric=ground_metric, **kwargs)
    metrics, scores = ggml_ot.benchmark.train_test_external(dataset, train_fixed, **kwargs)

    assert scores.attrs["split_train_idx"] == expected.attrs["split_train_idx"]
    for metric, values in expected.attrs["per_split"].items():
        np.testing.assert_allclose(scores.attrs["per_split"][metric], values)
    assert len(metrics["Ws"]) == 2
    # Mean and SD across splits exist for array-valued ground metrics only.
    assert ("mean" in metrics) == (metric_type == "array")


def test_per_split_ground_metrics():
    """A list of ground metrics must evaluate metric i on split i, as a single fixed
    metric would on that split; a list of the wrong length must be rejected."""
    dataset = ggml_ot.data.from_synth(**get_synth_config(), show=False)
    n_features = dataset.supports.shape[-1]
    metrics = [np.eye(n_features), np.diag(np.linspace(0.5, 2.0, n_features))]
    kwargs = get_kwargs("smoke", "test", n_splits=2, scoring=["knn", "ari"])
    per_split = ggml_ot.test(dataset, ground_metric=metrics, **kwargs).attrs["per_split"]
    for i, metric in enumerate(metrics):
        fixed = ggml_ot.test(dataset, ground_metric=metric, **kwargs).attrs["per_split"]
        for name, values in per_split.items():
            np.testing.assert_allclose(values[i], fixed[name][i])
    with pytest.raises(ValueError, match="per-split metrics"):
        ggml_ot.test(dataset, ground_metric=metrics[:1], **kwargs)


def _tuning_table(rows, names=("n_comps", "reg_type", "alpha", "reg", "mi_reg")):
    """Scores table shaped like the output of ggml_ot.tune: one row per combination."""
    import pandas as pd

    index = pd.MultiIndex.from_tuples([r[0] for r in rows], names=list(names))
    columns = pd.MultiIndex.from_tuples([("knn", "mean"), ("ari", "mean")])
    return pd.DataFrame([r[1] for r in rows], index=index, columns=columns)


def test_tie_break_rules():
    """Combinations tying on both scores go to the first in grid order by default, to the
    largest reg (then mi_reg), the largest mi_reg (then reg), or the largest log10 sum."""
    from ggml_ot.benchmark._metrics import _best_index_by_scoring

    table = _tuning_table(
        [
            ((5, "fro", 1, 1e-4, 1e-4), (0.9, 0.8)),
            ((5, "fro", 1, 1e-2, 1e-1), (0.9, 0.8)),
            ((5, "fro", 1, 1.0, 1e-2), (0.9, 0.8)),
            ((5, "fro", 1, 1e-4, 1.0), (0.9, 0.8)),
            ((5, "fro", 1, 10.0, 10.0), (0.8, 0.9)),  # higher ARI but lower KNN: never chosen
            ((5, "fro", 1, 100.0, 100.0), (0.9, 0.7)),  # same KNN, lower ARI: scores decide before weights
        ]
    )
    expected = {
        "first": (5, "fro", 1, 1e-4, 1e-4),
        "reg": (5, "fro", 1, 1.0, 1e-2),
        "mi_reg": (5, "fro", 1, 1e-4, 1.0),
        "both": (5, "fro", 1, 1.0, 1e-2),  # log10 sum -2 beats -3 and -4
    }
    for tie_break, best in expected.items():
        assert _best_index_by_scoring(table, ["knn", "ari"], tie_break=tie_break) == best
    assert _best_index_by_scoring(table, ["knn", "ari"]) == expected["first"]


def test_tie_break_without_mi_reg_level():
    """Without an mi_reg level (exact Bures tuning) every rule but "first" picks the largest
    reg; a weight of 0 counts as the smallest."""
    from ggml_ot.benchmark._metrics import _best_index_by_scoring

    table = _tuning_table(
        [((5, "fro", 1, 0.0), (0.9, 0.8)), ((5, "fro", 1, 0.1), (0.9, 0.8)), ((5, "fro", 1, 1e-3), (0.9, 0.8))],
        names=("n_comps", "reg_type", "alpha", "reg"),
    )
    for tie_break in ("reg", "mi_reg", "both"):
        assert _best_index_by_scoring(table, ["knn", "ari"], tie_break=tie_break) == (5, "fro", 1, 0.1)
    assert _best_index_by_scoring(table, ["knn", "ari"]) == (5, "fro", 1, 0.0)
