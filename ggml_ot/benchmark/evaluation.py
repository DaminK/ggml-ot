from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from typing import Callable, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from ggml_ot.data import AnnData_TripletDataset, TripletDataset

from ggml_ot.benchmark._clustering import hierarchical_clustering, silhouette_score_wrapper
from ggml_ot.benchmark._knn import knn_classification
from ggml_ot.benchmark._metrics import _resolve_default_scoring, _scoring_metadata
from ..optimization.api import train


def train_test(
    dataset: TripletDataset | AnnData_TripletDataset,
    n_splits: int = 5,
    train_size: float = 0.6,
    test_size: float | None = None,
    scoring: list[str] | None = None,
    knn_k: int = 5,
    knn_weights: str = "distance",
    cluster_linkage: str | list[str] = "complete",
    split_strategy: Literal["auto", "shuffle", "stratified"] = "auto",
    plot_split: bool = True,
    plot_type: Literal["clustermap_embedding", "clustermap", "embedding"] | list[str] | tuple[str, ...] | bool = True,
    print_table: bool = True,
    print_latex: bool = False,
    return_dataset: bool = False,
    ground_metric: np.ndarray | str | callable | list | tuple | None = None,
    plot_split_dir: str | Path | None = None,
    plot_title: str | None = None,
    **kwargs,
) -> TripletDataset | AnnData_TripletDataset | tuple[dict, pd.DataFrame]:
    """Trains and cross-validates ground metrics on train-test splits.

    This function performs `n_splits` stratified train-test splits on the provided `dataset`.
    For each split, it trains a ground metric on the training set and evaluates it on the test set using a k-NN classification and hierarchical clustering.

    Classification accuracy and clustering metrics are summarized in a table, and results can be plotted as clustermap and embeddings.

    Parameters
    ----------
    dataset
        Dataset to perform train-test splits on.

        .. seealso:: The documentation for the provided interfaces to :meth:`AnnData <ggml_ot.from_anndata>` and :meth:`numpy arrays <ggml_ot.from_numpy>`.
    n_splits
        Number of train-test splits.
    train_size
        Proportion of dataset to include in train split.
    test_size
        Proportion of dataset to include in test split, if None 1 - train_size is used.
    scoring
        Evaluation scores to compute per test split. If ``None``, defaults
        are resolved from the dataset label type. In this stable release,
        categorical datasets default to ``["knn", "ari"]``. The first metric
        drives best-ground-metric selection and result-table highlighting.
        Supported keys:

        - ``"knn"`` — k-NN classification accuracy.
        - ``"ari"`` / ``"mi"`` / ``"vi"`` — clustering scores (Adjusted Rand
          Index, Mutual Information, Variation of Information) via
          hierarchical clustering.
        - ``"silhouette"`` — mean silhouette width using true labels and
          OT distances on the combined train/test set, independent of linkage.
    knn_k
        Number of neighbors used for benchmark k-NN classification.
    knn_weights
        ``weights`` argument forwarded to :class:`sklearn.neighbors.KNeighborsClassifier`.
        Default ``"distance"`` weights neighbors by proximity; pass ``"uniform"``
        for unweighted prediction.
    cluster_linkage
        Linkage method used by hierarchical clustering for clustering scores
        and clustermap plots. Defaults to ``"complete"``, the canonical choice
        for disease subtyping; produces more balanced cuts than ``"average"``
        and avoids singleton outlier clusters that can destabilize ARI.
        With a list, clustering scores are computed for each linkage: the first
        linkage is used for plots and the unsuffixed scores (e.g. ``"ari"``),
        each further linkage adds suffixed scores (e.g. ``"ari_complete"``).
    split_strategy
        Strategy to construct train/test folds. ``"auto"`` (default) uses
        stratified splitting when class sizes permit and falls back to
        shuffle otherwise. ``"shuffle"`` and ``"stratified"`` force the
        respective sklearn strategies.
    plot_split
        Whether to plot OT distances for each split
    plot_type
        Defines which plots to generate. One of
        ``"clustermap_embedding"``, ``"clustermap"``, ``"embedding"``, a list of
        those values, or ``False``.
    print_table
        Whether to print the results table
    print_latex
        Whether to print the results table in LaTeX format
    return_dataset
        If False, returns a dict containing the trained ground metrics and a dataframe of the test scores.
        If True, returns the dataset with projected data using the best learned ground_metric.

        .. attention:: `return_dataset=True` only works if ground metric is learned (default: `ground_metric=None`)
    ground_metric
        If provided, this ground_metric is used for testing. You are encouraged to use :meth:`ggml_ot.test` instead.
        A list or tuple gives one ground metric per split, in split order (e.g. the ``"Ws"`` returned by this function).
        The splits must match those the metrics were trained on: use the same ``n_splits``, ``train_size``,
        ``split_strategy`` and ``settings.random_seed``, otherwise test samples can leak into training.
        Pass a single matrix as an ``np.ndarray``; a nested list is read as one metric per split.
    plot_split_dir
        Optional output directory for per-split plots. When provided, split plots
        are saved into ``split_XX/`` subdirectories below this path.
    plot_title
        Base title used for split plots. When ``n_splits > 1``, the split name is
        appended automatically.
    **kwargs
        Additional arguments passed to :meth:`ggml_ot.train`, see the corresponding docs for details.

    Returns
    -------
    TripletDataset | AnnData_TripletDataset
        If `return_dataset` is set to True, the dataset is returned with the best performing ground metric (`dataset.map_A`).

        If the dataset is of type `AnnData_TripletDataset`, the cells are projected into the learned gene subspace (`dataset.adata.obsm["X_ggml"]`) and the loadings of the gene subspace are stored in `dataset.adata.varm["W_ggml"]`.

    tuple[dict, pd.DataFrame]
        If `return_dataset` is False, a tuple is returned containing:
            - A dict with keys:
                - "Ws": List of learned ground metrics for each split
                - "best": Best performing ground metric based on the first scoring metric
                - "mean": Mean ground metric across splits
                - "sd": Standard deviation of ground metrics across splits
            - A DataFrame summarizing the mean and standard deviation of evaluation metrics across splits.
              Its ``attrs`` mapping retains ``per_split`` metric values,
              ``split_train_idx``, ``split_test_idx``, and ``n_splits`` for
              downstream uncertainty analyses without changing the tabular
              return schema.

    """

    train_kwargs = {"plot_iter": 0, **kwargs, "measure_time": True, "return_dataset": False}
    return _train_test(
        dataset,
        train_fn=train,
        train_kwargs=train_kwargs,
        n_splits=n_splits,
        train_size=train_size,
        test_size=test_size,
        scoring=scoring,
        knn_k=knn_k,
        knn_weights=knn_weights,
        cluster_linkage=cluster_linkage,
        split_strategy=split_strategy,
        plot_split=plot_split,
        plot_type=plot_type,
        print_table=print_table,
        print_latex=print_latex,
        return_dataset=return_dataset,
        ground_metric=ground_metric,
        plot_split_dir=plot_split_dir,
        plot_title=plot_title,
        **kwargs,
    )


def train_test_external(
    dataset: TripletDataset | AnnData_TripletDataset,
    train_fn: Callable[..., tuple[np.ndarray | str | Callable, float]],
    train_kwargs: dict | None = None,
    n_splits: int = 5,
    train_size: float = 0.6,
    test_size: float | None = None,
    scoring: list[str] | None = None,
    knn_k: int = 5,
    knn_weights: str = "distance",
    cluster_linkage: str | list[str] = "complete",
    split_strategy: Literal["auto", "shuffle", "stratified"] = "auto",
    plot_split: bool = True,
    plot_type: Literal["clustermap_embedding", "clustermap", "embedding"] | list[str] | tuple[str, ...] | bool = True,
    print_table: bool = True,
    print_latex: bool = False,
    plot_split_dir: str | Path | None = None,
    plot_title: str | None = "External ground metric",
    **kwargs,
) -> tuple[dict, pd.DataFrame]:
    """Trains and cross-validates an external ground metric learning method on train-test splits.

    Use this to benchmark a ground metric learned by another method, e.g.
    Wasserstein Discriminant Analysis from POT, under exactly the splits and
    scores of :meth:`ggml_ot.train_test`. For each split, ``train_fn`` is called
    on the training distributions only and the ground metric it returns is
    evaluated on that split. Training once on the full dataset and passing the
    result to :meth:`ggml_ot.test` would instead expose the test labels.

    Parameters
    ----------
    dataset
        Dataset to perform train-test splits on.
    train_fn
        Called as ``train_fn(train_dataset, **train_kwargs)`` once per split.
        Returns ``(ground_metric, epoch_time)``, the same contract as
        :meth:`ggml_ot.train` with ``measure_time=True``: ``ground_metric`` is
        anything :meth:`ggml_ot.test` accepts (a projection array, a metric name
        or a callable projection), and ``epoch_time`` is the mean training time
        per iteration in seconds.
    train_kwargs
        Keyword arguments passed to ``train_fn``.
    n_splits, train_size, test_size, scoring, knn_k, knn_weights, cluster_linkage, split_strategy, plot_split, plot_type, print_table, print_latex, plot_split_dir, plot_title
        As in :meth:`ggml_ot.train_test`.
    **kwargs
        Additional arguments passed to the evaluation of each split.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        As :meth:`ggml_ot.train_test` with ``return_dataset=False``: a dict with
        the ground metric of each split (``"Ws"``) and the best one (``"best"``),
        plus ``"mean"`` and ``"sd"`` if the ground metrics are arrays, and a
        DataFrame of test scores with per-split values in its ``attrs``.
    """
    return _train_test(
        dataset,
        train_fn=train_fn,
        train_kwargs=train_kwargs or {},
        n_splits=n_splits,
        train_size=train_size,
        test_size=test_size,
        scoring=scoring,
        knn_k=knn_k,
        knn_weights=knn_weights,
        cluster_linkage=cluster_linkage,
        split_strategy=split_strategy,
        plot_split=plot_split,
        plot_type=plot_type,
        print_table=print_table,
        print_latex=print_latex,
        return_dataset=False,
        ground_metric=None,
        plot_split_dir=plot_split_dir,
        plot_title=plot_title,
        **kwargs,
    )


def _train_test(
    dataset,
    train_fn,
    train_kwargs,
    n_splits,
    train_size,
    test_size,
    scoring,
    knn_k,
    knn_weights,
    cluster_linkage,
    split_strategy,
    plot_split,
    plot_type,
    print_table,
    print_latex,
    return_dataset,
    ground_metric,
    plot_split_dir,
    plot_title,
    **kwargs,
):
    """Shared implementation of :func:`train_test` and :func:`train_test_external`.

    For each split, ``train_fn(train_dataset, **train_kwargs)`` trains a ground
    metric on the training distributions only, unless a fixed ``ground_metric``
    is given, which is then evaluated unchanged on every split. A list or tuple
    of ground metrics gives one metric per split.
    """
    scoring = _resolve_default_scoring(dataset, scoring)
    best_score, lower_is_better = _scoring_metadata(scoring)

    split_indices = dataset._train_test_split(n_splits, train_size, test_size, split_strategy=split_strategy)
    actual_n_splits = len(split_indices)
    per_split_metrics = isinstance(ground_metric, (list, tuple))
    if per_split_metrics and len(ground_metric) != actual_n_splits:
        raise ValueError(f"ground_metric holds {len(ground_metric)} per-split metrics for {actual_n_splits} splits.")
    from ggml_ot.plot._utils import _resolve_plot_title

    plot_titles = [
        _resolve_plot_title(
            plot_title,
            ground_metric,
            i,
            actual_n_splits,
            split_strategy=split_strategy,
            label_type=dataset.label_type,
        )
        for i in range(actual_n_splits)
    ]

    Ws, scores, times = [], [], []

    split_range = (
        range(actual_n_splits)
        if (ground_metric is None or actual_n_splits == 1)
        else tqdm(range(actual_n_splits), desc="Train/Test Splits")
    )

    for i in split_range:
        # Train on this split's training distributions only
        if ground_metric is None:
            train_dataset = dataset._subset(split_indices[i][0])
            w, time = train_fn(train_dataset, **train_kwargs)
            times.append(time)
        else:
            w = ground_metric[i] if per_split_metrics else ground_metric
        Ws.append(w)

        # Test
        scores.append(
            _test(
                dataset,
                split_indices[i],
                w,
                scoring=scoring,
                plot_split=plot_split,
                plot_type=plot_type,
                split_idx=i,
                n_splits=actual_n_splits,
                knn_k=knn_k,
                knn_weights=knn_weights,
                cluster_linkage=cluster_linkage,
                plot_split_dir=plot_split_dir,
                title=plot_titles[i],
                **kwargs,
            )
        )

    # list of dicts to dicts of list
    scores = {metric: [dic[metric] for dic in scores] for metric in scores[0]}
    if len(times) > 0:
        scores["epoch_time(s)"] = np.asarray(times)

    df_scores = pd.concat(
        {
            metric: pd.DataFrame({"mean": [np.mean(scores[metric])]} | {"SD": [np.std(scores[metric])]})
            for metric in scores
        },
        axis=1,
        names=["metric", "mean±SD"],
    )
    # Keep the reduced table backward-compatible while making repeated-split
    # results available to callers that need uncertainty estimates or paired
    # method comparisons. Lists (rather than NumPy arrays) keep the metadata
    # straightforward to serialize in downstream pipelines.
    df_scores.attrs["per_split"] = {metric: [float(value) for value in values] for metric, values in scores.items()}
    df_scores.attrs["n_splits"] = int(actual_n_splits)
    df_scores.attrs["split_train_idx"] = [np.asarray(train_idx, dtype=int).tolist() for train_idx, _ in split_indices]
    df_scores.attrs["split_test_idx"] = [np.asarray(test_idx, dtype=int).tolist() for _, test_idx in split_indices]

    if print_table:
        from ggml_ot.plot import table

        # Print Average
        styler = df_scores.style.set_caption(f"Results ({actual_n_splits} splits)")  # .hide(axis="index")
        table(
            styler,
            style_performance=True,
            print_latex=print_latex,
            best_metric=best_score,
        )

    if isinstance(ground_metric, str) or callable(ground_metric):
        # Fixed ground metric evaluated on all splits: no per-split ground metrics to return.
        Ws_dict = None
    else:
        best_fn = np.argmin if lower_is_better else np.argmax
        Ws_dict = {"Ws": Ws, "best": Ws[best_fn(scores[best_score])]}
        # Mean and SD are defined for array-valued ground metrics only, not for
        # callables such as the projections returned by external methods.
        if not any(isinstance(w, str) or callable(w) for w in Ws):
            Ws_dict["mean"] = np.mean(np.asarray(Ws), axis=0)
            Ws_dict["sd"] = np.std(np.asarray(Ws), axis=0)

    if return_dataset:
        dataset.map_A = Ws_dict["best"]
        return dataset
    else:
        return Ws_dict, df_scores


def test(
    dataset: TripletDataset | AnnData_TripletDataset,
    ground_metric: np.ndarray | str | callable | list | tuple | None = None,
    *args,
    knn_k: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """Tests ground metric on a given dataset.

    This function evaluates a provided ground metric on `n_splits` stratified train-test splits using k-NN classification and hierarchical clustering.
    For each split, the ground metric is evaluated on the test set using a k-NN classification and hierarchical clustering.

    Classification accuracy and clustering metrics are summarized in a table, and visualizations of the results are plotted.

    Parameters
    ----------
    dataset
        Dataset to perform cross-validation on.

         .. seealso:: The documentation for the provided interfaces to :meth:`AnnData <ggml_ot.from_anndata>` and :meth:`numpy arrays <ggml_ot.from_numpy>`.

    ground_metric
        Ground metric to use for testing. If None (default), tries to use `dataset.map_A`. You can also explicitly provide a ground metric trained with :meth:`ggml_ot.train` as a numpy array.

        To use a fixed metric provide the metric name as a string (e.g. "euclidean","cosine"), see [scipy.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) for supported metrics.

        A list or tuple gives one ground metric per split, in split order, e.g. the ``"Ws"`` returned by :meth:`ggml_ot.train_test` or metrics trained separately on the same splits. The splits must match those the metrics were trained on: use the same ``n_splits``, ``train_size``, ``split_strategy`` and ``settings.random_seed``, otherwise test samples can leak into training. Pass a single matrix as an ``np.ndarray``; a nested list is read as one metric per split.

        .. warning:: If no ground_metric is provided and dataset has not been trained, this function will issue a warning and train a ground metric for each split. If you want to train and test ground metrics, you are encouraged to directly use :meth:`ggml_ot.train_test`.

    knn_k
        Number of neighbors used for benchmark k-NN classification.

    args, kwargs
        Additional arguments passed to :meth:`ggml_ot.train_test`. Internally, this function calls :meth:`ggml_ot.train_test` with the provided ground metric and skips training.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing the mean and standard deviation of the evaluation metrics across test splits.

        .. note:: While this function can be used to train a ground metric, it only does so for evaluation purposes and does not return the trained metric. For training ground metrics for later use, please use :meth:`ggml_ot.train` or :meth:`ggml_ot.train_test`.
    """
    if ground_metric is None:
        ground_metric = dataset.map_A

    kwargs["return_dataset"] = False

    _, score_df = train_test(dataset, *args, ground_metric=ground_metric, knn_k=knn_k, **kwargs)
    return score_df


def _test(
    dataset,
    split_index,
    map_A,
    scoring=["knn", "ari"],
    plot_split=False,
    plot_type=True,
    time_inference=False,
    split_idx=0,
    n_splits=1,
    knn_k=5,
    knn_weights="distance",
    cluster_linkage="complete",
    plot_split_dir=None,
    title=None,
    **kwargs,
):
    train_index, test_index = split_index
    """ Internal helper function to test a ground metric on a train-test split"""
    # The first linkage is used for plots and the unsuffixed clustering scores
    linkages = [cluster_linkage] if isinstance(cluster_linkage, str) else list(cluster_linkage)
    if not linkages:
        raise ValueError("cluster_linkage must name at least one linkage method.")

    # Compute OT distances
    train_test_set = dataset._subset(np.concatenate((train_index, test_index)))

    train_symbols = ["train"] * len(train_index) + ["test"] * len(test_index)

    ot_distances = train_test_set.compute_OT(
        ground_metric=map_A,
        symbols=train_symbols,
        plot_type=False,
        measure_time=time_inference,
        **kwargs,
    )
    if time_inference:
        ot_distances, total_time = ot_distances

    if isinstance(ot_distances, torch.Tensor):
        ot_distances = ot_distances.cpu().numpy()

    # Clustermap & Embedding
    if plot_split and plot_type:
        from ggml_ot.plot._utils import _save_train_test_split_plots

        _save_train_test_split_plots(
            ot_distances,
            train_test_set.distribution_labels_plot,
            train_symbols,
            plot_type,
            split_idx=split_idx,
            n_splits=n_splits,
            plot_split_dir=plot_split_dir,
            title=title,
            cluster_linkage=linkages[0],
            label_type=train_test_set.label_type,
        )

    scores = {}

    # Classification
    if {"knn"} & set(scoring):
        # FUTURE: support confusion matrix from returned predicted_labels
        scores["knn"], _ = knn_classification(
            ot_distances,
            train_test_set.distribution_labels_str,
            np.arange(len(train_index)),
            np.arange(len(test_index)) + len(train_index),
            n_neighbors=knn_k,
            weights=knn_weights,
        )

    # Clustering
    if {"mi", "ari", "vi"} & set(scoring):
        for i, linkage in enumerate(linkages):
            _cluster_scores = hierarchical_clustering(ot_distances, train_test_set.distribution_labels, linkage=linkage)
            suffix = "" if i == 0 else f"_{linkage}"
            scores.update({k + suffix: _cluster_scores[k] for k in _cluster_scores if k in scoring})

    if "silhouette" in scoring:
        scores["silhouette"] = silhouette_score_wrapper(ot_distances, train_test_set.distribution_labels)

    # Inference time: total full OT computation and per computed off-diagonal pair.
    if time_inference:
        n_distributions = len(train_index) + len(test_index)
        n_pairs = n_distributions * (n_distributions - 1) / 2
        scores["full_ot_time(s)"] = total_time
        scores["inf_time(ms)"] = 1000 * total_time / n_pairs

    return scores
