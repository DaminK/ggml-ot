from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from typing import Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from ggml_ot.data import AnnData_TripletDataset, TripletDataset

from ggml_ot.benchmark.cluster import hierarchical_clustering
from ggml_ot.benchmark.classify import knn
from ..optimization.api import train


def train_test(
    dataset: TripletDataset | AnnData_TripletDataset,
    n_splits: int = 5,
    train_size: float = 0.6,
    test_size: float | None = None,
    scoring: tuple[str, ...] | list[str] = ("knn", "ari"),
    knn_k: int = 5,
    plot_split: bool = True,
    plot_type: Literal["clustermap_embedding", "clustermap", "embedding"] | list[str] | tuple[str, ...] | bool = True,
    print_table: bool = True,
    print_latex: bool = False,
    return_dataset: bool = False,
    ground_metric: np.ndarray | str | callable | None = None,
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
        Tuple or list of evaluation scores to compute on each test split.
        ``"knn"`` computes k-NN classification accuracy; ``"ari"`` (Adjusted Rand Index),
        ``"mi"`` (Mutual Information), and ``"vi"`` (Variation of Information)
        are clustering scores obtained via hierarchical clustering.
    knn_k
        Number of neighbors used for benchmark k-NN classification.
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
                - "best": Best performing ground metric based on k-NN accuracy
                - "mean": Mean ground metric across splits
                - "sd": Standard deviation of ground metrics across splits
            - A DataFrame summarizing the mean and standard deviation of evaluation metrics across splits.

    """

    split_indices = dataset._train_test_split(n_splits, train_size, test_size)

    Ws, scores, times = [], [], []

    enum = (
        range(n_splits) if (ground_metric is None or n_splits == 1) else tqdm(range(n_splits), desc="Train/Test Splits")
    )

    for i in enum:
        # Train
        if ground_metric is None:
            train_dataset = dataset._subset(split_indices[i][0])
            train_kwargs = dict(kwargs)
            train_kwargs.setdefault("plot_iter", 0)
            w, time = train(train_dataset, measure_time=True, return_dataset=False, **train_kwargs)
            times.append(time)
        else:
            w = ground_metric
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
                n_splits=n_splits,
                knn_k=knn_k,
                plot_split_dir=plot_split_dir,
                plot_title=plot_title,
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

    if print_table:
        from ggml_ot.plot import table

        # Print Average
        styler = df_scores.style.set_caption(f"Results ({n_splits} splits)")  # .hide(axis="index")
        table(styler, style_performance=True, print_latex=print_latex)

    if not isinstance(ground_metric, str) and not callable(ground_metric):
        best_score = scoring[0]
        best_fn = np.argmin if best_score == "vi" else np.argmax
        Ws_dict = {
            "Ws": Ws,
            "best": Ws[best_fn(scores[best_score])],
            "mean": np.mean(np.asarray(Ws), axis=0),
            "sd": np.std(np.asarray(Ws), axis=0),
        }
    else:
        Ws_dict = None

    if return_dataset:
        dataset.map_A = Ws_dict["best"]
        return dataset
    else:
        return Ws_dict, df_scores


def test(
    dataset: TripletDataset | AnnData_TripletDataset,
    ground_metric: np.ndarray | str | None = None,
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
    scoring=("knn", "ari"),
    plot_split=False,
    plot_type=True,
    time_inference=False,
    split_idx=0,
    n_splits=1,
    knn_k=5,
    plot_split_dir=None,
    plot_title=None,
    **kwargs,
):
    train_index, test_index = split_index
    """ Internal helper function to test a ground metric on a train-test split"""
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

        labels = getattr(train_test_set, "distribution_labels_str", train_test_set.distribution_labels)
        _save_train_test_split_plots(
            ot_distances,
            labels,
            train_symbols,
            plot_type,
            split_idx=split_idx,
            n_splits=n_splits,
            plot_split_dir=plot_split_dir,
            plot_title=plot_title,
            ground_metric=map_A,
        )

    scores = {}

    # Classification
    if {"knn"} & set(scoring):
        # FUTURE: support confusion matrix from returned predicted_labels
        scores["knn"], _ = knn(
            ot_distances,
            train_test_set.distribution_labels,
            np.arange(len(train_index)),
            np.arange(len(test_index)) + len(train_index),
            n_neighbors=knn_k,
        )

    # Clustering
    if {"mi", "ari", "vi"} & set(scoring):
        _cluster_scores = hierarchical_clustering(ot_distances, train_test_set.distribution_labels)
        scores.update({k: _cluster_scores[k] for k in _cluster_scores if k in scoring})

    # Inference time per pairwise distance
    if time_inference:
        scores["inf_time(ms)"] = (
            1000 * total_time / ((len(train_index) + len(test_index)) * (len(train_index) + len(test_index) + 1) / 2)
        )

    return scores
