import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
from tqdm import tqdm
from typing import Literal
import warnings

from ggml_ot.data import AnnData_TripletDataset, TripletDataset
from ggml_ot.benchmark.cluster import hierachical_clustering
from ggml_ot.benchmark.classify import knn
from ggml_ot.plot import table
from ggml_ot import train


def train_test(
    dataset: TripletDataset | AnnData_TripletDataset,
    n_splits: int = 5,
    train_size: float = 0.6,
    test_size: float | None = None,
    plot_split: bool = True,
    plot_type: Literal["clustermap_emb", "clustermap", "emb"] | bool = True,
    print_table: bool = True,
    print_latex: bool = False,
    return_dataset: bool = False,
    ground_metric: np.ndarray | None = None,
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
    plot_split
        Whether to plot OT distances for each split
    plot_type
        Defines which plots to generate, see :mod:`ggml.pl` for details.
    print_table
        Whether to print the results table
    print_latex
        Whether to print the results table in LaTeX format
    return_table
        Whether to return the results table
    return_dataset
        If False, returns a dict containing the trained ground metrics and a dataframe of the test scores.
        If True, returns the dataset with projected data using the best learned ground_metric.

        .. attention:: `return_dataset=True` only works if ground metric is learned (default: `ground_metric=None`)
    ground_metric
        If provided, this ground_metric is used for testing. You are encouraged to use :meth:`ggml_ot.test` instead.
    **kwargs
        Additional arguments passed to :meth:`ggml_ot.train`, see the corresponding docs for details.

    Returns
    -------
    TripletDataset | AnnData_TripletDataset
        If `return_dataset` is set to True, the dataset is returned with the best performing ground metric (`dataset.w_theta`).

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
            train_dataset = dataset.subset(split_indices[i][0])
            w, time = train(train_dataset, measure_time=True, return_dataset=False, plot_iter=False, **kwargs)
            times.append(time)
        else:
            w = ground_metric
        Ws.append(w)

        # Test
        scores.append(_test(dataset, split_indices[i], w, plot_split=plot_split, plot=plot_type, **kwargs))

    # list of dicts to dicts of list
    scores = {metric: [dic[metric] for dic in scores] for metric in scores[0]}
    if len(times) > 0:
        scores["epoch_time"] = np.asarray(times)

    df_scores = pd.concat(
        {
            metric: pd.DataFrame({"Mean": [np.mean(scores[metric])]} | {"SD": [np.std(scores[metric])]})
            for metric in scores
        },
        axis=1,
        names=["Metric", "Mean±SD"],
    )

    if print_table:
        # Print Average
        styler = df_scores.style.set_caption(f"Results ({n_splits} splits)")  # .hide(axis="index")
        table(styler, style_performance=True, print_latex=print_latex)

    if not isinstance(ground_metric, str):
        Ws_dict = {
            "Ws": Ws,
            "best": Ws[np.argmax(scores["knn_acc"])],
            "mean": np.mean(np.asarray(Ws), axis=0),
            "sd": np.std(np.asarray(Ws), axis=0),
        }
    else:
        Ws_dict = None

    if return_dataset:
        dataset.w_theta = Ws_dict["best"]
        if isinstance(dataset, AnnData_TripletDataset):
            dataset.project(Ws_dict["best"])
        return dataset
    else:
        return Ws_dict, df_scores


# Dynamically add to class to avoid circular imports
TripletDataset.train_test = train_test


def test(
    dataset: TripletDataset | AnnData_TripletDataset, ground_metric: np.ndarray | str | None = None, *args, **kwargs
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
        Ground metric to use for testing. If None (default), tries to use `dataset.w_theta`. You can also explicitly provide a ground metric trained with :meth:`ggml_ot.train` as a numpy array.

        To use a fixed metric provide the metric name as a string (e.g. "euclidean","cosine"), see [scipy.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) for supported metrics.

        .. warning:: If no ground_metric is provided and dataset has not been trained, this function will issue a warning and train a ground metric for each split. If you want to train and test ground metrics, you are encouraged to directly use :meth:`ggml_ot.train_test`.

    args, kwargs
        Additional arguments passed to :meth:`ggml_ot.train_test`. Internally, this function calls :meth:`ggml_ot.train_test` with the provided ground metric and skips training.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing the mean and standard deviation of the evaluation metrics across test splits.

        .. note:: While this function can be used to train a ground metric, it only does so for evaluation purposes and does not return the trained metric. For training ground metrics for later use, please use :meth:`ggml_ot.train` or :meth:`ggml_ot.train_test`.
    """
    if ground_metric is None:
        ground_metric = dataset.w_theta

    kwargs["return_dataset"] = False

    _, score_df = train_test(dataset, *args, ground_metric=ground_metric, **kwargs)
    return score_df


# Dynamically add to class to avoid circular imports
TripletDataset.test = test


def _test(dataset, split_index, w_theta, plot_split=False, plot=True, n_threads=128, **kwargs):
    train_index, test_index = split_index
    """ Internal helper function to test a ground metric on a train-test split"""

    # Compute OT distances
    train_test_set = dataset.subset(np.concatenate((train_index, test_index)))

    train_symbols = ["train"] * len(train_index) + ["test"] * len(test_index)

    ot_distances = train_test_set.compute_OT(
        ground_metric=w_theta,
        symbols=train_symbols,
        plot=plot if plot_split else False,
        n_threads=n_threads,
        **kwargs,
    )

    # Classification
    # TODO support confusion matrix from returned predicted_labels
    knn_acc, _ = knn(
        ot_distances,
        train_test_set.distribution_labels,
        np.arange(len(train_index)),
        np.arange(len(test_index)) + len(train_index),
    )

    # Clustering
    mi, ari, vi = hierachical_clustering(ot_distances, train_test_set.distribution_labels)

    return {"knn_acc": knn_acc, "mi": mi, "ari": ari, "vi": vi}


def _train_test_split(dataset, n_splits=10, train_size=0.8, test_size=None, validation_size=0):
    """Internal function to generate stratified train-test(-validation) splits

    :param dataset: number of re-shuffling and splitting iterations, defaults to 10
    :type dataset: ggml_ot.TripletDataset
    :param n_splits: number of re-shuffling and splitting iterations, defaults to 10
    :type n_splits: int, optional
    :param train_size: proportion of the dataset to include in the train split, defaults to 0.8
    :type train_size: float, optional
    :param test_size: proportion of dataset to include in the test split, defaults to 1 - train_size
    :type test_size: float, optional
    :param validation_size: proportion of dataset to include in the validation split, defaults to 0
    :type validation_size: float, optional
    :return: indices of train, test data of each split
    :rtype: array-like of tuples
    """
    if validation_size > 0:
        # TODO: draw validation inds in test split and later split into two test sets
        warnings.warn("Validation split not implemented yet")

    if test_size is not None and round(train_size + test_size, 2) != 1.00:
        # test_size passed, but doesn't sum to 1. Assume train_size was left to default value
        train_size = 1.0 - test_size

    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=42)

    return list(skf.split(np.zeros(len(dataset.distribution_labels)), dataset.distribution_labels))

    indices = np.arange(len(dataset.distribution_labels))
    split_indices = []

    for train_idx, test_idx in skf.split(indices, dataset.distribution_labels):
        split_indices.append((train_idx.tolist(), test_idx.tolist()))

    return split_indices


# Dynamically add to class to avoid circular imports with ggml_ot.data.generic.py, ggml_ot.ggml.py and this file
TripletDataset._train_test_split = _train_test_split
