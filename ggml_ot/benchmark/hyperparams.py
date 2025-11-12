import numpy as np
import pandas as pd
from tqdm.contrib.itertools import product

from ggml_ot.data import TripletDataset
from ggml_ot.plot import table, contour_hyperparams
from ggml_ot.benchmark.test import train_test
from ggml_ot.data.anndata import AnnData_TripletDataset


def tune(
    dataset: TripletDataset | AnnData_TripletDataset,
    alpha: float | list = [0.1, 1, 10],
    reg: float | list = [0.01, 0.1, 1, 10],
    reg_type: str | list = ["cos"],
    n_comps: int | list = [2, 5],
    print_latex: bool = False,
    plot_contour: bool = True,
    verbose: bool = False,
    return_dataset: bool = False,
    **kwargs,
):
    """Tune hyperparameters by performing a Grid Search and Cross-Validation.

    Parameters
    ----------
    dataset
        A dataset containing triplets of distributions.

        .. seealso:: The documentation for the provided interfaces to :meth:`AnnData <ggml_ot.from_anndata>` and :meth:`numpy arrays <ggml_ot.from_numpy>`.
    alpha
        A list or float of margin(s) between distributions from different classes (e.g. disease states). Large values lead to strong separations on the train set, but potential overfitting.

    reg
        A list or float of regularization strength(s).

    reg_type
        A list or str of type(s) of regularization, `1 | 2` use the corresponding matrx norm (oversimplified: use L1 for sparsity and L2 for robustness), `"cos"` uses the cosine similarity between subspace axis if you want to enforce orthogonality.

    n_comps
        A list or int of number of components in the learned subspaces, i.e., rank of the subspace.

    print_latex
        Whether to print the hyperparameter tuning results as a LaTeX table.

    plot_contour
        Plot hyperparameter tuning results over alpha and reg for best n_comps and reg_type. You can also manually create contour plots from the returned dataframe using :meth:`ggml_ot.pl.contour`

    verbose
        Whether to print progress information during training.

    return_dataset
        If False, returns a tuple containing the results of the hyperparameter tuning.

        If True, returns the dataset with the best performing ground metric assigned to `dataset.w_theta`. If the dataset is of type `AnnData_TripletDataset`, the cells are projected into the learned gene subspace (`dataset.adata.obsm["X_ggml"]`) and the loadings of the gene subspace are stored in `dataset.adata.varm["W_ggml"]`.

    **kwargs
        Additional arguments passed to :meth:`ggml_ot.train_test`.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        If `return_dataset` is set to False, a tuple is returned containing:
        - A dictionary mapping hyperparameter combinations to the best performing ground metric for that combination.
        - A DataFrame summarizing the mean and standard deviation of the evaluation metrics across test splits for each hyperparameter combination.

    TripletDataset | AnnData_TripletDataset
        If `return_dataset` is set to True, the dataset is returned with the best performing ground metric (`dataset.w_theta`).

        If the dataset is of type `AnnData_TripletDataset`, the cells are projected into the learned gene subspace (`dataset.adata.obsm["X_ggml"]`) and the loadings of the gene subspace are stored in `dataset.adata.varm["W_ggml"]`.

    """
    # Ensure all hyperparameters are lists
    if not isinstance(alpha, list) and not isinstance(alpha, np.ndarray):
        alpha = [alpha]
    if not isinstance(reg, list) and not isinstance(reg, np.ndarray):
        reg = [reg]
    if not isinstance(n_comps, list) and not isinstance(n_comps, np.ndarray):
        n_comps = [n_comps]
    if not isinstance(reg_type, list) and not isinstance(reg_type, np.ndarray):
        reg_type = [reg_type]

    # Change default values for train_test
    kwargs.setdefault("plot_split", False)
    kwargs.setdefault("print_table", False)
    if kwargs.pop("print_table", False):
        Warning.warn(
            "`tune` outputs a formatted table after hyperparameter tuning. Setting `print_table=True` will output a separate table for each hyperparameter combination during tuning. This is not recommended for `tune` as it may clutter the output."
        )

    scores, Ws = {}, {}

    # Grid search over hyperparameter combinations
    for k, n, a, r in product(n_comps, reg_type, alpha, reg, desc="Hyperparameter grid search"):
        param_W, param_scores = train_test(
            dataset,
            alpha=a,
            reg=r,
            reg_type=n,
            n_comps=k,
            return_table=True,
            return_dataset=False,
            print_table=False,
            verbose=verbose,
            **kwargs,
        )

        scores[(k, n, a, r)] = param_scores
        Ws[(k, n, a, r)] = param_W["best"]

    # Compile results into a DataFrame
    df_scores = pd.concat(
        {params: score for params, score in scores.items()},
        axis=0,
        names=["n_comps", "reg_type", "alpha", "reg"],
    )
    df_scores = df_scores.droplevel(4, axis=0)

    # Format and display results
    table(df_scores, style_performance=True, print_latex=print_latex, title="Hyperparameter tuning")

    # Plot hyperparameter tuning results over alpha and reg for best n_comps and reg_type
    if plot_contour:
        best_index = df_scores[("knn_acc", "Mean")].idxmax()
        contour_hyperparams(
            df_scores,
            x="alpha",
            y="reg",
            fixed_params={"n_comps": best_index[0], "reg_type": best_index[1]},
            value_col=("knn_acc", "Mean"),
            log_axis=True,
            levels=20,
        )

    if return_dataset:
        dataset.w_theta = Ws[best_index]
        if isinstance(dataset, AnnData_TripletDataset):
            dataset.project(dataset.w_theta)
        return dataset

    else:
        return Ws, df_scores


# Dynamically add to class to avoid circular imports with ggml_ot.data.generic.py, ggml_ot.ggml.py and this file
TripletDataset.tune = tune
