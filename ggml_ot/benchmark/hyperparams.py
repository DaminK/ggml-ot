from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from tqdm.contrib.itertools import product

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ggml_ot.data import AnnData_TripletDataset, TripletDataset

from ggml_ot.benchmark.evaluation import train_test


def tune(
    dataset: TripletDataset | AnnData_TripletDataset,
    alpha: float | list = [0.1, 1, 10],
    reg: float | list = [0.01, 0.1, 1, 10],
    reg_type: str | list = ["fro"],
    n_comps: int | list = [2, 5],
    mi_reg: float | list | None = None,
    knn_k: int = 5,
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
        A list or str of regularization type(s): `1` for L1, `2` or `"fro"` for L2/Frobenius,
        and `"nuc"` for nuclear norm.

    n_comps
        A list or int of number of components in the learned subspaces, i.e., rank of the subspace.

    mi_reg
        Optional mutual-information regularization strength(s). Can only be used when the dataset has covariances
        (i.e. a GMM dataset). If not provided, the behavior is unchanged and the default from
        :meth:`ggml_ot.train_test`/training is used.

    knn_k
        Number of neighbors used for benchmark k-NN classification during each
        train/test evaluation.

    print_latex
        Whether to print the hyperparameter tuning results as a LaTeX table.

    plot_contour
        Plot hyperparameter tuning results over alpha and reg for best n_comps and reg_type. You can also manually create contour plots from the returned dataframe using :meth:`ggml_ot.pl.contour_hyperparams`

    verbose
        Whether to print progress information during training.

    return_dataset
        If False, returns a tuple containing the results of the hyperparameter tuning.

        If True, returns the dataset with the best performing ground metric assigned to `dataset.map_A`. If the dataset is of type `AnnData_TripletDataset`, the cells are projected into the learned gene subspace (`dataset.adata.obsm["X_ggml"]`) and the loadings of the gene subspace are stored in `dataset.adata.varm["W_ggml"]`.

    **kwargs
        Additional arguments passed to :meth:`ggml_ot.train_test`.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        If `return_dataset` is set to False, a tuple is returned containing:
        - A dictionary mapping hyperparameter combinations to the best performing ground metric for that combination.
        - A DataFrame summarizing the mean and standard deviation of the evaluation metrics across test splits for each hyperparameter combination.

    TripletDataset | AnnData_TripletDataset
        If `return_dataset` is set to True, the dataset is returned with the best performing ground metric (`dataset.map_A`).

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

    # MI regularization is only defined for datasets with covariances (GMMs).
    if mi_reg is not None:
        if getattr(dataset, "covariances", None) is None:
            raise ValueError("`mi_reg` can only be used when `dataset.covariances` is not None (GMM dataset).")
        if not isinstance(mi_reg, list) and not isinstance(mi_reg, np.ndarray):
            mi_reg = [mi_reg]

    # Change default values for train_test
    kwargs.setdefault("plot_split", False)
    kwargs.setdefault("print_table", False)
    if kwargs.pop("print_table", False):
        warnings.warn(
            "`tune` outputs a formatted table after hyperparameter tuning. Setting `print_table=True` will output a separate table for each hyperparameter combination during tuning. This is not recommended for `tune` as it may clutter the output."
        )

    scores, Ws = {}, {}

    # Grid search over hyperparameter combinations
    if mi_reg is None:
        for k, n, a, r in product(n_comps, reg_type, alpha, reg, desc="Hyperparameter grid search"):
            param_W, param_scores = train_test(
                dataset,
                alpha=a,
                reg=r,
                reg_type=n,
                n_comps=k,
                return_dataset=False,
                knn_k=knn_k,
                print_table=False,
                verbose=verbose,
                **kwargs,
            )

            scores[(k, n, a, r)] = param_scores
            Ws[(k, n, a, r)] = param_W["best"]
    else:
        for k, n, a, r, mi in product(n_comps, reg_type, alpha, reg, mi_reg, desc="Hyperparameter grid search"):
            param_W, param_scores = train_test(
                dataset,
                alpha=a,
                reg=r,
                reg_type=n,
                n_comps=k,
                mi_reg=mi,
                return_dataset=False,
                knn_k=knn_k,
                print_table=False,
                verbose=verbose,
                **kwargs,
            )

            scores[(k, n, a, r, mi)] = param_scores
            Ws[(k, n, a, r, mi)] = param_W["best"]

    # Compile results into a DataFrame
    index_names = ["n_comps", "reg_type", "alpha", "reg"]
    if mi_reg is not None:
        index_names.append("mi_reg")

    df_scores = pd.concat(
        {params: score for params, score in scores.items()},
        axis=0,
        names=index_names,
    )

    # `train_test(..., return_table=True)` returns a table with its own index (e.g. per-split).
    # Drop those extra levels after `pd.concat(..., axis=0)` and only keep hyperparameter levels.
    if df_scores.index.nlevels > len(index_names):
        df_scores = df_scores.droplevel(list(range(len(index_names), df_scores.index.nlevels)), axis=0)

    # Format and display results
    from ggml_ot.plot import contour_hyperparams, table

    table(df_scores, style_performance=True, print_latex=print_latex, title="Hyperparameter tuning")

    # Compute best configuration when needed for contour plotting and/or return_dataset.
    best_index = None
    if plot_contour or return_dataset:
        best_index = df_scores[("knn", "mean")].idxmax()

    # Plot hyperparameter tuning results over alpha and reg for best n_comps and reg_type
    if plot_contour:
        contour_hyperparams(
            df_scores,
            x="alpha",
            y="reg",
            fixed_params=(
                {"n_comps": best_index[0], "reg_type": best_index[1]}
                if mi_reg is None
                else {"n_comps": best_index[0], "reg_type": best_index[1], "mi_reg": best_index[4]}
            ),
            value_col=("knn", "mean"),
            log_axis=True,
            levels=20,
        )

    if return_dataset:
        dataset.map_A = Ws[best_index]
        return dataset
    else:
        return Ws, df_scores
