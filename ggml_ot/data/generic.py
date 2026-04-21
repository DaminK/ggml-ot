from __future__ import annotations
import inspect
from functools import update_wrapper

import numpy as np
import torch
from torch.utils.data import Dataset
import copy

from ggml_ot.distances import compute_OT
from ggml_ot._utils._array import not_none, to_float32
from ggml_ot.settings import settings

import warnings


class TripletDataset(Dataset):
    """Dataset to train GGML based on array data.

    This class stores a collection of distributions ("supports") and produces triplets
    (i, j, k) of relative relationships where i and j are from the same class and
    k is from a different class. These triplets are used to train GGML such that distributions
    i and j are closer to each other than j and k by some margin alpha.

    This class exposes the dataset to the standardized interfaces used by :meth:`ggml_ot.train`, :meth:`ggml_ot.tune`,
    :meth:`ggml_ot.test` and :meth:`ggml_ot.train_test`.

    Parameters
    ----------
    supports : Sequence[np.ndarray]
        Sequence of per-distribution supports. Each element is an array of points
        (for empirical distributions) or component means (for GMM-style representations).
    distribution_labels : Sequence[int] | np.ndarray
        Integer labels identifying the class/group of each distribution.
    n_triplets : int, optional
        Number of triplets to generate per "anchor" distribution (default: 3).
    weights : Sequence[np.ndarray] | None, optional
        Per-distribution probability weights (e.g., cluster proportions) or None for
        uniform weights (default: None).
    covariances : Sequence[np.ndarray] | None, optional
        Optional per-distribution covariance arrays when supports represent Gaussian
        mixture components (default: None).
    identical_supports : bool, optional
        If True, indicates that all distributions share the same supports
        (e.g., identical component locations across distributions). This changes the
        __getitem__ return format and allows faster OT evaluation (default: False).

    Notes
    -----
    - The class generates triplets by sampling t "positive" neighbors from the same
      class and t "negative" neighbors from each different class for every distribution.

    """

    supports: list[int] | np.ndarray = None
    "Stored supports."

    weights: list[int] | np.ndarray = None
    "Stored per-distribution weights (if provided)."

    distribution_labels = None
    "Integer class labels for each distribution."

    # triplets: list[tuple[int, int, int]] = None
    "Generated triplet index tuples used for training."

    dim: int = None
    "Dimensions of space underlying the distributions."

    identical_supports = None
    "Flag as passed to the constructor."

    _n_triplets = None
    _map_A = None
    covariances = None

    def __init__(
        self,
        supports,
        distribution_labels,
        n_triplets=3,
        weights=None,
        covariances=None,
        identical_supports=False,
        **kwargs,
    ):
        self.identical_supports = identical_supports
        self.dim = supports[0].shape[-1]

        self.supports = to_float32(supports, backend="torch")
        self.covariances = to_float32(covariances, backend="torch")
        self.weights = to_float32(weights, backend="torch")

        self.distribution_labels = distribution_labels
        # self.triplets = create_triplets(distribution_labels, n_triplets)
        self._n_triplets = n_triplets

        self._map_A = None

    @property
    def points(self):
        return np.concatenate(self.supports)

    @property
    def points_labels(self):
        """Returns list of the distribution_labels of all points concatenated over all distributions"""
        return np.array(
            sum(
                [[label] * len(support) for label, support in zip(self.distribution_labels, self.supports)],
                [],
            )
        )

    @property
    def distribution_labels_str(self):
        return self.distribution_labels

    @property
    def map_A(self):
        """Learned ground metric as a linear map (raises a warning if dataset is not trained yet)."""
        if self._map_A is None:
            warnings.warn("This dataset has not been trained yet, please call train() on this object first.")
        return self._map_A

    @map_A.setter
    def map_A(self, map_A):
        self._map_A = map_A

    @property
    def w_theta(self):
        """Deprecated: use ``map_A`` instead."""
        warnings.warn(
            "dataset.w_theta is deprecated and will be removed in v2.0. Use dataset.map_A instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.map_A

    @w_theta.setter
    def w_theta(self, map_A):
        warnings.warn(
            "dataset.w_theta is deprecated and will be removed in v2.0. Use dataset.map_A instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.map_A = map_A

    def __len__(self):
        """Returns the number of triplets.
        :return: number of triplets
        :rtype: int
        """
        return len(self.distribution_labels)

    def __getitem__(self, idx):
        """Returns a distribution and labels at position idx.
        :return: distribution and label
        :rtype: (supports, covariances, weights, labels)
        """

        if self.identical_supports:
            # Return weights and labels for distributions with identical_supports
            return (
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
                self.weights[idx],
                self.distribution_labels[idx],
            )
        else:
            # Return means (supports), covariances (if exists), weights (if exists) and labels of GMMs
            return (
                self.supports[idx],
                self.covariances[idx] if not_none(self.covariances) else torch.tensor([], dtype=torch.float32),
                self.weights[idx] if not_none(self.weights) else torch.tensor([], dtype=torch.float32),
                self.distribution_labels[idx],
            )

    def normalize(self):
        # supports have shape N, n, d
        global_mean = self.supports.mean(dim=(0, 1), keepdim=True)  # (1, 1, d)
        global_std = self.supports.std(dim=(0, 1), keepdim=True)  # (1, 1, d)

        epsilon = 1e-6
        scale = global_std + epsilon

        self.supports = (self.supports - global_mean) / scale

        if not_none(self.covariances):
            # We create a (1, 1, D, D) scaling matrix to broadcast over B and N
            # Outer product: scale_matrix[d_i, d_j] = scale[d_i] * scale[d_j]
            inv_scale_M = torch.diag_embed(scale.pow(-1))
            self.covariances = inv_scale_M @ self.covariances @ inv_scale_M

        return self

    def train(self, *args, **kwargs):
        """Train GGML on this dataset.

        Thin wrapper around :func:`ggml_ot.train`.
        Uses a lazy import to avoid circular imports.
        """
        from ..optimization.api import train as _train  # noqa: E402

        return _train(self, *args, **kwargs)

    def train_emd2(self, *args, **kwargs):
        """Train GGML with exact OT (EMD2) on this dataset.

        Thin wrapper around :func:`ggml_ot.train_emd2`.
        Uses a lazy import to avoid circular imports.
        """
        from ..optimization.api import train_emd2 as _train_emd2  # noqa: E402

        return _train_emd2(self, *args, **kwargs)

    def train_sinkhorn(self, *args, **kwargs):
        """Train GGML with Sinkhorn-regularized OT on this dataset.

        Thin wrapper around :func:`ggml_ot.train_sinkhorn`.
        Uses a lazy import to avoid circular imports.
        """
        from ..optimization.api import train_sinkhorn as _train_sinkhorn  # noqa: E402

        return _train_sinkhorn(self, *args, **kwargs)

    def train_test(self, *args, **kwargs):
        """Cross-validate GGML on train/test splits.

        Thin wrapper around :func:`ggml_ot.benchmark.evaluation.train_test`.
        Uses a lazy import to avoid circular imports.
        """

        from ..benchmark.evaluation import train_test as _train_test  # noqa: E402

        return _train_test(self, *args, **kwargs)

    def test(self, *args, **kwargs):
        """Evaluate a ground metric on this dataset.

        Thin wrapper around :func:`ggml_ot.benchmark.evaluation.test`.
        Uses a lazy import to avoid circular imports.
        """

        from ..benchmark.evaluation import test as _test  # noqa: E402

        return _test(self, *args, **kwargs)

    def tune(self, *args, **kwargs):
        """Tune hyperparameters by grid search + cross-validation.

        Thin wrapper around :func:`ggml_ot.benchmark.hyperparams.tune`.
        Uses a lazy import to avoid circular imports.
        """

        from ..benchmark.hyperparams import tune as _tune  # noqa: E402

        return _tune(self, *args, **kwargs)

    def fit_gmm(self, *args, **kwargs):
        """Fit a GMM representation for this dataset.

        Thin wrapper around :func:`ggml_ot.gmm.fit_gmm`.
        Uses a lazy import to avoid circular imports.
        """
        from ..gmm import fit_gmm as _fit_gmm  # noqa: E402

        return _fit_gmm(self, *args, **kwargs)

    def validate_gmm(self, *args, **kwargs):
        """Validate a fitted GMM on an AnnData-backed dataset.

        Thin wrapper around :func:`ggml_ot.gmm.validate_gmm`.
        Uses a lazy import to avoid circular imports.
        """
        from ..gmm.validation import validate_gmm as _validate_gmm  # noqa: E402

        return _validate_gmm(self, *args, **kwargs)

    def _train_test_split(self, n_splits=10, train_size=0.8, test_size=None, validation_size=0):
        """Generate stratified train-test(-validation) splits.

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
            warnings.warn("Validation split not implemented yet")

        if test_size is not None and round(train_size + test_size, 2) != 1.00:
            train_size = 1.0 - test_size

        from sklearn.model_selection import StratifiedShuffleSplit

        skf = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=settings.random_seed
        )
        return list(skf.split(np.zeros(len(self.distribution_labels)), self.distribution_labels))

    def _subset(self, indices):
        """Returns dataset subset for indices"""
        split_Dataset = copy.deepcopy(self)

        split_Dataset.distribution_labels = [self.distribution_labels[i] for i in indices]
        # split_Dataset.triplets = create_triplets(split_Dataset.distribution_labels, self._n_triplets)
        if not self.identical_supports:
            if not_none(self.supports):
                split_Dataset.supports = self.supports[indices]

                if not_none(self.covariances):
                    split_Dataset.covariances = self.covariances[indices]

            if not_none(self.weights):
                split_Dataset.weights = self.weights[indices]

            return split_Dataset
        else:
            split_Dataset.weights = self.weights[indices]
            return split_Dataset

    def compute_OT(
        self,
        precomputed_distances=None,
        ground_metric=None,
        legend="Side",
        plot_type: str | bool = "clustermap_embedding",
        symbols=None,
        measure_time=False,
        show: bool | None = None,
        save: str | bool | None = None,
        **kwargs,
    ):
        """Compute the Optimal Transport distances between all distributions.

        :param precomputed_distances: optional matrix of precomputed distances for computing the OT, defaults to None
        :type precomputed_distances: array-like, optional
        :param ground_metric: ground metric for OT computation, defaults to None
        :type ground_metric: "euclidean", "cosine", "cityblock", optional
        :param entropic_reg: pass ``entropic_reg > 0`` via ``**kwargs`` to use
            Sinkhorn-regularized OT instead of exact EMD2
        :type entropic_reg: float, optional
        :param w: weight matrix for the mahalanobis distance, defaults to None
        :type w: array-like, optional
        :param legend: defines where to place the legend, defaults to "Side"
        :type legend: "Top", "Side", optional
        :param plot_type: which visualisation to produce after computing OT distances.
            One of ``"clustermap_embedding"`` (default), ``"clustermap"``,
            ``"embedding"``, or ``False`` to skip plotting entirely.
        :type plot_type: str or bool, optional
        :param show: Whether to display the plot.  ``None`` (default) automatically
            shows in interactive environments (notebooks, IPython) and
            suppresses in scripts.  ``True``/``False`` override explicitly.
        :type show: bool or None, optional
        :param save: Whether to save the figure to disk.  ``None``/``False`` skip
            saving.  ``True`` saves under the default name into
            ``settings.figdir``.  A *str* is used as the filename.
        :type save: str, bool, or None, optional
        :return: pairwise OT distance matrix
        :rtype: numpy.ndarray
        """
        if "save_path" in kwargs:
            raise TypeError("`save_path` has been removed. Use `save=` and configure `ggml_ot.settings.figdir`.")

        # Back-compat: accept old `plot=` keyword
        if "plot" in kwargs:
            import warnings

            warnings.warn(
                "`plot=` is deprecated, use `plot_type=` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            plot_type = kwargs.pop("plot")

        distance_kwargs = dict(kwargs)

        if measure_time:
            import time

            start_time = time.time()

        # compute the OT distances
        D = compute_OT(
            self.supports,
            self.covariances,
            self.weights,
            self.identical_supports,
            precomputed_distances=precomputed_distances,
            ground_metric=ground_metric,
            **distance_kwargs,
        )

        if measure_time:
            end_time = time.time()
            total_time = end_time - start_time

        # plot the clustermap and embedding
        if isinstance(plot_type, str) or plot_type:
            from ggml_ot.plot import clustermap_embedding

            clustermap_embedding(
                D.cpu() if isinstance(D, torch.Tensor) else D,
                self.distribution_labels_str,
                symbols=self.symbols if (symbols is None and hasattr(self, "symbols")) else symbols,
                legend=legend,
                plot=plot_type if isinstance(plot_type, str) else "clustermap_embedding",
                title=f"{ground_metric} ground metric" if isinstance(ground_metric, str) else "GGML",
                show=show,
                save=save,
            )
        if measure_time:
            return D, total_time
        else:
            return D


# Update TripletDataset methods with signatures and docstrings from actual implementations
from ..optimization.api import train as _ggml_train  # noqa: E402
from ..optimization.api import train_emd2 as _ggml_train_emd2  # noqa: E402
from ..optimization.api import train_sinkhorn as _ggml_train_sinkhorn  # noqa: E402
from ..benchmark.evaluation import train_test as _bench_train_test  # noqa: E402
from ..benchmark.evaluation import test as _bench_test  # noqa: E402
from ..benchmark.hyperparams import tune as _bench_tune  # noqa: E402
from ..gmm.fit import fit_gmm as _gmm_fit_gmm  # noqa: E402
from ..gmm.validation import validate_gmm as _gmm_validate_gmm  # noqa: E402

update_wrapper(TripletDataset.train, _ggml_train)
TripletDataset.train.__signature__ = inspect.signature(_ggml_train)

update_wrapper(TripletDataset.train_emd2, _ggml_train_emd2)
TripletDataset.train_emd2.__signature__ = inspect.signature(_ggml_train_emd2)

update_wrapper(TripletDataset.train_sinkhorn, _ggml_train_sinkhorn)
TripletDataset.train_sinkhorn.__signature__ = inspect.signature(_ggml_train_sinkhorn)

update_wrapper(TripletDataset.train_test, _bench_train_test)
TripletDataset.train_test.__signature__ = inspect.signature(_bench_train_test)

update_wrapper(TripletDataset.test, _bench_test)
TripletDataset.test.__signature__ = inspect.signature(_bench_test)

update_wrapper(TripletDataset.tune, _bench_tune)
TripletDataset.tune.__signature__ = inspect.signature(_bench_tune)

update_wrapper(TripletDataset.fit_gmm, _gmm_fit_gmm)
TripletDataset.fit_gmm.__signature__ = inspect.signature(_gmm_fit_gmm)

update_wrapper(TripletDataset.validate_gmm, _gmm_validate_gmm)
TripletDataset.validate_gmm.__signature__ = inspect.signature(_gmm_validate_gmm)
