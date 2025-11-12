import numpy as np
from torch.utils.data import Dataset
import copy

from ggml_ot.distances import compute_OT
from ggml_ot.plot import clustermap_embedding

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

    triplets: list[tuple[int, int, int]] = None
    "Generated triplet index tuples used for training."

    dim: int = None
    "Dimensions of space underlying the distributions."

    identical_supports = None
    "Flag as passed to the constructor."

    _n_triplets = None
    _w_theta = None

    covariances = None  # experimental

    def __init__(
        self,
        supports,
        distribution_labels,
        n_triplets=3,
        weights=None,
        covariances=None,
        identical_supports=False,
    ):
        self.identical_supports = identical_supports  # TODO or infer from shape of supports
        self.dim = supports[0].shape[-1]

        self.supports = supports
        self.covariances = covariances
        self.weights = weights

        self.distribution_labels = distribution_labels  # TODO handle when str are passed
        self.triplets = create_triplets(distribution_labels, n_triplets)
        self._n_triplets = n_triplets

        self._w_theta = None

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
    def w_theta(self):
        """Learned ground metric as linear transformation (raises a warning if dataset is not trained yet)."""
        if self._w_theta is None:
            warnings.warn("This dataset has not been trained yet, please call train() on this object first.")
        return self._w_theta

    def __len__(self):
        """Returns the number of triplets.
        :return: number of triplets
        :rtype: int
        """
        return len(self.triplets)

    def __getitem__(self, idx):
        """Returns a triplet at position idx.
        :return: triplet
        :rtype: ([weights],[labels]) | ([supports],[weights],[labels]) #TODO
        """
        i, j, k = self.triplets[idx]

        if self.identical_supports:
            # Return weights and labels for distributions with identical_supports
            return (
                np.stack((self.weights[i], self.weights[j], self.weights[k])),
                np.stack(
                    (
                        self.distribution_labels[i],
                        self.distribution_labels[j],
                        self.distribution_labels[k],
                    )
                ),
            )
        else:
            if self.covariances is None:
                # Return supports, weights and labels of Empirical Distribution
                return (
                    np.stack((self.supports[i], self.supports[j], self.supports[k])),
                    np.stack((self.weights[i], self.weights[j], self.weights[k])) if self.weights is not None else [],
                    np.stack(
                        (
                            self.distribution_labels[i],
                            self.distribution_labels[j],
                            self.distribution_labels[k],
                        )
                    ),
                )
            else:
                # Return means (supports), covariances, weights and labels of GMMs
                # TODO maybe remove as we rarely will have seperate GMMs per patient
                return (
                    np.stack((self.supports[i], self.supports[j], self.supports[k])),
                    np.stack((self.covariances[i], self.covariances[j], self.covariances[k])),
                    np.stack((self.weights[i], self.weights[j], self.weights[k])) if self.weights is not None else [],
                    np.stack(
                        (
                            self.distribution_labels[i],
                            self.distribution_labels[j],
                            self.distribution_labels[k],
                        )
                    ),
                )

    def train():
        """Calls train function on this TripletDataset instance"""
        pass

    def test():
        """Calls test function on this TripletDataset instance"""
        pass

    def train_test():
        """Calls test_train function on this TripletDataset instance"""
        pass

    def test_train_splits():
        """Generates stratified train-test(-validation) splits on this TripletDataset instance and returns the indices"""
        pass

    def subset(self, indices):
        """Returns dataset subset for indices"""
        split_Dataset = copy.deepcopy(self)

        split_Dataset.distribution_labels = [self.distribution_labels[i] for i in indices]
        split_Dataset.triplets = create_triplets(split_Dataset.distribution_labels, self._n_triplets)

        if not self.identical_supports:
            split_Dataset.supports = [self.supports[i] for i in indices]
            if self.covariances is not None:
                split_Dataset.covariances = [self.covariances[i] for i in indices]

        if self.weights is not None:
            split_Dataset.weights = [self.weights[i] for i in indices]

        return split_Dataset

    def compute_OT(
        self,
        precomputed_distances=None,
        ground_metric=None,
        legend="Side",
        plot="clustermap_embedding",
        symbols=None,
        n_threads=1,
        **kwargs,
    ):
        """Compute the Optimal Transport distances between all distributions.

        :param precomputed_distances: optional matrix of precomputed distances for computing the OT, defaults to None
        :type precomputed_distances: array-like, optional
        :param ground_metric: ground metric for OT computation, defaults to None
        :type ground_metric: "euclidean", "cosine", "cityblock", optional
        :param w: weight matrix for the mahalanobis distance, defaults to None
        :type w: array-like, optional
        :param legend: defines where to place the legend, defaults to "Top"
        :type legend: "Top", "Side", optional
        :param plot: whether to plot the embedding and clustermap, defaults to True
        :type plot: bool, optional
        :param n_threads: either "max" to use all available threads during calculation or the specifc number of threads, defaults to 1
        :type n_threads: string, int
        :return: pairwise OT distance matrix
        :rtype: numpy.ndarray
        """

        # compute the OT distances
        D = compute_OT(
            self.supports,
            self.covariances,
            self.weights,
            self.identical_supports,
            precomputed_distances=precomputed_distances,
            ground_metric=ground_metric,
            n_threads=n_threads,
            **kwargs,
        )

        # plot the clustermap and embedding
        if isinstance(plot, str) or plot:
            clustermap_embedding(
                D,
                self.distribution_labels_str,
                symbols=self.symbols if (symbols is None and hasattr(self, "symbols")) else symbols,
                legend=legend,
                plot=plot if isinstance(plot, str) else "clustermap_embedding",
                title=f"{ground_metric} ground metric" if isinstance(ground_metric, str) else "GGML",
                s=200,
            )
        return D


def create_triplets(labels, t=5, **kwargs):
    """Creates t triplets for each point for metric learning where i and j are from the same class and
    k is from a different class.

    :param labels: distribution labels to create triplets from
    :type labels: array-like
    :param t: number of neighbors to sample from both the same and different classes, defaults to 5
    :type t: int, optional
    :return: list of created triplets
    :rtype: list of tuples
    """
    labels = np.asarray(labels)
    triplets = []
    replace = any(np.unique(labels, return_counts=True)[1] < t)

    def get_neighbors(class_, skip=None):
        # get t elements from distributions where labels = class
        # TODO optional skip self
        return np.random.choice(np.where(labels == class_)[0], size=t, replace=replace)

    for j, c_j in enumerate(labels):
        for i in get_neighbors(c_j):
            for c_k in np.unique(labels):
                if c_k != c_j:
                    for k in get_neighbors(c_k):
                        triplets.append((i, j, k))
    return triplets
