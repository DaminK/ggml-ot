from .util import create_t_triplets
from torch.utils.data import Dataset
import pandas as pd

from ggml_ot.distances import compute_OT
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from ggml_ot.plot import plot_emb, plot_clustermap


class synthetic_Dataset(Dataset):
    """A dataset for generating and handling synthetic point cloud data. It organizes the data into triplets for training.

    :ivar distributions: list of 2D arrays representing the synthetic distributions
    :vartype distributions: list of numpy.ndarray
    :ivar distributions_labels: class labels for each distribution
    :vartype distributions_labels: array-like of int
    :ivar distribution_modes: indicates whether a datapoint comes from a shared mean (1) or not (0)
    :vartype distribution_modes: array-like of int
    :ivar datapoints: array of all datapoints across all distributions
    :vartype datapoints: array-like
    :ivar datapoints_labels: class labels for each datapoint
    :vartype datapoints_labels: array-like of int
    :ivar triplets: list of triplet indices used for training
    :vartype triplets: array-like of tuples
    """
    # The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, *args, **kwargs):
        # Generate syntehtic data
        (
            distributions,
            distributions_labels,
            points,
            point_labels,
            distribution_modes,
        ) = get_pointcloud(*args, **kwargs)

        # Population-level
        self.distributions = distributions
        self.distributions_labels = distributions_labels
        self.distribution_modes = distribution_modes

        # Unit-level
        self.datapoints = points
        self.datapoints_labels = point_labels

        # Triplets
        self.triplets = create_t_triplets(distributions, distributions_labels, **kwargs)

    def __len__(self):
        # Datapoints to train are always given as triplets
        return len(self.triplets)

    def __getitem__(self, idx):
        # Returns elements and labels of triplet at idx
        i, j, k = self.triplets[idx]
        return np.stack(
            (self.distributions[i], self.distributions[j], self.distributions[k]),
            dtype='f',
        ), np.stack(
            (
                self.distributions_labels[i],
                self.distributions_labels[j],
                self.distributions_labels[k],
            ),
            dtype='f',
        )

    def get_raw_distributions(self):
        return self.distributions, self.distributions_labels
    
    def compute_OT_on_dists(self, precomputed_distances = None, 
                            ground_metric=None, w=None, legend='Side', plot=True):
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
        :return: pairwise OT distance matrix
        :rtype: numpy.ndarray
        """
        
        # compute the OT distances
        D = compute_OT(self.distributions, precomputed_distances = precomputed_distances,
                       ground_metric = ground_metric, w = w)

        # optionally plot the embedding and clustermap
        if plot:
            hardcoded_symbols = [i % 10 for i in range(len(self.distributions))]
            plot_emb(
                D,
                method='umap',
                colors=self.distributions_labels,
                symbols=hardcoded_symbols,
                legend=legend,
                title='UMAP',
                verbose=True,
                cmap=sns.cubehelix_palette(as_cmap=True),
                annotation=None,
                s=200,
            )
            
            plot_clustermap(
                D,
                self.distributions_labels,
                cmap=sns.cubehelix_palette(
                    as_cmap=False, n_colors=len(np.unique(self.distributions_labels))
                ),
                dist_name='W_θ',
            )

        return D


def get_pointcloud(
    distribution_size=100,
    #class_means=[0, 5, 10],
    class_means = [5,10,15],
    #offsets=[0, 5, 10, 15],
    offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
    # shared_means_x=[],
    # shared_means_y=[],
    shared_means_x = [0, 40],
    shared_means_y = [0, 50],
    plot=True,
    #varying_size=True,
    varying_size=False,
    noise_scale=1000,
    noise_dims=1,
    #return_dict=True,
    return_dict=False,
    *args,
    **kwargs,
):
    """Generates a synthetic pointcloud with labeled 2D distributions.

    :param distribution_size: number of points per distribution, defaults to 100
    :type distribution_size: int, optional
    :param class_means: means of the different classes, defaults to [5,10,15]
    :type class_means: array-like, optional
    :param offsets: shifts applied to the datapoints in y-dimension, defaults to [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    :type offsets: array-like, optional
    :param shared_means_x: shared means in x-dimension, defaults to [0, 40]
    :type shared_means_x: array-like, optional
    :param shared_means_y: shared means in y-dimension, defaults to [0, 50]
    :type shared_means_y: array-like, optional
    :param plot: whether to plot the generated points (1) or not (0), defaults to True
    :type plot: bool, optional
    :param varying_size: whether to vary the size of each distribution (1) or not (0), defaults to False
    :type varying_size: bool, optional
    :param noise_scale: scaling factor for the noise, defaults to 1000
    :type noise_scale: int, optional
    :param noise_dims: defines how many dimensions of noise are added, defaults to 1
    :type noise_dims: int, optional
    :param return_dict: whether to return a dictionary (1) or a tuple (0), defaults to False
    :type return_dict: bool, optional
    :return: generated data (distributions and their labels, points and their labels, distribution modes)
    :rtype: tuple or dict
    """
    # Gaussian along dim 1, uniform along dim 2 (only information is the mean of the gaussian)
    unique_label = np.arange(len(class_means), dtype=int)
    # generated data points for each distribution
    distributions = []
    # class labels for each distribution
    distributions_labels = []
    # data points for plotting
    plotting_df = []
    label_distribution_modes = []

    # create one distribution for each mean
    for mean, label in zip(class_means, unique_label):
        i = 0
        # generate a distribution of points for each offset
        for offset in offsets:
            rand_size = (
                np.random.randint(20, distribution_size)
                if varying_size
                else distribution_size
            )

            dim1 = np.random.normal(10 + mean, size=rand_size, scale=1.5)
            dim2 = np.random.uniform(
                7.5 + offset, 12.5 + offset, size=(rand_size, noise_dims)
            )
            label_distribution_modes = label_distribution_modes + [1] * rand_size
            
            # add "noise"
            for shared_mean_x, shared_mean_y in zip(shared_means_x, shared_means_y):
                dim1 = np.concatenate(
                    (dim1, np.random.normal(shared_mean_x, size=rand_size, scale=1.5))
                )
                dim2 = np.concatenate(
                    (
                        dim2,
                        np.random.normal(
                            shared_mean_y, size=(rand_size, noise_dims), scale=1.5
                        ),
                    ),
                    axis=0,
                )  # #np.random.normal(2.5+offset,size=n)
                label_distribution_modes = label_distribution_modes + [0] * rand_size

            # scale values and combine to 2D array
            dim1 = dim1 * 5 / 4
            dim2 = dim2 * noise_scale
            stacked = np.insert(dim2, 0, dim1, axis=1)

            # stacked = np.append(dim2,[dim1],axis=0)
            # stacked = np.stack((dim1,dim2),axis=-1)
            plotting_df.append(
                pd.DataFrame(
                    {'x': dim1, 'y': dim2[:, 0], 'class': label, 'distribution': i}
                )
            )

            distributions.append(stacked)
            distributions_labels.append(label)

            i += 1

    # plot if desired
    if plot:
        df = pd.concat(plotting_df, axis=0)

        plt.figure(figsize=(6, 5))
        ax = sns.scatterplot(df, x='x', y='y', hue='class', style='distribution')
        sns.move_legend(ax, 'center right', bbox_to_anchor=(1.3, 0.5))

        # xticks = ax.xaxis.get_major_ticks()
        # xticks[0].label1.set_visible(False)
        # yticks = ax.yaxis.get_major_ticks()
        # xticks[-1].label1.set_visible(False)

        plt.show()

    points = np.concatenate(distributions)
    point_labels = sum(
        [[l] * len(D) for l, D in zip(distributions_labels, distributions)], []
    )  # flattens list of lists

    # return as dictionary 
    if return_dict:
        data_dict = {}
        (
            data_dict['distributions'],
            data_dict['distributions_labels'],
            data_dict['points'],
            data_dict['point_labels'],
            data_dict['patient'],
        ) = (
            distributions,
            distributions_labels,
            points,
            point_labels,
            label_distribution_modes,
        )
        return data_dict
    # return as tuple
    else:
        return (
            distributions,
            distributions_labels,
            points,
            point_labels,
            label_distribution_modes,
        )
