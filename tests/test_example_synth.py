import ggml_ot
import numpy as np
import seaborn as sns
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def test_example_synth():
    # basically the content from examples/synth_Test.ipynb

    # define parameters
    alpha = 10
    lambda_ = 0.1

    neighbor_t = 5
    rank_k = 5

    lr = 0.02
    norm = 1

    # Number of element sampled
    n = 100

    # Length of list defines number of classes
    means = [5, 10, 15]
    # label = [0, 1, 2]

    # Length of list defines number of distributions in each class
    offsets = np.arange(0, 30, 3) + 1.5

    # Length of list defines number of modes that can not be differentiated between classes
    shared_means_x = [0, 40]
    shared_means_y = [0, 50]

    noise_scale = 1
    noise_dims = 199

    # create a synthetic dataset with defined paramters
    data = ggml_ot.synthetic_Dataset(
        distribution_size=n,
        class_means=means,
        offsets=offsets,
        shared_means_x=shared_means_x,
        shared_means_y=shared_means_y,
        plot=False,
        varying_size=False,
        t=neighbor_t,
        noise_scale=noise_scale,
        noise_dims=noise_dims,
    )

    # create training dataset that contains the dataset and meta data like the batching information
    training_dataset = DataLoader(data, batch_size=128, shuffle=True)

    # train the model
    # --> learns the global ground metric on the given dataset
    w_theta = ggml_ot.ggml(
        training_dataset,
        a=alpha,
        lam=lambda_,
        k=rank_k,
        lr=lr,
        norm=norm,
        max_iterations=5,
        plot_i_iterations=5,
        dataset=data,
        n_threads=64,
    )

    # plot distributions using PCA
    a = data.distributions
    b = data.distributions_labels
    ggml_ot.plot_distribution(a, b, title="Distributions (with PCA)", legend=True)

    train_distributions, _ , train_labels = next(iter(training_dataset))
    trpl_distributions = train_distributions[0]
    trpl_label = train_labels[0]
    ggml_ot.plot_distribution(
        trpl_distributions[:, :, :2], np.asarray(trpl_label, dtype=int)
    )

    np.set_printoptions(suppress=True, precision=16)

    fig, axs = plt.subplots(ncols=3, figsize=(12, 6))

    # plot no.1: Euclidean distance (identity matrix as covariance)
    ax = ggml_ot.plot_ellipses(np.identity(2), ax=axs[0])
    ax.set_title(r"Euclidean $d_2$ (Baseline)")

    # plot no.2: Mahalanobis distance (based on w_theta)
    m = np.transpose(w_theta) @ w_theta  # Mahalanobis matrix
    ax = ggml_ot.plot_ellipses(m[:2, :2], ax=axs[1])
    ax.set_title(r"Mahalanobis $d_\theta$ (GGML)")

    # plot no.3: Euclidean and Mahalanobis distances
    ax = ggml_ot.plot_ellipses([np.identity(2), m[:2, :2]], ax=axs[2])
    ax.set_title(r"Euclidean and Mahalanobis")

    plt.tight_layout()
    plt.show()

    ggml_ot.plot_heatmap(w_theta)

    # get the optimal transport distances using w_theta
    D_ggml = data.compute_OT_on_dists(w=w_theta, plot=False)

    ggml_ot.plot_heatmap(D_ggml)

    symbols = [i % 10 for i in range(len(data.distributions))]
    colors = data.distributions_labels

    _ = ggml_ot.plot_emb(
        D_ggml,
        method="mds",
        symbols=symbols,
        colors=colors,
        verbose=True,
        cmap=sns.color_palette("tab10", 3),
        s=150,
        legend="Side",
    )

    _ = ggml_ot.plot_clustermap(D_ggml, data.distributions_labels)

    # first placebo test
    assert w_theta is not None
