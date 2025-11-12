import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import inspect

from ggml_ot.data.generic import TripletDataset
from ggml_ot.data.util import wraps


def synth_distributions(
    distribution_size=100,
    class_means=[5, 10, 15],
    offsets=[1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5],
    shared_means_x=[0, 40],
    shared_means_y=[0, 50],
    varying_size=False,
    noise_scale=10,
    noise_dims=1,
    plot=True,
):
    """Generates distributions, labels and weights from sythetic data."""
    # Gaussian along dim 1, uniform along dim 2 (only information is the mean of the gaussian)
    unique_label = np.arange(len(class_means), dtype=int)
    distributions, distributions_labels, distributions_nr = [], [], []
    plotting_df = []

    # create one distribution for each mean
    for mean, label in zip(class_means, unique_label):
        i = 0
        for offset in offsets:
            rand_size = np.random.randint(20, distribution_size) if varying_size else distribution_size

            dim1 = np.random.normal(10 + mean, size=rand_size, scale=1.5)
            dim2 = np.random.uniform(7.5 + offset, 12.5 + offset, size=(rand_size, noise_dims)).astype(np.float32)

            # add "noise"
            for shared_mean_x, shared_mean_y in zip(shared_means_x, shared_means_y):
                dim1 = np.concatenate(
                    (
                        dim1,
                        np.random.normal(shared_mean_x, size=rand_size, scale=1.5).astype(np.float32),
                    )
                )
                dim2 = np.concatenate(
                    (
                        dim2,
                        np.random.normal(shared_mean_y, size=(rand_size, noise_dims), scale=1.5).astype(np.float32),
                    ),
                    axis=0,
                )

            # scale and stack
            dim1 = dim1 * 5 / 4
            dim2 = dim2 * noise_scale
            stacked = np.insert(dim2, 0, dim1, axis=1)
            distributions.append(stacked)
            distributions_labels.append(label)
            distributions_nr.append(i)

            # collect plotting info
            plotting_df.append(pd.DataFrame({"x": dim1, "y": dim2[:, 0], "class": label, "distribution": i}))
            i += 1
    weights = None

    # Plot if requested
    if plot:
        df_plot = pd.concat(plotting_df, axis=0)
        plt.figure(figsize=(6, 5))
        ax = sns.scatterplot(df_plot, x="x", y="y", hue="class", style="distribution", palette="Set2")
        sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))
        plt.show()

    return distributions, distributions_labels, distributions_nr, weights


@wraps(synth_distributions)
def from_synth(*args, t=5, **kwargs) -> TripletDataset:
    """Creates synthetic dataset to train GGML"""
    distributions, distributions_labels, distributions_nr, weights = synth_distributions(*args, **kwargs)
    dataset = TripletDataset(distributions, distributions_labels, t, weights)
    dataset.symbols = distributions_nr
    return dataset


base_sig = inspect.signature(synth_distributions)
extra_param = inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=5)
from_synth.__signature__ = base_sig.replace(parameters=list(base_sig.parameters.values()) + [extra_param])
