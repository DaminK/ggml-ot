import numpy as np
import pandas as pd
import inspect

from ggml_ot import settings
from ggml_ot.data.generic import TripletDataset
from ggml_ot._utils._docs import wraps


def synth_distributions(
    distribution_size=100,
    class_means=[5, 10, 15],
    offsets=[1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5],
    shared_means_x=[0, 40],
    shared_means_y=[0, 50],
    varying_size=False,
    noise_scale=10,
    noise_dims=1,
    show: bool | None = None,
    save: str | bool | None = None,
    return_generating_mode=False,
):
    """Generates distributions, labels and weights from synthetic data.

    Parameters
    ----------
    distribution_size : int
        Number of points per generating mode in each distribution.
    class_means : list
        Mean values for each class-specific Gaussian.
    offsets : list
        Offset values creating multiple distributions per class.
    shared_means_x : list
        X-coordinates of shared noise modes.
    shared_means_y : list
        Y-coordinates of shared noise modes.
    varying_size : bool
        If True, randomize distribution sizes.
    noise_scale : float
        Scale factor for noise dimensions.
    noise_dims : int
        Number of noise dimensions.
    show : bool or None
        Whether to display the plot.  ``None`` (default) automatically
        shows in interactive environments (notebooks, IPython) and
        suppresses in scripts.  ``True``/``False`` override explicitly.
    save : str, bool, or None
        Whether to save the figure to disk.  ``None``/``False`` skip
        saving.  ``True`` saves under the default name into
        ``settings.figdir``.  A *str* is used as the filename.
    return_generating_mode : bool
        If True, return a 5th element with per-point generating mode indices.
        Mode 0 = class-specific Gaussian, Mode 1+ = shared modes.

    Returns
    -------
    distributions : list[np.ndarray]
        List of point arrays, each shape (n_points, 1 + noise_dims).
    distributions_labels : list[int]
        Class label for each distribution.
    distributions_nr : list[int]
        Globally unique distribution ID.
    weights : None
        Placeholder for distribution weights (always None).
    distributions_generating_mode : list[np.ndarray], optional
        Per-point generating mode indices (only if return_generating_mode=True).
    """
    # Gaussian along dim 1, uniform along dim 2 (only information is the mean of the gaussian)
    unique_label = np.arange(len(class_means), dtype=int)
    distributions, distributions_labels, distributions_nr = [], [], []
    distributions_generating_mode = []
    plotting_df = []

    # create one distribution for each mean
    for mean, label in zip(class_means, unique_label):
        i = 0
        for offset in offsets:
            rand_size = settings.numpy_generator.integers(20, distribution_size) if varying_size else distribution_size

            dim1 = settings.numpy_generator.normal(10 + mean, size=rand_size, scale=1.5)
            dim2 = settings.numpy_generator.uniform(7.5 + offset, 12.5 + offset, size=(rand_size, noise_dims))

            # Track generating mode: 0 = class-specific
            generating_mode = np.zeros(rand_size, dtype=int)

            # add "noise" from shared modes
            for mode_idx, (shared_mean_x, shared_mean_y) in enumerate(zip(shared_means_x, shared_means_y), start=1):
                dim1 = np.concatenate((dim1, settings.numpy_generator.normal(shared_mean_x, size=rand_size, scale=1.5)))
                dim2 = np.concatenate(
                    (
                        dim2,
                        settings.numpy_generator.normal(shared_mean_y, size=(rand_size, noise_dims), scale=1.5),
                    ),
                    axis=0,
                )
                # Track generating mode: 1, 2, ... = shared modes
                generating_mode = np.concatenate((generating_mode, np.full(rand_size, mode_idx, dtype=int)))

            # scale and stack
            dim1 = dim1 * 5 / 4
            dim2 = dim2 * noise_scale
            stacked = np.insert(dim2, 0, dim1, axis=1).astype(np.float32)
            distributions.append(stacked)
            distributions_labels.append(label)
            distributions_nr.append(i)
            distributions_generating_mode.append(generating_mode)

            # collect plotting info
            plotting_df.append(pd.DataFrame({"x": dim1, "y": dim2[:, 0], "class": label, "distribution": i}))
            i += 1

    weights = None

    # Plot if requested
    if show is not False or save:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from ggml_ot.plot._utils import savefig_or_show

        df_plot = pd.concat(plotting_df, axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(df_plot, x="x", y="y", hue="class", style="distribution", palette="Set2", ax=ax)
        sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))
        savefig_or_show(fig, default_name="synth_distributions", show=show, save=save)

    if return_generating_mode:
        return distributions, distributions_labels, distributions_nr, weights, distributions_generating_mode
    return distributions, distributions_labels, distributions_nr, weights


def synth_anndata(
    distribution_size=100,
    class_means=[5, 10, 15],
    offsets=[1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5, 28.5],
    shared_means_x=[0, 40],
    shared_means_y=[0, 50],
    varying_size=False,
    noise_scale=100,
    noise_dims=1,
    n_top_genes=None,
):
    """Generate synthetic data and wrap it into an AnnData object.

    Creates an AnnData object with structure matching real-world scRNA-seq data,
    suitable for testing AnnData integration without network access.

    Parameters
    ----------
    distribution_size : int
        Number of points per generating mode in each distribution.
    class_means : list
        Mean values for each class (patient_group labels).
    offsets : list
        Offset values creating multiple distributions (samples) per class.
    shared_means_x : list
        X-coordinates of shared noise modes.
    shared_means_y : list
        Y-coordinates of shared noise modes.
    varying_size : bool
        If True, randomize distribution sizes.
    noise_scale : float
        Scale factor for noise dimensions.
    noise_dims : int
        Number of noise dimensions.
    n_top_genes : int, optional
        If provided, run highly variable genes selection (requires noise_dims >= n_top_genes).

    Returns
    -------
    adata : anndata.AnnData
        AnnData object with:
        - .X : Expression matrix (stacked distributions)
        - .obs["sample"] : Unique distribution/patient ID
        - .obs["patient_group"] : Class label
        - .obs["cell_type"] : Generating mode (0=class-specific, 1+=shared)
        - .obsm["X_pca"] : PCA embedding (same as X for synthetic)
    """
    import anndata
    import scanpy as sc

    # Generate synthetic data with generating mode info
    distributions, labels, dist_nrs, _, generating_modes = synth_distributions(
        distribution_size=distribution_size,
        class_means=class_means,
        offsets=offsets,
        shared_means_x=shared_means_x,
        shared_means_y=shared_means_y,
        varying_size=varying_size,
        noise_scale=noise_scale,
        noise_dims=noise_dims,
        show=False,
        return_generating_mode=True,
    )

    # Stack all distributions
    X = np.vstack(distributions)

    # Build obs dataframe with globally unique sample IDs
    # dist_nrs are only unique within a class, so combine label + dist_nr for global uniqueness
    sample_ids = []
    patient_groups = []
    cell_types = []

    for dist_nr, label, modes in zip(dist_nrs, labels, generating_modes):
        n_points = len(modes)
        # Create globally unique sample ID by combining label and dist_nr
        sample_ids.extend([f"sample_{label}_{dist_nr}"] * n_points)
        patient_groups.extend([f"group_{label}"] * n_points)
        cell_types.extend([f"mode_{m}" for m in modes])

    obs = pd.DataFrame(
        {
            "sample": sample_ids,
            "patient_group": patient_groups,
            "cell_type": cell_types,
        }
    )
    # AnnData normalizes obs indices to strings; set them upfront to avoid warning noise in tests.
    obs.index = obs.index.astype(str)

    # Create AnnData
    adata = anndata.AnnData(X=X, obs=obs)

    # Add gene names (var)
    adata.var_names = [f"gene_{i}" for i in range(X.shape[1])]

    # Run PCA to populate obsm["X_pca"]
    # n_comps must be < min(n_samples, n_features)
    max_comps = min(X.shape[0], X.shape[1]) - 1
    n_comps = min(max_comps, 50)
    if n_comps > 0:
        sc.pp.pca(adata, n_comps=n_comps)
    else:
        # Fallback: use X directly as "PCA"
        adata.obsm["X_pca"] = X

    # Optionally select highly variable genes
    if n_top_genes is not None:
        # For synthetic data, we simulate HVG by adding variance info
        adata.var["highly_variable"] = True
        if n_top_genes < adata.n_vars:
            adata.var["highly_variable"] = False
            adata.var.iloc[:n_top_genes, adata.var.columns.get_loc("highly_variable")] = True

    return adata


# FUTURE: deprecate/remove triplet param t, or integrate into training loop (where triplets are now created)
@wraps(synth_distributions)
def from_synth(*args, t=4, **kwargs) -> TripletDataset:
    """Creates synthetic dataset to train GGML"""
    kwargs.pop("return_generating_mode", None)
    distributions, distributions_labels, distributions_nr, weights = synth_distributions(*args, **kwargs)
    dataset = TripletDataset(distributions, distributions_labels, t, weights)
    dataset.symbols = distributions_nr
    return dataset


base_sig = inspect.signature(synth_distributions)
extra_param = inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=4)
from_synth.__signature__ = base_sig.replace(parameters=list(base_sig.parameters.values()) + [extra_param])
