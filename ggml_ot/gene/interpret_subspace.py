import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gprofiler

from ggml_ot.plot._utils import savefig_or_show


def importance(
    adata,
    n_top_genes=10,
    reconstruct_covariances=False,
    only_diagonal=False,
    show: bool | None = None,
    save: str | bool | None = None,
):
    """Identifes and visualizes the most important genes contributing to each component of a learned low-dimensional gene embedding from an Anndata object.

    :param adata: Anndata object containing the single-cell RNA data
    :type adata: Anndata
    :param n_top_genes: number of top-ranking genes to extract per component, defaults to 10
    :type n_top_genes: int, optional
    :param reconstruct_covariances: whether to reconstruct the full covariance matrices from the embedding components, defaults to False
    :type reconstruct_covariances: bool, optional
    :param only_diagonal: consider only the diagonal of the reconstructed covariance matrix if set to True, defaults to False
    :type only_diagonal: bool, optional
    :param show: Whether to display the plot.  ``None`` (default) automatically shows in
        interactive environments (notebooks, IPython) and suppresses in scripts.
        ``True``/``False`` override explicitly.
    :type show: bool or None, optional
    :param save: Whether to save the figure to disk.  ``None``/``False`` skip saving.
        ``True`` saves under the default name into ``settings.figdir``.
        A *str* is used as the filename.
    :type save: str, bool, or None, optional
    :return: a list of lists containing the names of the most important genes for each component, sorted by relative importance
    :rtype: array-like
    """
    if "W_GGMLs" not in adata.uns.keys():
        raise Exception("GGML not trained on this Anndata object yet")

    if "use_rep_GGML" in adata.uns.keys() and adata.uns["use_rep_GGML"] is not None:
        print(
            "not implemented yet for use_rep as we need to project the components back into the gene space and noone saves the components itself...."
        )
        return []

    map_A = adata.uns["W_GGMLs"]
    gene_name = (
        [gene for gene in adata.var["feature_name"]]
        if "feature_name" in adata.var.keys()
        else [gene for gene in adata.var.index]
    )

    rank_k = map_A.shape[0]
    components = range(rank_k) if not reconstruct_covariances else [0]

    most_important_genes_list = []

    fig, axs = plt.subplots(1, len(components))  # figsize=(3, int(n_top_genes * 0.7))

    for component in components:
        if reconstruct_covariances:
            w_theta_gene_space = np.dot(np.transpose(map_A), map_A)
        else:
            w_theta_gene_space = map_A[component, :]

        if not only_diagonal and reconstruct_covariances:
            gene_pair_names = np.zeros((len(gene_name), len(gene_name)), dtype="object")
            for i, gene_i in enumerate(gene_name):
                for j, gene_j in enumerate(gene_name):
                    if i == j:
                        gene_pair_names[i, j] = f"{gene_i}"
                    else:
                        gene_pair_names[i, j] = f"{gene_i} x {gene_j}"

            flat_gene_importance = np.abs(w_theta_gene_space[np.triu_indices(len(w_theta_gene_space))].flatten())
            flat_gene_name = gene_pair_names[np.triu_indices(len(gene_pair_names))].flatten()
        else:
            if reconstruct_covariances:
                flat_gene_importance = np.abs(np.sum(w_theta_gene_space, axis=0))
            else:
                flat_gene_importance = np.abs(w_theta_gene_space)
            flat_gene_name = np.asarray(gene_name)

        most_important_genes_ind = np.argsort(flat_gene_importance)[-n_top_genes:]
        most_important_genes_names = flat_gene_name[most_important_genes_ind]
        average_importance = np.mean(flat_gene_importance)
        most_important_genes_values = flat_gene_importance[most_important_genes_ind] / average_importance
        sort_by_value = np.flip(np.argsort(most_important_genes_values))

        most_important_genes_list.append(most_important_genes_names[sort_by_value])

        gene_df = pd.DataFrame(
            {
                "genes": most_important_genes_names[sort_by_value],
                "relative importance": most_important_genes_values[sort_by_value],
                "up/down": np.sign(most_important_genes_values[sort_by_value]),
            }
        )
        up_color = "lightsteelblue"
        down_color = "darkred"
        no_color = "grey"

        def addlabels(x, y, color=None):
            for i in range(len(y)):
                plt.text(
                    x[i] // 2,
                    i,
                    y[i],
                    ha="center",
                    color=color,
                    verticalalignment="center",
                )

        sns.barplot(
            gene_df,
            x="relative importance",
            y="genes",
            hue="up/down",
            legend=False,
            palette={-1: down_color, 0: no_color, 1: up_color},
            ax=axs[component],
        ).set_title(f"W_GGML {component + 1}")

        addlabels(
            most_important_genes_values[sort_by_value],
            most_important_genes_names[sort_by_value],
            color="black",
        )
        axs[component].set_yticks([])
        axs[component].vlines(
            x=1,
            ymin=-1 / 2,
            ymax=len(sort_by_value),
            color="black",
            label="axvline - full height",
            linestyles="dashed",
        )
        axs[component].text(
            x=1 + 0.3,
            y=len(sort_by_value) - 1 / 2,
            s="average",
            verticalalignment="top",
        )

    fig.suptitle(f"Gene Importance (Top {n_top_genes})")
    savefig_or_show(fig, default_name="gene_importance", show=show, save=save)

    return most_important_genes_list


def enrichment1(
    top_genes,
    ordered=True,
    show: bool | None = None,
    save: str | bool | None = None,
    thresh=0.05,
    organism="hsapiens",
):
    """Performs enrichment analysis on top-ranked genes and visualizes the enriched biological terms.

    :param top_genes: a list of gene lists where each list contains the top genes for a given component
    :type top_genes: array-like
    :param ordered: whether the gene lists are ordered by importance, defaults to True
    :type ordered: bool, optional
    :param show: Whether to display the plot.  ``None`` (default) automatically shows in
        interactive environments (notebooks, IPython) and suppresses in scripts.
        ``True``/``False`` override explicitly.
    :type show: bool or None, optional
    :param save: Whether to save the figure to disk.  ``None``/``False`` skip saving.
        ``True`` saves under the default name into ``settings.figdir``.
        A *str* is used as the filename.
    :type save: str, bool, or None, optional
    :param thresh: threshold for significance in enrichment, defaults to 0.01
    :type thresh: float, optional
    :param organism: organism ID for g:Profiler, defaults to "hsapiens"
    :type organism: str, optional
    """
    for i, most_important_genes in enumerate(top_genes):
        gp = gprofiler.GProfiler(return_dataframe=True)
        enrich = gp.profile(
            query=list(most_important_genes),
            ordered=ordered,
            user_threshold=thresh,
            organism=organism,
        )
        enrich["NES"] = -np.log10(enrich["p_value"])

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0.5)
        sns.barplot(x="p_value", y="name", data=enrich, color="green", ax=ax)
        ax.set_title(f"Gene Enrichment for W_GGML{i + 1} (Top {len(most_important_genes)} Genes)")

        savefig_or_show(fig, default_name=f"biological_process_com{i}", show=show, save=save)

    gp = gprofiler.GProfiler(return_dataframe=True)
    enrich = gp.profile(
        organism=organism,
        query=np.concatenate(top_genes).tolist(),
        ordered=ordered,
        user_threshold=thresh,
    )
    enrich["NES"] = -np.log10(enrich["p_value"])

    fig, ax = plt.subplots(figsize=(10, 30))
    fig.subplots_adjust(left=0.5)
    sns.barplot(x="p_value", y="name", data=enrich, color="green", ax=ax)
    ax.set_title("Gene Enrichment combined all components")

    savefig_or_show(fig, default_name="biological_process_combined", show=show, save=save)
