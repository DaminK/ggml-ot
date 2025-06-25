import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gprofiler  # https://pypi.org/project/gprofiler-official/


def importance(
    adata,
    n_top_genes=10,
    reconstruct_covariances=False,
    only_diagonal=False,
    plot=True,
    save_path=None,
):
    if "W_ggml" not in adata.uns.keys():
        raise Exception("GGML not trained on this Anndata object yet")

    if "use_rep_GGML" in adata.uns.keys() and adata.uns["use_rep_GGML"] is not None:
        print(
            "not implemented yet for use_rep as we need to project the components back into the gene space and noone saves the components itself...."
        )
        return []

    w_theta = adata.uns["W_ggml"]
    gene_name = (
        [gene for gene in adata.var["feature_name"]]
        if "feature_name" in adata.var.keys()
        else [gene for gene in adata.var.index]
    )

    rank_k = w_theta.shape[0]
    components = range(rank_k) if not reconstruct_covariances else [0]

    most_important_genes_list = []

    for component in components:
        if reconstruct_covariances:
            w_theta_gene_space = np.dot(np.transpose(w_theta), w_theta)
        else:
            w_theta_gene_space = w_theta[component, :]

        if not only_diagonal and reconstruct_covariances:
            gene_pair_names = np.zeros((len(gene_name), len(gene_name)), dtype="object")
            for i, gene_i in enumerate(gene_name):
                for j, gene_j in enumerate(gene_name):
                    if i == j:
                        gene_pair_names[i, j] = f"{gene_i}"
                    else:
                        gene_pair_names[i, j] = f"{gene_i} x {gene_j}"

            flat_gene_importance = np.abs(
                w_theta_gene_space[np.triu_indices(len(w_theta_gene_space))].flatten()
            )
            flat_gene_name = gene_pair_names[
                np.triu_indices(len(gene_pair_names))
            ].flatten()
        else:
            if reconstruct_covariances:
                flat_gene_importance = np.abs(np.sum(w_theta_gene_space, axis=0))
            else:
                flat_gene_importance = np.abs(w_theta_gene_space)
            flat_gene_name = np.asarray(gene_name)

        most_important_genes_ind = np.argsort(flat_gene_importance)[-n_top_genes:]
        most_important_genes_names = flat_gene_name[most_important_genes_ind]
        average_importance = np.mean(flat_gene_importance)
        most_important_genes_values = (
            flat_gene_importance[most_important_genes_ind] / average_importance
        )
        sort_by_value = np.flip(np.argsort(most_important_genes_values))

        most_important_genes_list.append(most_important_genes_names[sort_by_value])

        if plot:
            print(f"Component {component}")
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

            # TODO Decide which plots look better
            # plt.figure(figsize=(2,int(n_top_genes *0.5)))
            # sns.stripplot(gene_df, x="relative importance", y="genes", hue="up/down",legend=False,palette={-1:down_color,0:no_color,1:up_color})

            plt.figure(figsize=(3, int(n_top_genes * 0.7)))
            sns.barplot(
                gene_df,
                x="relative importance",
                y="genes",
                hue="up/down",
                legend=False,
                palette={-1: down_color, 0: no_color, 1: up_color},
            )

            addlabels(
                most_important_genes_values[sort_by_value],
                most_important_genes_names[sort_by_value],
                color="black",
            )
            plt.yticks([])
            plt.vlines(
                x=1,
                ymin=-1 / 2,
                ymax=len(sort_by_value),
                color="black",
                label="axvline - full height",
                linestyles="dashed",
            )
            plt.text(
                x=1 + 0.3,
                y=len(sort_by_value) - 1 / 2,
                s="average",
                verticalalignment="top",
            )  # rotation=90)

            if save_path is not None:
                plt.savefig(save_path + f"feature_importance_comp{component}.pdf")
            plt.show()

    return most_important_genes_list


def enrichment(
    most_important_genes_list,
    ordered=True,
    save_path=None,
    thresh=0.01,
    organism="hsapiens",
):
    for i, most_important_genes in enumerate(most_important_genes_list):
        gp = gprofiler.GProfiler(return_dataframe=True)
        enrich = gp.profile(
            query=list(most_important_genes),
            ordered=ordered,
            user_threshold=thresh,
            organism=organism,
        )
        enrich["NES"] = -np.log10(enrich["p_value"])

        plt.figure(figsize=(10, 20))
        plt.subplots_adjust(left=0.5)
        sns.barplot(x="p_value", y="name", data=enrich, color="green")
        plt.title(
            f"Gene Enrichment for X_ggml{i + 1} (Top {len(most_important_genes)} Genes)"
        )

        if save_path is not None:
            plt.savefig(save_path + f"biological_process_com{i}.pdf")
        plt.show()

    gp = gprofiler.GProfiler(return_dataframe=True)
    enrich = gp.profile(
        query={
            f"component{i}": list(most_important_genes)
            for i, most_important_genes in enumerate(most_important_genes_list)
        },
        ordered=ordered,
        user_threshold=thresh,
        organism=organism,
    )
    enrich["NES"] = -np.log10(enrich["p_value"])

    plt.figure(figsize=(10, 30))
    plt.subplots_adjust(left=0.5)
    sns.barplot(x="p_value", y="name", data=enrich, color="green")
    plt.title("Gene Enrichment queried all components")

    if save_path is not None:
        plt.savefig(save_path + "biological_process_multiquery.pdf")
    plt.show()

    gp = gprofiler.GProfiler(return_dataframe=True)
    enrich = gp.profile(
        organism="hsapiens",
        query=np.concatenate(most_important_genes_list).tolist(),
        ordered=ordered,
        user_threshold=thresh,
    )
    enrich["NES"] = -np.log10(enrich["p_value"])

    plt.figure(figsize=(10, 30))
    plt.subplots_adjust(left=0.5)
    sns.barplot(x="p_value", y="name", data=enrich, color="green")
    plt.title("Gene Enrichment combined all components")

    if save_path is not None:
        plt.savefig(save_path + "biological_process_combined.pdf")
    plt.show()
