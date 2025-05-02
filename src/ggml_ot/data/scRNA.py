## Import the usual libraries
import sys

sys.path.insert(0, "..")
# for local import of parent dict

import numpy as np
import scanpy as sc

# synth Data
from .util import create_t_triplets
from torch.utils.data import Dataset

# Optimal Transport
from ggml_ot.distances import compute_OT

# Plotting
from ggml_ot.plot import plot_emb, plot_clustermap
from sklearn.decomposition import PCA


class scRNA_Dataset(Dataset):
    """A dataset for handling single-cell RNA-seq data, generating triplets and computing OT distances.

    :ivar distributions: list of 2D arrays representing the distributions
    :vartype distributions: list of numpy.ndarray
    :ivar distributions_labels: class labels for each distribution
    :vartype distributions_labels: array-like of int
    :ivar disease_labels: disease label for each patient distribution
    :vartype disease_labels: array-like
    :ivar patient_labels: patient IDs
    :vartype patient_labels: array-like
    :ivar datapoints: array of all datapoints across all distributions
    :vartype datapoints: array-like
    :ivar datapoints_labels: class labels for each datapoint
    :vartype datapoints_labels: array-like of int
    :ivar cell_type_node_labels: list of cell_type labels for each patient
    :vartype cell_type_node_labels: array-like
    :ivar triplets: list of triplet indices used for training
    :vartype triplets: array-like of tuples
    """

    def __init__(self, *args, **kwargs):
        # Generate syntehtic data
        (
            distributions,
            distributions_labels,
            points,
            point_labels,
            disease_labels,
            celltype_node_labels,
            patient_labels,
        ) = get_cells_by_patients(*args, **kwargs)

        # Population-level
        self.distributions = distributions
        self.distributions_labels = distributions_labels
        self.disease_labels = disease_labels
        self.patient_labels = patient_labels

        # Unit-level
        self.datapoints = points
        self.datapoints_labels = point_labels
        self.celltype_node_labels = celltype_node_labels

        # Triplets
        self.triplets = create_t_triplets(distributions, distributions_labels, **kwargs)

    def __len__(self):
        # Datapoints to train are always given as triplets
        return len(self.triplets)

    def __getitem__(self, idx):
        # Returns elements and labels of triplet at idx
        i, j, k = self.triplets[idx]
        return np.stack(
            (self.distributions[i], self.distributions[j], self.distributions[k])
        ), np.stack(
            (
                self.distributions_labels[i],
                self.distributions_labels[j],
                self.distributions_labels[k],
            )
        )

    def get_cells_by_patients(self):
        return (
            self.distributions,
            self.distributions_labels,
            self.datapoints,
            self.datapoints_labels,
            self.patient_labels,
        )

    def compute_OT_on_dists(
        self,
        precomputed_distances=None,
        ground_metric=None,
        w=None,
        legend="Side",
        plot=True,
        symbols=None,
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
        :return: pairwise OT distance matrix
        :rtype: numpy.ndarray
        """

        # compute the OT distances
        D = compute_OT(
            self.distributions,
            precomputed_distances=precomputed_distances,
            ground_metric=ground_metric,
            w=w,
        )

        # plot the embedding and clustermap if wanted
        if plot:
            plot_emb(
                D,
                method="umap",
                colors=self.disease_labels,
                symbols=symbols,
                legend=legend,
                title="UMAP",
                verbose=True,
                annotation=None,
                s=200,
            )
            plot_clustermap(D, self.disease_labels, dist_name="W_θ")
        return D


def get_cells_by_patients(
    adata_path,
    patient_col="sample",
    label_col="patient_group",
    subsample_patient_ratio=1,
    n_cells=1000,
    n_feats=None,
    filter_genes=True,
    **kwargs,
):
    """Load and preprocess cells from an anndata set.

    :param adata_path: path to the ".h5ad" file containing the single-cell RNA data
    :type adata_path: str, optional
    :param patient_col: column name that identifies patients, defaults to 'donor_id'
    :type patient_col: str, optional
    :param label_col: column name that identifies disease labels, defaults to 'reported_diseases'
    :type label_col: str, optional
    :param subsample_patient_ratio: fraction of patients to randomly subsample, defaults to 0.25
    :type subsample_patient_ratio: float, optional
    :param n_cells: number of cells to subsample per patient, defaults to 1000
    :type n_cells: int, optional
    :param n_feats: number of features to retain when using PCA, defaults to None
    :type n_feats: int or None, optional
    :param filter_genes: whether to filter out genes with low variance (if True), defaults to True
    :type filter_genes: bool, optional
    :return: generated data (distributions and their labels, points and their labels, disease labels, celltype labels, patient labels)
    :rtype: tuple
    """
    # load data
    adata = sc.read_h5ad(adata_path)
    string_class_labels = np.unique(adata.obs[label_col])

    # detect low variable genes
    if filter_genes:
        gene_var = np.var(adata.X.toarray(), axis=0)

        # filter
        thresh = np.mean(gene_var)
        adata = adata[:, gene_var > thresh]
        print(adata)

    distributions = []
    distributions_class = []
    patient_labels = []
    disease_labels = []
    celltype_node_label = []

    # use PCA if given
    if n_feats is not None:
        global pca
        pca = PCA(n_components=n_feats, svd_solver="auto")
        pca.fit(adata.X)

    # subsample patients
    unique_patients = np.unique(adata.obs[patient_col])
    unique_patients_subsampled = np.random.choice(
        unique_patients,
        size=int(len(unique_patients) * subsample_patient_ratio),
        replace=False,
    )

    # iterate through each patient, store disease labels etc., subsample cells per patient
    for patient in unique_patients_subsampled:
        patient_adata = adata[adata.obs[patient_col] == patient]
        disease_label = np.unique(patient_adata.obs[label_col].to_numpy())
        string_class_label = disease_label[0]

        if len(disease_label) > 1:
            print(
                "Warning, sample_ids refer to cells with multiple disease labels (likely caused by referencing by patients and having multiple samples from different zones)"
            )

        if patient_adata.n_obs >= n_cells:
            sc.pp.subsample(patient_adata, n_obs=n_cells)
            p_arr = np.asarray(patient_adata.X.toarray(), dtype="f")
            if n_feats is not None:
                p_arr = pca.transform(p_arr)

            distributions.append(p_arr)
            disease_labels.append(string_class_label)
            distributions_class.append(
                np.where(string_class_labels == string_class_label)[0][0]
            )
            patient_labels.append(patient)
            celltype_node_label.append(list(patient_adata.obs["cell_type"]))

    # collect individual points and their labels from the distributions
    points = np.concatenate(distributions)
    point_labels = sum(
        [[l] * len(D) for l, D in zip(disease_labels, distributions)], []
    )

    return (
        distributions,
        distributions_class,
        points,
        point_labels,
        disease_labels,
        celltype_node_label,
        patient_labels,
    )
