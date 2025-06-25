## Import the usual libraries
import sys

sys.path.insert(0, "..")

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import scanpy as sc
from anndata import AnnData

# synth Data
from .util import create_t_triplets
from torch.utils.data import Dataset

# Optimal Transport
from ggml_ot.distances import compute_OT

# Plotting
from ggml_ot.plot import plot_emb, plot_clustermap


class scRNA_Dataset(Dataset):
    """An Anndata interface to pytorch datasets to be used with Supervised Optimal Transport. It formats
    patient-level single-cell RNA-seq data as triplets of relative relationships to train GGML.

    :param adata: Anndata object containing the single-cell RNA data or path to the ".h5ad" file containing the Anndata
    :type adata: Anndata or str
    :param patient_col: column name that identifies patients, defaults to 'donor_id'
    :type patient_col: str, optional
    :param label_col: column name that identifies disease labels, defaults to 'reported_diseases'
    :type label_col: str, optional
    :param groub_by: column name that identifies cell clusters that should be used instead of single cells, defaults to None
    :type groub_by: str, optional
    :param use_rep: key that identifies low-dimensional represention of the cells in .obsm (like "X_pca") that should be used instead of .X, defaults to None
    :type use_rep: str, optional
    :param subsample_patient_ratio: fraction of patients to randomly subsample, defaults to 1
    :type subsample_patient_ratio: float, optional
    :param n_cells: number of cells to subsample per patient, set to 0 to keep all cells (only supported for centroids and gaussians), defaults to 1000
    :type n_cells: int, optional
    :param filter_genes: whether to filter out genes with low variance (if True), defaults to True
    :type filter_genes: bool, optional
    :param train_size: fraction of train split. If None, no splitting is done, defaults to None
    :type train_size: float, optional
    :return: generated data (distributions and their labels, points and their labels, disease labels, celltype labels, patient labels)
    :rtype: tuple

    :ivar adata: Anndata object containing the single-cell RNA data
    :vartype adata: Anndata
    :ivar distributions: a 2D array representing cells and their gene expressions of shape (n_cells, n_features) for each patient (n_patients)
    :vartype distributions: list of numpy.ndarray
    :ivar distributions_labels: integer-encoded class labels of shape (n_patients) for each distribution
    :vartype distributions_labels: array-like of int
    :ivar disease_labels: string labels of diseases of shape (n_patients) for each distribution
    :vartype disease_labels: array-like
    :ivar patient_labels: ID of each patient (distribution) of shape (n_patients)
    :vartype patient_labels: array-like
    :ivar datapoints: a concatenation of all distributions of shape (n_cells * n_patients, n_features)
    :vartype datapoints: array-like
    :ivar datapoints_labels: string class labels for each datapoint of shape (n_cells * n_patients)
    :vartype datapoints_labels: array-like of int
    :ivar cell_ype_node_labels: list of cell_type labels for each patient of shape (n_patients, n_cells)
    :vartype celltype_node_labels: array-like
    :ivar triplets: list of triplet indices of shape (n_cells * n_patients, 3) used for training
    :vartype triplets: array-like of tuples

    :ivar test_distributions: a 2D array representing cells and their gene expressions of shape (n_cells, n_features) for each patient (n_patients); is only returned if train_size not None
    :vartype test_distributions: list of numpy.ndarray
    :ivar test_distributions_labels: integer-encoded class labels of shape (n_patients) for each distribution; is only returned if train_size not None
    :vartype test_distributions_labels: array-like of int
    :ivar test_datapoints: a concatenation of all distributions of shape (n_cells * n_patients, n_features)
    :vartype test_datapoints: array-like
    """

    def __init__(
        self,
        adata,
        patient_col="sample",
        label_col="patient_group",
        group_by=None,
        use_rep=None,
        subsample_patient_ratio=1,
        n_cells=1000,
        train_size=None,
        filter_genes=True,
        **kwargs,
    ):
        result = self.get_cells_by_patients(
            adata=adata,
            patient_col=patient_col,
            label_col=label_col,
            group_by=group_by,
            use_rep=use_rep,
            subsample_patient_ratio=subsample_patient_ratio,
            n_cells=n_cells,
            filter_genes=filter_genes,
            train_size=train_size,
            **kwargs,
        )

        if train_size is None:
            (
                distributions,
                distributions_labels,
                points,
                point_labels,
                disease_labels,
                celltype_node_labels,
                patient_labels,
                cluster_centers,
                adata_object,
                point_patient_labels,
            ) = result
        else:
            (
                (
                    distributions,
                    distributions_labels,
                    points,
                    point_labels,
                    disease_labels,
                    celltype_node_labels,
                    patient_labels,
                    cluster_centers,
                    adata_object,
                    point_patient_labels,
                ),
                (
                    test_distributions,
                    test_distributions_labels,
                    test_points,
                    test_point_labels,
                    test_disease_labels,
                    test_celltype_node_labels,
                    test_patient_labels,
                    test_cluster_centers,
                    test_adata_object,
                    test_point_patient_labels,
                ),
            ) = result
            self.test_distributions = test_distributions
            self.test_distributions_labels = test_distributions_labels
            self.test_datapoints = test_points
            # self.test_datapoint_labels = test_point_labels
            # self.test_disease_labels = test_disease_labels
            # self.test_celltype_node_labels = test_celltype_node_labels
            # self.test_patient_labels = test_patient_labels
            # self.test_cluster_centers = test_cluster_centers
            # self.test_adata = test_adata_object
            # self.test_point_patient_labels = test_point_patient_labels
            # self.test_triplets = create_t_triplets(test_distributions, test_distributions_labels, **kwargs)

        self.adata = adata_object

        # Population-level
        self.distributions = distributions
        self.distributions_labels = distributions_labels
        self.disease_labels = disease_labels
        self.patient_labels = patient_labels
        self.point_patient_labels = point_patient_labels

        # Cluster-level
        self.cluster_centers = cluster_centers

        # Unit-level
        self.datapoints = points
        self.datapoints_labels = point_labels
        self.celltype_node_labels = celltype_node_labels

        # Triplets
        self.triplets = create_t_triplets(distributions, distributions_labels, **kwargs)

    def __len__(self):
        """Returns the number of triplets.

        :return: number of triplets
        :rtype: int
        """
        # Datapoints to train are always given as triplets
        return len(self.triplets)

    def __getitem__(self, idx):
        """Returns elements and labels of triplet at idx, and cluster_centers if clusters are used
        else this entry in the triplets is None

        :param idx: index of the triplet to retrieve
        :type idx: int
        :return: a tuple of (triplet_data, cluster_centers, triplet_labels) where triplet_data is a tuple of the three distributions of the triplet, cluster_centers are the used cluster centers if clustering is used and triplet_labels are the labels for each distribution in the triplet
        :rtype: tuple
        """
        i, j, k = self.triplets[idx]

        return (
            np.stack(
                (self.distributions[i], self.distributions[j], self.distributions[k])
            ),
            self.cluster_centers,
            np.stack(
                (
                    self.distributions_labels[i],
                    self.distributions_labels[j],
                    self.distributions_labels[k],
                )
            ),
        )

    def get_cells_by_patients(
        self,
        adata,
        patient_col="sample",
        label_col="patient_group",
        group_by=None,
        use_rep=None,
        subsample_patient_ratio=1,
        n_cells=1000,
        filter_genes=True,
        train_size=None,
        **kwargs,
    ):
        """Load and preprocess cells from an anndata set.

        :param adata: Anndata object containing the single-cell RNA data or path to the ".h5ad" file containing
        the Anndata
        :type adata: Anndata or str
        :param patient_col: column name that identifies patients, defaults to 'donor_id'
        :type patient_col: str, optional
        :param label_col: column name that identifies disease labels, defaults to 'reported_diseases'
        :type label_col: str, optional
        :param groub_by: column name that identifies cell clusters that should be used instead of single cells, defaults to None
        :type groub_by: str, optional
        :param use_rep: key that identifies low-dimensional represention of the cells in .obsm (like "X_pca") that should be used instead of .X, defaults to None
        :type use_rep: str, optional
        :param subsample_patient_ratio: fraction of patients to randomly subsample, defaults to 1
        :type subsample_patient_ratio: float, optional
        :param n_cells: number of cells to subsample per patient, set to 0 to keep all cells (only supported for centroids and gaussians), defaults to 1000
        :type n_cells: int, optional
        :param filter_genes: whether to filter out genes with low variance (if True), defaults to True
        :type filter_genes: bool, optional
        :param train_size: fraction of train split. If None, no splitting is done, defaults to None
        :type train_size: float, optional
        :return: generated data (distributions and their labels, points and their labels, disease labels, celltype labels, patient labels)
        :rtype: tuple
        """
        # load data
        if isinstance(adata, str):
            adata = sc.read_h5ad(adata)
        elif not isinstance(adata, AnnData):
            raise Exception("Error: No AnnData or Path provided")

        string_class_labels = np.unique(adata.obs[label_col])

        # detect low variable genes
        if filter_genes and use_rep is None:
            # TODO decide if we also want to filter low variance components in use_rep
            gene_var = np.var(adata.X.toarray(), axis=0)
            thresh = np.mean(gene_var)
            adata = adata[:, gene_var > thresh]
            print(f"keeping {len(adata.var.index)} high variable genes")

        distributions = []
        distributions_class = []
        patient_labels = []
        disease_labels = []
        celltype_node_label = []

        # subsample patients
        unique_patients = np.unique(adata.obs[patient_col])
        unique_patients_subsampled = np.random.choice(
            unique_patients,
            size=int(len(unique_patients) * subsample_patient_ratio),
            replace=False,
        )

        cluster_centers = []
        if group_by is not None:
            cluster_names = np.unique(adata.obs[group_by])
            for cluster in cluster_names:
                cluster_centers.append(
                    np.mean(
                        np.asarray(
                            adata[adata.obs[group_by] == cluster].X.toarray()
                            if use_rep is None
                            else adata[adata.obs[group_by] == cluster].obsm[use_rep],
                            dtype="f",
                        ),
                        axis=0,
                    )
                )
            cluster_centers = np.asarray(cluster_centers)

        # iterate through each patient, store disease labels etc., subsample cells per patient
        patient_level_labels = []
        for patient in unique_patients_subsampled:
            patient_adata = adata[adata.obs[patient_col] == patient]
            disease_label = np.unique(patient_adata.obs[label_col].to_numpy())
            string_class_label = disease_label[0]
            patient_level_labels.append(disease_label)

            if len(disease_label) > 1:
                print(
                    "Warning, sample_ids refer to cells with multiple disease labels (likely caused by referencing by patients and having multiple samples from different zones)"
                )

            if patient_adata.n_obs >= n_cells:
                if group_by is None:
                    sc.pp.subsample(patient_adata, n_obs=n_cells)

                if group_by is None:
                    distributions.append(
                        np.asarray(
                            patient_adata.X.toarray()
                            if use_rep is None
                            else patient_adata.obsm[use_rep],
                            dtype="f",
                        )
                    )
                else:
                    # hardcoded own counting function to enforce identical order of clusters for all patients (neither np.unique with counts nor pandas.value_counts do this)
                    distributions.append(
                        np.asarray(
                            [
                                len(
                                    patient_adata[
                                        patient_adata.obs[group_by] == cluster
                                    ]
                                )
                                for cluster in cluster_names
                            ]
                        )
                        / len(patient_adata)
                    )

                disease_labels.append(string_class_label)
                distributions_class.append(
                    np.where(string_class_labels == string_class_label)[0][0]
                )
                patient_labels.append(patient)
                celltype_node_label.append(list(patient_adata.obs["cell_type"]))

        # collect individual points and their labels from the distributions
        # TODO decide how to handle this for clusters
        points = np.concatenate(distributions)
        point_labels = sum(
            [[label] * len(D) for label, D in zip(disease_labels, distributions)], []
        )
        point_patient_labels = sum(
            [[label] * len(D) for label, D in zip(patient_labels, distributions)], []
        )

        # to track in other methods which representation to use
        adata = adata.copy()
        adata.uns["use_rep_GGML"] = use_rep

        if train_size is None:
            # return full dataset exactly as before
            return (
                distributions,
                distributions_class,
                points,
                point_labels,
                disease_labels,
                celltype_node_label,
                patient_labels,
                cluster_centers,
                adata,
                point_patient_labels,
            )
        else:
            # shuffle unique patients for splitting
            # shuffled_patients = np.random.permutation(unique_patients_subsampled)
            # n_train = int(len(shuffled_patients) * train_size)
            # train_patients = shuffled_patients[:n_train]
            # test_patients = shuffled_patients[n_train:]
            test_size = 1 - train_size
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=train_size, test_size=test_size
            )

            for train_idx, test_idx in sss.split(
                unique_patients_subsampled, patient_level_labels
            ):
                train_patients = unique_patients_subsampled[train_idx]
                test_patients = unique_patients_subsampled[test_idx]

            # helper to filter data by patients from existing full data arrays
            def filter_by_patients(patients):
                # filter points and labels by mask
                patient_mask = [p in patients for p in point_patient_labels]
                filtered_points = points[patient_mask]
                filtered_point_labels = np.array(point_labels)[patient_mask].tolist()
                filtered_point_patient_labels = np.array(point_patient_labels)[
                    patient_mask
                ].tolist()

                # filter for remaining return variables
                filtered_indices = [
                    i for i, p in enumerate(patient_labels) if p in patients
                ]
                filtered_distributions = [distributions[i] for i in filtered_indices]
                filtered_distributions_class = [
                    distributions_class[i] for i in filtered_indices
                ]
                filtered_patient_labels = [patient_labels[i] for i in filtered_indices]
                filtered_disease_labels = [disease_labels[i] for i in filtered_indices]
                filtered_celltype_node_label = [
                    celltype_node_label[i] for i in filtered_indices
                ]

                return (
                    filtered_distributions,
                    filtered_distributions_class,
                    filtered_points,
                    filtered_point_labels,
                    filtered_disease_labels,
                    filtered_celltype_node_label,
                    filtered_patient_labels,
                    cluster_centers,
                    adata,
                    filtered_point_patient_labels,
                )

            train_data = filter_by_patients(train_patients)
            test_data = filter_by_patients(test_patients)

            return train_data, test_data

    def compute_OT_on_dists(
        self,
        precomputed_distances=None,
        ground_metric=None,
        w=None,
        legend="Side",
        plot=True,
        symbols=None,
        n_threads=1,
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
            self.distributions,
            precomputed_distances=precomputed_distances,
            ground_metric=ground_metric,
            cluster_centers=self.cluster_centers,
            w=w,
            n_threads=n_threads,
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
