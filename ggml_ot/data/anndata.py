import numpy as np
import anndata as ad
from sklearn import mixture

from ggml_ot.data.generic import TripletDataset


class AnnData_TripletDataset(TripletDataset):
    """Dataset to train GGML based on AnnData.

    This subclass of TripletDataset formats triplets of patient-level cell distributions from an AnnData object.
    The triplets capture the relative relationship between patient groups (e.g. disease state) that GGML aims to learn.

    By default, it captures the cells of a patient as a empirical distribution in the gene space of the AnnData (`.X`).
    Using the ``use_rep`` and/or ``group_by`` parameter, you can reduce the distribution to only cell_subtypes
    and/or low dimensional gene representations.

    This class exposes the dataset to the standardized interfaces used by :meth:`ggml_ot.train`, :meth:`ggml_ot.tune`,
    :meth:`ggml_ot.test` and :meth:`ggml_ot.train_test`.

    Parameters
    ----------
    adata : str | anndata.AnnData
        The AnnData object.
    patient_col : str, optional
        Column in ``adata.obs`` that identifies the patient / sample (default: "sample").
    label_col : str, optional
        Column in ``adata.obs`` that contains the patient group, e.g., disease state (default: "patient_group").
    n_cells : int, optional
        Number of cells to sample per patient (default: 250).
    n_triplets : int, optional
        Number of generated triplets for each patient to capture the relative relationship of the patient group. (default: 3).
        This will lead to ``n_patients * n_triplets * n_labels`` triplets being generated.
    group_by : None | str, optional
        Optional column in ``adata.obs`` to group cells and learn a ground metric between cell groups instead (default: None).
    use_rep : None | str, optional
        If provided, uses ``adata.obsm[use_rep]`` as the cell embedding representation;
        otherwise the raw .X matrix is used (default: None).
    celltype_col : str, optional
        Column in ``adata.obs`` with cell-type annotations (if used) (default: "cell_type").

    See also
    --------
    :class:`ggml_ot.data.generic.TripletDataset`: base class providing triplet creation and dataset API.

    """

    supports: list[np.ndarray]
    "Per-patient distribution supports, by default the cells of the patients. If group_by is passed, it is the centroids of the cell groups."
    weights: list[np.ndarray] | None
    "Probability for each support, by default uniform over the cells of the patients. If group_by is passed, it is the proportion of the cell groups."
    distribution_labels: np.ndarray
    "Patient group labels per patient-level distribution as int, references unique classes from ``adata.obs[label_col]``."
    identical_supports: bool
    "If True, indicates supports were forced identical across distributions by group_by. This significantly speeds up computation, but potential impacts performance."

    def __init__(
        self,
        adata,
        patient_col="sample",
        label_col="patient_group",
        n_cells=250,
        n_triplets=3,
        group_by=None,
        use_rep=None,
        celltype_col="cell_type",
        # experimental
        n_components=None,
        covariance_type="full",
    ):
        supports, covariances, weights, distribution_labels, adata, index_mask = process_anndata(
            adata=adata,
            patient_col=patient_col,
            label_col=label_col,
            group_by=group_by,
            n_components=n_components,
            covariance_type=covariance_type,
            use_rep=use_rep,
            n_cells=n_cells,
        )
        identical_supports = not (group_by is None and n_components is None)

        super().__init__(
            supports,
            distribution_labels,
            n_triplets,
            weights,
            covariances,
            identical_supports,
        )
        self.adata = adata
        self.index_mask = index_mask
        self.label_col = label_col
        self.patient_col = patient_col
        self.celltype_col = celltype_col
        self.use_rep = use_rep

    @property
    def points_labels_str(self):
        "Patient group labels per cell as string, taken from ``adata.obs[label_col]``."
        points_labels = np.concatenate(
            [np.full(len(D), label) for label, D in zip(self.distribution_labels_str, self.supports)]
        )
        return points_labels

    @property
    def distribution_labels_str(self):
        "Patient group labels per patient-level distribution as string, taken from ``adata.obs[label_col]``."
        string_class_labels = np.unique(self.adata.obs[self.label_col])
        return string_class_labels[self.distribution_labels]

    @property
    def celltype_points_labels(self):
        celltype_points_labels = [
            list(self.adata.obs[self.celltype_col].iloc[self.index_mask][i : i + len(support)])
            for i, support in enumerate(self.supports)
        ]
        return celltype_points_labels

    @property
    def patient_labels(self):
        if self.group_by is not None:
            return list(self.adata.obs[self.patient_col].unique())
        else:
            patient_labels = []
            idx = 0
            all_patients = self.adata.obs[self.patient_col].to_numpy()
            for support in self.supports:
                unique_patient = np.unique(all_patients[self.index_mask][idx : idx + len(support)])[0]
                patient_labels.append(unique_patient)
                idx += len(support)
            return patient_labels

    def project(self, w_theta):
        "Store learned ground metric in ``.adata.varm['W_ggml']`` and project cells into gene subspace ``self.adata.obsm['X_ggml']``."
        self.adata.uns["W_ggml"] = w_theta
        if self.use_rep is None:
            self.adata.obsm["X_ggml"] = self.adata.X @ np.transpose(w_theta)
            self.adata.varm["W_ggml"] = np.transpose(w_theta)
        else:
            self.adata.obsm["X_ggml"] = self.adata.obsm[self.use_rep] @ np.transpose(w_theta)


def process_anndata(
    adata,
    patient_col,
    label_col,
    group_by,
    n_components,
    covariance_type,
    use_rep,
    n_cells,
):
    # verify inputs
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)
    elif isinstance(adata, ad.AnnData):
        pass
    else:
        raise Exception("Error: No AnnData or Path provided")

    if group_by is not None:
        assert group_by in adata.obs

    string_class_labels = np.unique(adata.obs[label_col])
    index_mask = np.zeros(adata.n_obs, dtype=bool)
    unique_patients = np.unique(adata.obs[patient_col])

    supports, covariances, weights, labels = [], [], [], []

    # Clustering of Supports across Samples = Identical Supports
    if group_by is not None and n_components is None:
        cluster_names = np.unique(adata.obs[group_by])
        for cluster in cluster_names:
            supports.append(
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
        supports = np.asarray(supports)

    # GMM across Samples = Identical Supports (Same Components for all Patients)
    if n_components is not None:
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        print(
            f"fitting {n_components} comps to {len(adata)} cells using {'.X' if use_rep is None else use_rep} with dim {adata.X.toarray().shape[-1] if use_rep is None else adata.obsm[use_rep].shape[-1]}"
        )
        gmm.fit(adata.X.toarray() if use_rep is None else adata.obsm[use_rep])

        adata.obs["gmm_component"] = gmm.predict(adata.X.toarray() if use_rep is None else adata.obsm[use_rep]).astype(
            "str"
        )
        group_by = "gmm_component"
        cluster_names = np.unique(adata.obs[group_by])

        supports = np.asarray(gmm.means_)
        covariances = np.asarray(gmm.covariances_)

    for patient in unique_patients:
        patient_indices = np.where(adata.obs[patient_col] == patient)[0]
        patient_adata = adata[patient_indices]
        disease_label = np.unique(patient_adata.obs[label_col].to_numpy())

        if len(disease_label) > 1:
            print(
                "Warning, sample_ids refer to cells with multiple disease labels (likely caused by referencing by patients and having multiple samples from different zones)"
            )

        if patient_adata.n_obs >= n_cells:
            if group_by is None and n_components is None:
                replace = patient_adata.n_obs < n_cells
                sampled_idx = np.random.choice(patient_indices, size=n_cells, replace=replace)
                index_mask[sampled_idx] = True

                if use_rep is None:
                    patient_points = adata.X[sampled_idx, :].toarray()
                else:
                    patient_points = adata.obsm[use_rep][sampled_idx]

                supports.append(np.asarray(patient_points, dtype="f"))

            else:
                # per-patient cluster proportions
                cluster_means = np.asarray(
                    [len(patient_adata[patient_adata.obs[group_by] == cluster]) for cluster in cluster_names]
                ) / len(patient_adata)
                weights.append(cluster_means)

            labels.append(np.where(string_class_labels == disease_label)[0][0])
        else:
            print(
                f"Patient {patient} has only {patient_adata.n_obs} cells which is less than specified n_cells={n_cells}, skipping"
            )

    adata = adata.copy()
    adata.uns["use_rep_GGML"] = use_rep
    adata.uns["data_type"] = "Patient_level"

    if len(covariances) == 0:
        covariances = None

    if len(weights) == 0:
        weights = None

    return supports, covariances, weights, labels, adata, index_mask


def get_cluster_centers():
    pass


def fit_GMM():
    pass
