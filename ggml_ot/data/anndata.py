import numpy as np
import scipy.sparse as sp
import warnings

from ggml_ot import settings
from ggml_ot.data.generic import TripletDataset
from ggml_ot._utils._docs import get_defaults
from ggml_ot.gmm._anndata_dataset_interface import gmm_from_anndata as _gmm_from_anndata
from ggml_ot._utils._anndata_utils import (
    extract_centroids,
    label_index_map,
    patient_labels,
    validate_anndata_inputs,
)
from ggml_ot.data._parameter_setup import set_preprocessed
from ggml_ot._utils._weights import normalize_weights


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
    gmm_key : None | str, optional
        If provided, loads a previously fitted GMM representation from ``adata.uns[gmm_key]`` (default: None).
    sample_gmm : bool, optional
        If ``True``, samples empirical supports from fitted GMM mixtures instead of using parametric supports directly
        (default: False).
    gmm_weights_source : {"auto", "stored", "components"}, optional
        Controls how per-distribution GMM weights are reconstructed when ``gmm_key`` is provided.
        ``"auto"`` tries stored weights first, then hard assignments predicted from the stored GMM
        parameters (default: "auto").

    Notes
    -----
    Following scverse conventions, this class **modifies the AnnData object
    in-place** during dataset construction and training.  In particular:

    - ``adata.uns["ggml_params"]`` — stores dataset construction parameters.
    - ``adata.uns["W_ggml"]`` — the learned linear map after training.
    - ``adata.varm["W_ggml"]`` — gene-space loadings of the learned ground metric.
    - ``adata.obsm["X_ggml"]`` — cells projected into the learned gene subspace.

    If you need an unmodified copy, call ``adata.copy()`` before constructing
    the dataset.

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
        use_rep=None,
        group_by=None,
        gmm_key=None,
        sample_gmm=False,
        gmm_weights_source="auto",
    ):
        kwargs = {**locals()}
        for key in ["self", "adata", "__class__"]:
            kwargs.pop(key)

        if settings.restore_adata_params:
            defaults = get_defaults(type(self).__init__)
            if "ggml_params" in adata.uns:
                stored = dict(adata.uns["ggml_params"])
                warning_raised = False
                for param, default_value in defaults.items():
                    if (
                        param in kwargs
                        and kwargs[param] == default_value
                        and param in stored
                        and kwargs[param] != stored[param]
                    ):
                        kwargs[param] = stored[param]
                        if not warning_raised:
                            warnings.warn(
                                "Restoring previously stored ggml_params for AnnData_TripletDataset initialization (ggml_ot.settings.restore_adata_params = True). Set to False to disable this behavior."
                            )
                            warning_raised = True

        adata.uns["ggml_params"] = kwargs
        adata = _process_anndata(
            adata,
            patient_col=kwargs["patient_col"],
            label_col=kwargs["label_col"],
            use_rep=kwargs["use_rep"],
            group_by=kwargs["group_by"],
            gmm_key=kwargs["gmm_key"],
            n_cells=kwargs["n_cells"],
            sample_gmm=kwargs["sample_gmm"],
            gmm_weights_source=kwargs["gmm_weights_source"],
        )

        super().__init__(**adata.uns["ggml_preprocessed"], n_triplets=kwargs["n_triplets"])
        self.adata = adata
        self.index_mask = adata.uns["ggml_preprocessed"]["index_mask"]
        self.label_col = kwargs["label_col"]
        self.patient_col = kwargs["patient_col"]
        self.use_rep = adata.uns["ggml_preprocessed"]["use_rep"]
        self.group_by = kwargs["group_by"]
        if "W_ggml" in self.adata.uns:
            self._map_A = self.adata.uns["W_ggml"]

    @property
    def points_labels_str(self):
        "Patient group labels per cell as string, taken from ``adata.obs[label_col]``."
        points_labels = np.concatenate(
            [np.full(len(d), label) for label, d in zip(self.distribution_labels_str, self.supports)]
        )
        return points_labels

    @property
    def distribution_labels_str(self):
        "Patient group labels per patient-level distribution as string, taken from ``adata.obs[label_col]``."
        string_class_labels = np.unique(self.adata.obs[self.label_col])
        return string_class_labels[self.distribution_labels]

    @property
    def patient_labels(self):
        if self.group_by is not None:
            return list(self.adata.obs[self.patient_col].unique())

        patient_labels = []
        idx = 0
        all_patients = self.adata.obs[self.patient_col].to_numpy()
        for support in self.supports:
            unique_patient = np.unique(all_patients[self.index_mask][idx : idx + len(support)])[0]
            patient_labels.append(unique_patient)
            idx += len(support)
        return patient_labels

    @TripletDataset.map_A.setter
    def map_A(self, map_A):
        """Store learned ground metric and project cells into the gene subspace.

        Writes into the AnnData object in-place (see class-level Notes):
        ``adata.uns["W_ggml"]``, ``adata.varm["W_ggml"]``, ``adata.obsm["X_ggml"]``.
        """
        self._map_A = map_A
        self.adata.uns["W_ggml"] = map_A

        if map_A is not None:
            if self.use_rep is None:
                self.adata.obsm["X_ggml"] = self.adata.X @ np.transpose(map_A)
                self.adata.varm["W_ggml"] = np.transpose(map_A)
            else:
                self.adata.obsm["X_ggml"] = self.adata.obsm[self.use_rep] @ np.transpose(map_A)
                if self.use_rep in self.adata.varm.keys():
                    self.adata.varm["W_ggml"] = self.adata.varm[self.use_rep] @ np.transpose(map_A)
                elif "pca" in self.use_rep and "PCs" in self.adata.varm.keys():
                    self.adata.varm["W_ggml"] = self.adata.varm["PCs"] @ np.transpose(map_A)
                else:
                    warnings.warn(
                        f"Cannot project W_ggml back to gene space, since use_rep {self.use_rep} has no inverse transform (no .varm[{self.use_rep}]). GGML components only stored in .uns, and not in .varm"
                    )

    def to_anndata(self):
        "Returns the AnnData of the dataset object."
        return self.adata

    def fit_gmm(self, *args, **kwargs):
        """Fit or refit a GMM representation for this AnnData-backed dataset.

        Delegates to :func:`ggml_ot.gmm.fit_gmm` via the base class wrapper.
        """
        return super().fit_gmm(*args, **kwargs)

    # -------------------------------------------------------------------
    # Gene interpretation — axis level
    # -------------------------------------------------------------------

    def rank_latent_axes(self, *, axes=None, gene_symbols=None):
        """Rank genes by their contribution to each GGML latent axis.

        Thin wrapper around :func:`ggml_ot.gene.rank_latent_axes`.

        Parameters
        ----------
        axes : list[int] | None
            0-indexed axis indices.  ``None`` ranks all axes.
        gene_symbols : str | None
            Column in ``adata.var`` for gene names.

        Returns
        -------
        pandas.DataFrame
            Long-format table: ``axis``, ``gene``, ``score``, ``abs_score``,
            ``sign``, ``rank``.
        """
        from ..gene._axis import rank_latent_axes as _rank

        return _rank(self.adata, axes=axes, gene_symbols=gene_symbols)

    def enrich_latent_axes(
        self,
        *,
        axes=None,
        gene_symbols=None,
        method="gsea",
        backend="decoupler",
        resource="MSigDB",
        collection="hallmark",
        organism="human",
        min_n=5,
    ):
        """Run pathway enrichment on per-axis gene rankings.

        Thin wrapper that calls :func:`~ggml_ot.gene._axis.rank_latent_axes`
        followed by :func:`~ggml_ot.gene._decoupler.run_enrichment`.

        Uses ``ggml_ot.settings.random_seed`` for reproducibility.

        Parameters
        ----------
        axes : list[int] | None
            0-indexed axis indices.  ``None`` uses all axes.
        gene_symbols : str | None
            Column in ``adata.var`` for gene names.
        method : str
            Enrichment method (default ``"gsea"``).
        backend : str
            Enrichment backend (default ``"decoupler"``).
        resource : str
            Resource name for ``dc.get_resource()`` (default ``"MSigDB"``).
        collection : str
            Collection filter (default ``"hallmark"``).
        organism : str
            Organism for resource retrieval (default ``"human"``).
        min_n : int
            Minimum gene-set size (default 5).

        Returns
        -------
        pandas.DataFrame
            Long-format enrichment results: ``axis``, ``pathway``,
            ``score``, ``norm_score``, ``pvalue``.
        """
        from ..gene._axis import rank_latent_axes as _rank

        score_df = _rank(self.adata, axes=axes, gene_symbols=gene_symbols)

        if backend == "decoupler":
            from ..gene._decoupler import run_enrichment

            return run_enrichment(
                score_df,
                group_col="axis",
                method=method,
                resource=resource,
                collection=collection,
                organism=organism,
                min_n=min_n,
            )
        else:
            raise ValueError(f"Unknown enrichment backend: {backend!r}. Use 'decoupler'.")

    # -------------------------------------------------------------------
    # Gene interpretation — GMM component level
    # -------------------------------------------------------------------

    def summarize_gmm_components(
        self,
        *,
        gmm_key=None,
        groupby=None,
        weighting="responsibility",
        normalize="component",
        grouping_method=None,
        n_groups=None,
        group_representative="mean",
        barycenter_weighting="component_weights",
        grouping_key=None,
    ):
        """Summarize GMM components by cell-type composition and patient weights.

        Parameters
        ----------
        gmm_key : str | None
            GMM key in ``adata.uns``.  ``None`` auto-resolves from ``use_rep``.
        groupby : str | None
            Column in ``adata.obs`` for cell-type labels.  Required.
        weighting : str
            ``"responsibility"`` or ``"hard"`` (default ``"responsibility"``).
        normalize : str
            ``"component"``, ``"cell_type"``, or ``"none"`` (default ``"component"``).
        grouping_method : str | None
            ``"mean"`` or ``"bures-wasserstein"`` to group sample-specific components.
            ``None`` requires a globally shared GMM.
        n_groups : int | None
            Number of component groups when aggregating.  Defaults to K.
        group_representative : str
            Stored grouped representative type (default ``"mean"``).
        barycenter_weighting : str
            Weighting used for grouped Gaussian representatives (default
            ``"component_weights"``).
        grouping_key : str | None
            Explicit key for reading / writing grouped component results.

        Returns
        -------
        dict
            ``"celltype_table"``, ``"patient_weights"``, ``"purity"``
            — each a :class:`pandas.DataFrame`.
            When grouped summaries are requested, also includes ``"grouping"``.
        """
        if groupby is None:
            raise ValueError("groupby is required (e.g. groupby='cell_type').")

        from ..gene._gmm_summary import summarize_gmm_components as _summarize

        return _summarize(
            self,
            gmm_key=gmm_key,
            groupby=groupby,
            weighting=weighting,
            normalize=normalize,
            grouping_method=grouping_method,
            n_groups=n_groups,
            group_representative=group_representative,
            barycenter_weighting=barycenter_weighting,
            grouping_key=grouping_key,
        )

    def enrich_gmm_components(
        self,
        *,
        gmm_key=None,
        contrast="latent_mean_shift",
        reference="rest",
        gene_symbols=None,
        method="gsea",
        backend="decoupler",
        resource="MSigDB",
        collection="hallmark",
        organism="human",
        min_n=5,
        grouping_method=None,
        n_groups=None,
        group_representative="mean",
        barycenter_weighting="component_weights",
        grouping_key=None,
    ):
        """Run pathway enrichment on GMM component gene signatures.

        Uses a latent-space contrast to derive per-component gene scores,
        then runs enrichment via the specified backend.

        Uses ``ggml_ot.settings.random_seed`` for reproducibility.

        Parameters
        ----------
        gmm_key : str | None
            GMM key in ``adata.uns``.  ``None`` auto-resolves from ``use_rep``.
        contrast : str
            Contrast method (default ``"latent_mean_shift"``).
        reference : str
            ``"rest"`` or ``"global_mean"`` (default ``"rest"``).
        gene_symbols : str | None
            Column in ``adata.var`` for gene names (used in enrichment matching).
        method : str
            Enrichment method (default ``"gsea"``).
        backend : str
            Enrichment backend (default ``"decoupler"``).
        resource : str
            Resource for ``dc.get_resource()`` (default ``"MSigDB"``).
        collection : str
            Collection filter (default ``"hallmark"``).
        organism : str
            Organism (default ``"human"``).
        min_n : int
            Minimum gene-set size (default 5).
        grouping_method : str | None
            ``"mean"`` or ``"bures-wasserstein"`` to group sample-specific components.
            ``None`` requires a globally shared GMM.
        n_groups : int | None
            Number of component groups when aggregating.  Defaults to K.
        group_representative : str
            Stored grouped representative type (default ``"mean"``).
        barycenter_weighting : str
            Weighting used for grouped Gaussian representatives (default
            ``"component_weights"``).
        grouping_key : str | None
            Explicit key for reading / writing grouped component results.

        Returns
        -------
        pandas.DataFrame
            Long-format: ``component``, ``pathway``, ``score``,
            ``norm_score``, ``pvalue``.
        """
        from ..gene._gmm_summary import component_gene_scores

        score_df = component_gene_scores(
            self,
            gmm_key=gmm_key,
            contrast=contrast,
            reference=reference,
            gene_symbols=gene_symbols,
            grouping_method=grouping_method,
            n_groups=n_groups,
            group_representative=group_representative,
            barycenter_weighting=barycenter_weighting,
            grouping_key=grouping_key,
        )

        if backend == "decoupler":
            from ..gene._decoupler import run_enrichment

            return run_enrichment(
                score_df,
                group_col="component",
                method=method,
                resource=resource,
                collection=collection,
                organism=organism,
                min_n=min_n,
            )
        else:
            raise ValueError(f"Unknown enrichment backend: {backend!r}. Use 'decoupler'.")

    # -------------------------------------------------------------------
    # Deferred — planned for a future version
    # -------------------------------------------------------------------

    def rotate_latent_axes(self, method="varimax", **kwargs):
        """Apply post-hoc rotation to latent axes (varimax, promax).

        .. note:: Not yet implemented. Planned for a future release.
        """
        raise NotImplementedError(
            f"rotate_latent_axes(method={method!r}) is planned but not yet implemented. "
            "See gene_enrich_plan.md Phase 7 for details."
        )

    def enrich_discriminant(self, **kwargs):
        """Enrich a supervised discriminant direction in latent space.

        .. note:: Not yet implemented. Planned for a future release.
        """
        raise NotImplementedError(
            "enrich_discriminant() is planned but not yet implemented. See gene_enrich_plan.md Phase 7 for details."
        )


InitDefaults = AnnData_TripletDataset.__init__.__kwdefaults__


# ---------------------------------------------------------------------------
# AnnData preprocessing strategies
# ---------------------------------------------------------------------------


def _centroids_from_anndata(adata, patient_col, label_col, use_rep, group_by):
    if group_by not in adata.obs:
        raise ValueError(f"group_by {group_by} not in adata.obs columns {adata.obs.columns}")

    supports = extract_centroids(adata, group_by, use_rep)
    ordered_components = np.unique(adata.obs[group_by].to_numpy())
    _, label_to_idx = label_index_map(adata, label_col)

    weights = []
    labels = []
    for patient in patient_labels(adata, patient_col):
        patient_adata = adata[adata.obs[patient_col] == patient]
        label = np.unique(patient_adata.obs[label_col].to_numpy())
        if len(label) > 1:
            warnings.warn(f"Cells from one sample {patient_col}={patient} contain multiple labels {label_col}={label}")

        component_counts = np.array(
            [len(patient_adata[patient_adata.obs[group_by] == c]) for c in ordered_components], dtype=np.float64
        )
        weights.append(normalize_weights(component_counts))
        labels.append(label_to_idx[label[0]])

    set_preprocessed(
        adata,
        supports=np.asarray(supports, dtype="f"),
        covariances=None,
        weights=weights,
        distribution_labels=labels,
        identical_supports=True,
        use_rep=use_rep,
    )
    return adata


def _empirical_from_anndata(adata, patient_col, label_col, use_rep, n_cells):
    supports = []
    sampled_cells_mask = np.zeros(adata.n_obs, dtype=bool)
    _, label_to_idx = label_index_map(adata, label_col)
    labels = []

    for patient in patient_labels(adata, patient_col):
        patient_indices = np.where(adata.obs[patient_col] == patient)[0]
        patient_adata = adata[patient_indices]
        label = np.unique(patient_adata.obs[label_col].to_numpy())
        if len(label) > 1:
            warnings.warn(f"Cells from one sample {patient_col}={patient} contain multiple labels {label_col}={label}")

        if patient_adata.n_obs >= n_cells:
            sampled_idx = settings.numpy_generator.choice(patient_indices, size=n_cells, replace=False)
            sampled_cells_mask[sampled_idx] = True

            if use_rep is None:
                patient_points = adata.X[sampled_idx, :]
                if sp.issparse(patient_points):
                    patient_points = patient_points.toarray()
            else:
                patient_points = adata.obsm[use_rep][sampled_idx]

            supports.append(np.asarray(patient_points, dtype="f"))
            labels.append(label_to_idx[label[0]])
        else:
            print(f"Skip {patient} with {patient_adata.n_obs} cells (< n_cell = {n_cells})")

    set_preprocessed(
        adata,
        supports=supports,
        covariances=None,
        weights=None,
        distribution_labels=labels,
        identical_supports=False,
        use_rep=use_rep,
        index_mask=sampled_cells_mask,
    )
    return adata


def _process_anndata(
    adata, patient_col, label_col, use_rep, group_by, gmm_key, n_cells, sample_gmm, gmm_weights_source
):
    validate_anndata_inputs(adata, patient_col, label_col, use_rep)

    if gmm_key is not None:
        # ``gmm_key`` takes precedence: we are loading an already-fitted representation.
        if group_by is not None:
            warnings.warn(
                "Both parameters group_by and gmm_key provided, ignoring group_by. "
                "Set group_by to None to avoid this warning, or set gmm_key to None to use group_by."
            )
        return _gmm_from_anndata(
            adata, patient_col, label_col, use_rep, gmm_key, n_cells, sample_gmm, gmm_weights_source
        )

    if gmm_weights_source != "auto":
        raise ValueError("gmm_weights_source is only supported when gmm_key is provided.")

    if group_by is not None:
        return _centroids_from_anndata(adata, patient_col, label_col, use_rep, group_by)

    return _empirical_from_anndata(adata, patient_col, label_col, use_rep, n_cells)
